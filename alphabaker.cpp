#include "alphabaker.h"

#include <stdio.h>
#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <stdarg.h>
#include <assert.h>

#include "external/ufbx.h"
#include "external/ufbx_os.h"
#include "external/ufbx_stl.h"
#include "external/umath.h"
#include "external/rtk.h"
#include "external/stb_image.h"
#include "external/stb_image_write.h"

#include <string_view>
#include <unordered_map>
#include <optional>
#include <memory>
#include <mutex>
#include <chrono>
#include <algorithm>
#include <functional>

template <> struct ufbx_converter<um_vec2> {
	static inline um_vec2 from(const ufbx_vec2 &v) { return { (float)v.x, (float)v.y }; }
};
template <> struct ufbx_converter<um_vec3> {
	static inline um_vec3 from(const ufbx_vec3 &v) { return { (float)v.x, (float)v.y, (float)v.z }; }
};

template<> struct std::default_delete<rtk_scene> {
	void operator()(rtk_scene *ptr) const { rtk_free_scene(ptr); }
};

template<> struct std::default_delete<ufbx_os_thread_pool> {
	void operator()(ufbx_os_thread_pool *ptr) const { ufbx_os_free_thread_pool(ptr); }
};

static um_vec3 from_rtk(const rtk_vec3 &v) { return { v.x, v.y, v.z }; }
static rtk_vec3 to_rtk(const um_vec3 &v) { return { v.x, v.y, v.z }; }

AlphaBaker_Result fatalf(AlphaBaker_Result result, const char *fmt, ...)
{
	va_list args;
	va_start(args, fmt);
	fprintf(stderr, "Error: ");
	vfprintf(stderr, fmt, args);
	fprintf(stderr, "\n\n");
	va_end(args);

	return result;
}

std::nullopt_t failf(const char *fmt, ...)
{
	va_list args;
	va_start(args, fmt);
	fprintf(stderr, "\n");
	vfprintf(stderr, fmt, args);
	fprintf(stderr, "\n");
	va_end(args);

	return std::nullopt;
}

template <typename T>
T clamp(T value, T min_v, T max_v) { return std::min(std::max(value, min_v), max_v); }

struct BakeEntry
{
	ufbx_node *high_node = nullptr;
	ufbx_node *low_node = nullptr;
};

struct Vertex
{
	um_vec3 position;
	um_vec3 normal;
	um_vec2 uv;
};

struct Mesh
{
	bool ok;
	std::vector<Vertex> vertices;
	std::vector<uint32_t> indices;
};

struct Target
{
	std::vector<float> data;
	size_t width = 0;
	size_t height = 0;

	Target() { }
	Target(size_t width, size_t height)
		: width(width), height(height)
	{
		data.resize(width * height);
	}

	float &at(size_t x, size_t y) {
		assert(x < width && y < height);
		return data[y * width + x];
	}
	const float &at(size_t x, size_t y) const {
		assert(x < width && y < height);
		return data[y * width + x];
	}
};

struct PixelRegion
{
	size_t min_x, min_y;
	size_t max_x, max_y;
};

struct TraceTile
{
	PixelRegion region;
	size_t progress_count;
};

struct TraceState
{
	const Mesh &low_mesh;
	const rtk_scene *low_uv_scene;
	const rtk_scene *high_scene;

	std::atomic_uint32_t tile_index;
	std::vector<TraceTile> tiles;

	PixelRegion region;
};

struct Tracer
{
	static constexpr size_t tile_size = 32;

	Target target;
	size_t samples;
	std::vector<um_vec2> sample_offsets;

	uint32_t thread_count;
	size_t tiles_x;
	size_t tiles_y;

	float ray_dist_front;
	float ray_dist_back;

	std::unique_ptr<ufbx_os_thread_pool> thread_pool;
	std::mutex print_mutex;
};

template <typename Task>
uint64_t start_tasks(Tracer &tracer, uint32_t count, Task task)
{
	auto entry = [](void *user, uint32_t index) { (*(const Task*)user)(index); };
	return ufbx_os_thread_pool_run(tracer.thread_pool.get(), entry, (void*)&task, count);
}

void wait_tasks(Tracer &tracer, uint64_t id)
{
	ufbx_os_thread_pool_wait(tracer.thread_pool.get(), id);
}

template <typename Task>
void run_threaded(Tracer &tracer, Task task)
{
	uint64_t id = start_tasks(tracer, tracer.thread_count, task);
	wait_tasks(tracer, id);
}

std::optional<Mesh> create_mesh(ufbx_node *node)
{
	ufbx_mesh *mesh = node->mesh;
	if (!mesh) return failf("Node does not contain a mesh");

	ufbx_matrix geometry_to_world = node->geometry_to_world;
	ufbx_matrix normal_to_world = ufbx_matrix_for_normals(&geometry_to_world);

	std::vector<uint32_t> tri_indices(mesh->max_face_triangles * 3);

	Mesh result;

	for (ufbx_face face : mesh->faces) {
		size_t num_tris = ufbx_triangulate_face(tri_indices.data(), tri_indices.size(), mesh, face);

		for (size_t i = 0; i < num_tris * 3; i++) {
			uint32_t ix = tri_indices[i];

			Vertex v = { };
			v.position = ufbx_transform_position(&geometry_to_world, ufbx_get_vertex_vec3(&mesh->vertex_position, ix));
			v.normal = ufbx_transform_direction(&normal_to_world, ufbx_get_vertex_vec3(&mesh->vertex_normal, ix));
			v.normal = um_normalize3(v.normal);

			if (mesh->vertex_uv.exists) {
				v.uv = ufbx_get_vertex_vec2(&mesh->vertex_uv, ix);
			}
			result.vertices.push_back(v);
		}
	}

	ufbx_vertex_stream stream = { };
	stream.data = result.vertices.data();
	stream.vertex_count = result.vertices.size();
	stream.vertex_size = sizeof(Vertex);

	result.indices.resize(result.vertices.size());

	ufbx_error error;
	size_t num_vertices = ufbx_generate_indices(&stream, 1, result.indices.data(), result.indices.size(), nullptr, &error);
	if (error.type != UFBX_ERROR_NONE) {
		return failf("Failed to generate indices:\n%s", ufbx_format_error_string(error).c_str());
	}

	result.vertices.resize(num_vertices);

	return result;
}

template <uint32_t Base>
float radical_inverse(uint32_t bits)
{
	constexpr double rcp_base = 1.0 / (double)Base;

	double value = 0.0, step = 1.0;
	while (bits) {
		step *= rcp_base;
		value += step * (bits % Base);
		bits /= Base;
	}

	return (float)value;
}

um_vec2 halton_sequence(size_t index, size_t count)
{
	float u = radical_inverse<2>((uint32_t)index);
	float v = radical_inverse<3>((uint32_t)index);
	return { u, v };
}

void trace_tile(Tracer &tracer, TraceState &state, PixelRegion tile)
{
	Target &target = tracer.target;

	float local_values[Tracer::tile_size][Tracer::tile_size] = { };

	for (size_t x = tile.min_x; x < tile.max_x; x++) {
		for (size_t y = tile.min_y; y < tile.max_y; y++) {

			float sum = 0.0f;
			for (size_t i = 0; i < tracer.samples; i++) {
				um_vec2 offset = tracer.sample_offsets[i];

				um_vec2 uv;
				uv.x = ((float)x + offset.x) / (float)target.width;
				uv.y = ((float)y + offset.y) / (float)target.height;

				um_vec3 uv_origin = { uv.x, uv.y, 0.0f };
				um_vec3 uv_direction = { 0.0f, 0.0f, 1.0f };

				rtk_hit uv_hit;
				rtk_ray uv_ray = { to_rtk(uv_origin), to_rtk(uv_direction), -1.0f };
				if (!rtk_raytrace(state.low_uv_scene, &uv_ray, &uv_hit, 1.0f)) continue;

				um_vec3 p0 = state.low_mesh.vertices[uv_hit.vertex_index[0]].position;
				um_vec3 p1 = state.low_mesh.vertices[uv_hit.vertex_index[1]].position;
				um_vec3 p2 = state.low_mesh.vertices[uv_hit.vertex_index[2]].position;

				float w = 1.0f - uv_hit.geom.u - uv_hit.geom.v;
				um_vec3 position = p0 * uv_hit.geom.u + p1 * uv_hit.geom.v + p2 * w;
				um_vec3 normal = um_normalize3(from_rtk(uv_hit.interp.normal));

				um_vec3 high_origin = position;
				um_vec3 high_direction = normal;

				rtk_hit high_hit;
				rtk_ray high_ray = { to_rtk(high_origin), to_rtk(high_direction), -tracer.ray_dist_back };
				if (!rtk_raytrace(state.high_scene, &high_ray, &high_hit, tracer.ray_dist_front)) continue;

				sum += 1.0f;
			}

			local_values[y - tile.min_y][x - tile.min_x] = sum / (float)tracer.samples;
		}
	}

	for (size_t x = tile.min_x; x < tile.max_x; x++) {
		for (size_t y = tile.min_y; y < tile.max_y; y++) {
			float &result = target.at(x, y);
			result = std::max(result, local_values[y - tile.min_y][x - tile.min_x]);
		}
	}
}

void print_progress(Tracer &tracer, size_t &pending_progress, bool force)
{
	std::unique_lock<std::mutex> lock { tracer.print_mutex, std::defer_lock };
	if (force) {
		lock.lock();
	} else {
		if (!lock.try_lock()) return;
	}

	if (lock.owns_lock()) {
		char buffer[64];
		size_t count = pending_progress;
		assert(count < 64);
		for (size_t i = 0; i < count; i++) {
			buffer[i] = '.';
		}
		buffer[count] = '\0';
		printf("%s", buffer);
		fflush(stdout);
		pending_progress = 0;
	}
}

void trace_thread(Tracer &tracer, TraceState &state)
{
	size_t tile_count = state.tiles.size();
	size_t pending_progress = 0;

	for (;;) {
		size_t tile_index = state.tile_index.fetch_add(1u, std::memory_order_relaxed);
		if (tile_index >= tile_count) break;

		const TraceTile &tile = state.tiles[tile_index];
		trace_tile(tracer, state, tile.region);
		pending_progress += tile.progress_count;

		if (pending_progress > 0) {
			print_progress(tracer, pending_progress, false);
		}
	}

	print_progress(tracer, pending_progress, true);
}

std::optional<bool> process_entry(Tracer &tracer, const BakeEntry &entry)
{
	if (!entry.low_node) return failf("Low-poly node not found");
	if (!entry.high_node) return failf("High-poly node not found");

	std::optional<Mesh> low_mesh_opt = create_mesh(entry.low_node);
	if (!low_mesh_opt) return failf("Failed to create low-poly mesh");

	std::optional<Mesh> high_mesh_opt = create_mesh(entry.high_node);
	if (!high_mesh_opt) return failf("Failed to create high-poly mesh");

	Mesh low_mesh = std::move(*low_mesh_opt);
	Mesh high_mesh = std::move(*high_mesh_opt);

	std::unique_ptr<rtk_scene> low_uv_scene;
	std::unique_ptr<rtk_scene> high_scene;

	{
		std::vector<um_vec3> uv_vertices;

		for (const Vertex &v : low_mesh.vertices) {
			uv_vertices.push_back({ v.uv.x, v.uv.y, 0.0f });
		}

		rtk_mesh mesh = { };
		mesh.vertices = (rtk_vec3*)uv_vertices.data();
		mesh.normals = (rtk_vec3*)((char*)low_mesh.vertices.data() + offsetof(Vertex, normal));
		mesh.normals_stride = sizeof(Vertex);
		mesh.uvs = (rtk_vec2*)((char*)low_mesh.vertices.data() + offsetof(Vertex, uv));
		mesh.uvs_stride = sizeof(Vertex);
		mesh.indices = low_mesh.indices.data();
		mesh.num_triangles = low_mesh.indices.size() / 3;
		mesh.transform = rtk_identity;

		rtk_scene_desc desc = { };
		desc.meshes = &mesh;
		desc.num_meshes = 1;

		low_uv_scene.reset(rtk_create_scene(&desc));
		if (!low_uv_scene) return failf("Failed to create high-poly raytracing scene");
	}

	{
		rtk_mesh mesh = { };
		mesh.vertices = (rtk_vec3*)((char*)high_mesh.vertices.data() + offsetof(Vertex, position));
		mesh.vertices_stride = sizeof(Vertex);
		mesh.normals = (rtk_vec3*)((char*)high_mesh.vertices.data() + offsetof(Vertex, normal));
		mesh.normals_stride = sizeof(Vertex);
		mesh.uvs = (rtk_vec2*)((char*)high_mesh.vertices.data() + offsetof(Vertex, uv));
		mesh.uvs_stride = sizeof(Vertex);
		mesh.indices = high_mesh.indices.data();
		mesh.num_triangles = high_mesh.indices.size() / 3;
		mesh.transform = rtk_identity;

		rtk_scene_desc desc = { };
		desc.meshes = &mesh;
		desc.num_meshes = 1;

		high_scene.reset(rtk_create_scene(&desc));
		if (!high_scene) return failf("Failed to create high-poly raytracing scene");
	}

	TraceState state = { low_mesh };
	state.low_uv_scene = low_uv_scene.get();
	state.high_scene = high_scene.get();

	{
		um_vec2 min_uv = um_dup2(+INFINITY);
		um_vec2 max_uv = um_dup2(-INFINITY);

		for (Vertex &v : low_mesh.vertices) {
			min_uv.x = std::min(min_uv.x, v.uv.x);
			min_uv.y = std::min(min_uv.y, v.uv.y);
			max_uv.x = std::max(max_uv.x, v.uv.x);
			max_uv.y = std::max(max_uv.y, v.uv.y);
		}

		float width = (float)tracer.target.width;
		float height = (float)tracer.target.height;
		state.region.min_x = (size_t)clamp(min_uv.x * width - 1.0f, 0.0f, width);
		state.region.min_y = (size_t)clamp(min_uv.y * height - 1.0f, 0.0f, width);
		state.region.max_x = (size_t)clamp(max_uv.x * width + 1.0f, 0.0f, width);
		state.region.max_y = (size_t)clamp(max_uv.y * height + 1.0f, 0.0f, width);
	}

	for (size_t tile_y = 0; tile_y < tracer.tiles_y; tile_y++) {
		for (size_t tile_x = 0; tile_x < tracer.tiles_x; tile_x++) {
			size_t min_x = tile_x * tracer.tile_size; 
			size_t min_y = tile_y * tracer.tile_size; 
			size_t max_x = std::min(min_x + tracer.tile_size, tracer.target.width);
			size_t max_y = std::min(min_y + tracer.tile_size, tracer.target.height);

			if (min_x >= state.region.max_x) continue;
			if (min_y >= state.region.max_y) continue;
			if (max_x <= state.region.min_x) continue;
			if (max_y <= state.region.min_y) continue;

			state.tiles.push_back({ min_x, min_y, max_x, max_y });
		}
	}

	if (state.tiles.empty()) {
		printf("\nWarning: No tiles\n");
		return false;
	}

	const uint32_t total_progress = 32;
	double max_tile = (double)(state.tiles.size() - 1);
	for (uint32_t prog = 0; prog < total_progress; prog++) {
		size_t index = (size_t)clamp((double)prog / (double)(total_progress - 1) * max_tile, 0.0, max_tile);
		state.tiles[index].progress_count++;
	}

	run_threaded(tracer, [&](uint32_t thread_id) {
		trace_thread(tracer, state);
	});

	return true;
}

AlphaBaker_Result save_result(const Target &target, const std::string &path, bool invert_y)
{
	size_t pixel_count = target.width * target.height;
	std::vector<uint8_t> result;
	result.resize(target.width * target.height * 3);

	size_t dst_ix = 0;
	for (size_t y = 0; y < target.height; y++) {
		// Sic: Use top-left by default
		size_t src_y = invert_y ? y : target.height - y - 1;

		for (size_t x = 0; x < target.width; x++) {
			float value = target.at(x, src_y);
			value = std::max(value, 0.0f);
			value = std::min(value, 1.0f);

			uint8_t b = (uint8_t)(value * 255.0f);
			result[dst_ix + 0] = b;
			result[dst_ix + 1] = b;
			result[dst_ix + 2] = b;
			dst_ix += 3;
		}
	}

	if (!stbi_write_png(path.c_str(), (int)target.width, (int)target.height, 3, result.data(), 0)) {
		return fatalf(AlphaBaker_FailedToWriteOutput, "Failed to write output file");
	}
}

static AlphaBaker_String to_option(const char *str)
{
	return { str, (int)strlen(str) };
}

static std::string_view from_option(AlphaBaker_String str)
{
	if (str.length < 0) {
		return { str.data };
	} else {
		return { str.data, (size_t)str.length };
	}
}

extern "C" {

ALPHABAKER_API void AlphaBaker_defaults(AlphaBaker_Options *options)
{
	if (!options) return;
	AlphaBaker_Options &opts = *options;

	memset(options, 0, sizeof(AlphaBaker_Options));

	opts.low_suffix = to_option(" Low");
	opts.high_suffix = to_option(" High");
	opts.resolution = 1024;
	opts.samples = 256;
	opts.ray_dist_front = 1000.0f;
	opts.ray_dist_back = 1000.0f;
	opts.thread_count = -1;
}

ALPHABAKER_API void AlphaBaker_parseOptions(AlphaBaker_Options *options, char **arguments, int count)
{
}

ALPHABAKER_API int AlphaBaker_bake(const AlphaBaker_Options *options)
{
	if (!options) return fatalf(AlphaBaker_InvalidOptions, "Error: No options provided");
	const AlphaBaker_Options &input = *options;

	std::string source_path { from_option(input.source_path) };
	std::string output_path { from_option(input.output_path) };

	if (source_path.empty()) fatalf(AlphaBaker_InvalidOptions, "Source path not specified");
	if (output_path.empty()) fatalf(AlphaBaker_InvalidOptions, "Output path not specified");
	if (input.resolution <= 0) fatalf(AlphaBaker_InvalidOptions, "Resolution must be greater or equal to 1");
	if (input.samples <= 0) fatalf(AlphaBaker_InvalidOptions, "Samples must be greater or equal to 1");
	if (input.ray_dist_front < 0.0f) fatalf(AlphaBaker_InvalidOptions, "Ray distance must be non-negative");
	if (input.ray_dist_back < 0.0f) fatalf(AlphaBaker_InvalidOptions, "Ray distance must be non-negative");
	if (input.thread_count == 0) fatalf(AlphaBaker_InvalidOptions, "Thread count is zero, speicify or use -1 for automatic");

	size_t resolution = (size_t)input.resolution;

	Tracer tracer = { };
	tracer.thread_count = input.thread_count > 0 ? (uint32_t)input.thread_count : (uint32_t)std::thread::hardware_concurrency();
	tracer.ray_dist_front = input.ray_dist_front;
	tracer.ray_dist_back = input.ray_dist_back;
	tracer.samples = (size_t)input.samples;

	ufbx_os_thread_pool_opts thread_opts = { };
	thread_opts.max_threads = tracer.thread_count;

	tracer.thread_pool.reset(ufbx_os_create_thread_pool(&thread_opts));
	if (!tracer.thread_pool) fatalf(AlphaBaker_InternalError, "Error: Failed to create thread pool");

	ufbx_load_opts opts = { };
	opts.target_axes = ufbx_axes_right_handed_y_up;
	opts.target_unit_meters = 1.0f;
	opts.generate_missing_normals = true;
	ufbx_os_init_ufbx_thread_pool(&opts.thread_opts.pool, tracer.thread_pool.get());

	ufbx_error error;
	ufbx_unique_ptr<ufbx_scene> scene { ufbx_load_file(source_path, &opts, &error) };
	if (!scene) {
		if (error.type == UFBX_ERROR_FILE_NOT_FOUND) return fatalf(AlphaBaker_InputFileNotFound, "Input file not found: %s\n", source_path.c_str());
		return fatalf(AlphaBaker_BadInputFile, "Failed to load: %s\n%s", source_path.c_str(), ufbx_format_error_string(error).c_str());
	}

	tracer.target = Target{ resolution, resolution };

	tracer.tiles_x = (tracer.target.width + Tracer::tile_size - 1) / Tracer::tile_size;
	tracer.tiles_y = (tracer.target.height + Tracer::tile_size - 1) / Tracer::tile_size;

	tracer.sample_offsets.resize(tracer.samples);
	for (size_t i = 0; i < tracer.samples; i++) {
		tracer.sample_offsets[i] = halton_sequence(i, tracer.samples);
	}

	std::unordered_map<std::string, BakeEntry> entries;

	std::string_view low_suffix = from_option(input.low_suffix);
	std::string_view high_suffix = from_option(input.high_suffix);

	for (ufbx_node *node : scene->nodes) {
		std::string_view name = node->name;

		if (name.ends_with(high_suffix)) {
			name.remove_suffix(high_suffix.length());
			entries[std::string(name)].high_node = node;
		}

		if (name.ends_with(low_suffix)) {
			name.remove_suffix(low_suffix.length());
			entries[std::string(name)].low_node = node;
		}
	}

	if (entries.empty()) return fatalf(AlphaBaker_NoApplicableMeshes, "No applicable meshes found");

	printf("\nFound %zu models, starting to bake. (ctrl+C to cancel)\n> %zux%zu texture, %zu samples/pixel\n\n",
		entries.size(), tracer.target.width, tracer.target.height, tracer.samples);

	std::vector<std::string> order;
	for (const auto &[name, _] : entries) {
		order.push_back(name);
	}
	std::stable_sort(order.begin(), order.end());

	int count_digits = 1;
	for (size_t n = entries.size(); n >= 10; n /= 10) {
		count_digits++;
	}

	size_t ok_count = 0;
	size_t warning_count = 0;
	size_t total_count = 0;
	for (const std::string &name : order) {
		const BakeEntry &entry = entries.find(name)->second;

		printf("%0*zu/%zu %-30s ", count_digits, total_count + 1, order.size(), name.c_str());

		std::chrono::time_point begin = std::chrono::high_resolution_clock::now();
		auto result = process_entry(tracer, entry);
		std::chrono::time_point end = std::chrono::high_resolution_clock::now();

		if (result) {
			double time_seconds = std::chrono::duration<double>(end - begin).count();
			printf("%7.2fs\n", time_seconds);

			if (!result.value()) warning_count++;
			ok_count++;
		}
		total_count++;
	}

	printf("\n");
	if (warning_count > 0) {
		printf("%zu/%zu succeeded (%zu warnings)\n", ok_count, total_count, warning_count);
	} else {
		printf("%zu/%zu succeeded\n", ok_count, total_count);
	}

	AlphaBaker_Result res = save_result(tracer.target, output_path, input.invert_y != 0);
	if (res != AlphaBaker_Success) return res;

	printf("Saved output to: %s\n\n", output_path.c_str());

	if (ok_count < total_count)
		return AlphaBaker_PartialSuccess;
	return AlphaBaker_Success;
}

}
