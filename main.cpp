#include <stdio.h>
#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <stdarg.h>
#include <assert.h>

#include "external/ufbx.h"
#include "external/ufbx_stl.h"
#include "external/umath.h"
#include "external/rtk.h"
#include "external/stb_image.h"
#include "external/stb_image_write.h"
#include "external/im_arg.h"

#include <string_view>
#include <unordered_map>
#include <optional>
#include <memory>
#include <thread>
#include <atomic>
#include <mutex>

[[noreturn]] void fatalf(const char *fmt, ...)
{
	va_list args;
	va_start(args, fmt);
	vfprintf(stderr, fmt, args);
	fprintf(stderr, "\n");
	va_end(args);

	exit(1);
}

std::nullopt_t failf(const char *fmt, ...)
{
	va_list args;
	va_start(args, fmt);
	vfprintf(stderr, fmt, args);
	fprintf(stderr, "\n");
	va_end(args);

	return std::nullopt;
}

struct BakeEntry
{
	ufbx_node *high_node = nullptr;
	ufbx_node *low_node = nullptr;
};

struct Vertex
{
	um_vec3 position;
	um_vec3 normal;
	um_vec2 uv0;
	um_vec2 uv1;
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

struct TraceState
{
	const Mesh &low_mesh;
	const rtk_scene *low_uv_scene;
	const rtk_scene *high_scene;
	std::atomic_uint32_t tile_index;
};

struct Tracer
{
	static constexpr size_t tile_size = 32;

	Target target;
	size_t samples;

	size_t thread_count;
	size_t tiles_x;
	size_t tiles_y;
	size_t tile_count;

	float ray_dist_front;
	float ray_dist_back;

	std::mutex print_mutex;
};

struct RandomGenerator {
    uint64_t a, b;

	RandomGenerator() : a(1), b(5) { next(); }
	RandomGenerator(uint64_t seed) : a(seed), b(1) { next(); }

	uint64_t next() { 
		uint64_t x = a, y = b;
		a = y;
		x ^= x << 23;
		x ^= x >> 17;
		x ^= y;
		b = x + y;
		return x;
	}
};


template <> struct ufbx_converter<um_vec2> {
	static inline um_vec2 from(const ufbx_vec2 &v) { return { (float)v.x, (float)v.y }; }
};
template <> struct ufbx_converter<um_vec3> {
	static inline um_vec3 from(const ufbx_vec3 &v) { return { (float)v.x, (float)v.y, (float)v.z }; }
};

template<> struct std::default_delete<rtk_scene> {
	void operator()(rtk_scene *ptr) const { rtk_free_scene(ptr); }
};

static um_vec3 from_rtk(const rtk_vec3 &v) { return { v.x, v.y, v.z }; }
static rtk_vec3 to_rtk(const um_vec3 &v) { return { v.x, v.y, v.z }; }

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

			if (mesh->uv_sets.count >= 1) {
				v.uv0 = ufbx_get_vertex_vec2(&mesh->uv_sets[0].vertex_uv, ix);
			}
			if (mesh->uv_sets.count >= 2) {
				v.uv1 = ufbx_get_vertex_vec2(&mesh->uv_sets[1].vertex_uv, ix);
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

void trace_tile(Tracer &tracer, TraceState &state, size_t x, size_t y, size_t tile_index)
{
	Target &target = tracer.target;

	size_t min_x = x * tracer.tile_size; 
	size_t min_y = y * tracer.tile_size; 
	size_t max_x = std::min(min_x + tracer.tile_size, target.width);
	size_t max_y = std::min(min_y + tracer.tile_size, target.height);

	float tile[Tracer::tile_size][Tracer::tile_size] = { };

	RandomGenerator rng { tile_index };

	for (size_t x = min_x; x < max_x; x++) {
		for (size_t y = min_y; y < max_y; y++) {
			um_vec2 uv;
			uv.x = ((float)x + 0.5f) / (float)target.width;
			uv.y = ((float)y + 0.5f) / (float)target.height;

			float sum = 0.0f;
			for (size_t i = 0; i < tracer.samples; i++) {
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

			tile[y - min_y][x - min_x] = sum / (float)tracer.samples;
		}
	}

	for (size_t x = min_x; x < max_x; x++) {
		for (size_t y = min_y; y < max_y; y++) {
			float &result = target.at(x, y);
			result = std::max(result, tile[y - min_y][x - min_x]);
		}
	}
}

void trace_thread(Tracer &tracer, TraceState &state)
{
	size_t report_rate = std::max(tracer.tile_count / 32, (size_t)1);

	for (;;) {
		size_t tile_index = state.tile_index.fetch_add(1u, std::memory_order_relaxed);
		if (tile_index >= tracer.tile_count) break;

		size_t tile_x = tile_index % tracer.tiles_x;
		size_t tile_y = tile_index / tracer.tiles_y;
		trace_tile(tracer, state, tile_x, tile_y, tile_index);

		if (((tile_index + 1) % report_rate) == 0) {
			std::lock_guard lg { tracer.print_mutex };
			putchar('.');
		}
	}
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
			uv_vertices.push_back({ v.uv0.x, v.uv0.y, 0.0f });
		}

		rtk_mesh mesh = { };
		mesh.vertices = (rtk_vec3*)uv_vertices.data();
		mesh.normals = (rtk_vec3*)((char*)low_mesh.vertices.data() + offsetof(Vertex, normal));
		mesh.normals_stride = sizeof(Vertex);
		mesh.uvs = (rtk_vec2*)((char*)low_mesh.vertices.data() + offsetof(Vertex, uv0));
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
		mesh.uvs = (rtk_vec2*)((char*)high_mesh.vertices.data() + offsetof(Vertex, uv0));
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

	printf("Processing ");

	TraceState state = { low_mesh };
	state.low_uv_scene = low_uv_scene.get();
	state.high_scene = high_scene.get();

	std::vector<std::thread> threads;
	for (size_t i = 0; i < tracer.thread_count; i++) {
		threads.push_back(std::thread(trace_thread, std::ref(tracer), std::ref(state)));
	}

	for (std::thread &t : threads) {
		t.join();
	}

	printf("\n");

	return true;
}

void save_result(const Target &target, const std::string &path, bool invert_y)
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
		fatalf("Failed to write output file");
	}
}

int main(int argc, char **argv)
{
	std::string source_path;
	std::string output_path;

	std::string_view high_suffix = " High";
	std::string_view low_suffix = " Low";

	size_t resolution = 1024;
	bool invert_y = false;

	Tracer tracer = { };
	tracer.thread_count = (size_t)std::thread::hardware_concurrency();
	tracer.ray_dist_front = 1000.0f;
	tracer.ray_dist_back = 1000.0f;
	tracer.samples = 32;

	im_arg_begin_c(argc, argv);
	while (im_arg_next()) {
		im_arg_help("--help", "Show help");

		if (im_arg("path", "Path for input .fbx file")) {
			source_path = im_arg_str(0);
		}
		if (im_arg("-o output", "Output .png path")) {
			output_path = im_arg_str(0);
		}

		im_arg_category("Options");
		if (im_arg("--resolution R", "Output texture resolution")) {
			resolution = (size_t)im_arg_int_range(0, 1, 1 << 18);
		}
		if (im_arg("--samples S", "Number of samples to use")) {
			tracer.samples = (size_t)im_arg_int_range(0, 1, INT_MAX);
		}
		if (im_arg("--high-suffix suffix", "Suffix for high-poly meshes")) {
			high_suffix = im_arg_str(0);
		}
		if (im_arg("--low-suffix suffix", "Suffix for low-poly meshes")) {
			low_suffix = im_arg_str(0);
		}
		if (im_arg("--ray-range range", "Ray distance range")) {
			float dist = (float)im_arg_double_range(0, 0.0, 10000000.0);
			tracer.ray_dist_back = dist;
			tracer.ray_dist_front = dist;
		}
		if (im_arg("--invert-y", "Invert Y axis in the result")) {
			invert_y = true;
		}

		im_arg_category("Performance");
		if (im_arg("--threads N", "Number of threads to use")) {
			tracer.thread_count = (size_t)im_arg_int_range(0, 1, 1024);
		}
	}

	if (source_path.empty()) fatalf("Error: Source path not specified");
	if (output_path.empty()) fatalf("Error: Source path not specified");

	ufbx_load_opts opts = { };
	opts.target_axes = ufbx_axes_right_handed_y_up;
	opts.target_unit_meters = 1.0f;
	opts.generate_missing_normals = true;

	ufbx_error error;
	ufbx_unique_ptr<ufbx_scene> scene { ufbx_load_file(source_path, &opts, &error) };
	if (!scene) {
		fatalf("Failed to load: %s\n%s", source_path, ufbx_format_error_string(error).c_str());
	}

	tracer.target = Target{ resolution, resolution };

	tracer.tiles_x = (tracer.target.width + Tracer::tile_size - 1) / Tracer::tile_size;
	tracer.tiles_y = (tracer.target.height + Tracer::tile_size - 1) / Tracer::tile_size;
	tracer.tile_count = tracer.tiles_x * tracer.tiles_y;

	std::unordered_map<std::string, BakeEntry> entries;

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

	size_t ok_count = 0;
	size_t total_count = 0;
	for (const auto &[name, entry] : entries) {

		printf("> %s\n", name.c_str());

		auto result = process_entry(tracer, entry);
		if (result) ok_count++;
		total_count++;
	}

	save_result(tracer.target, output_path, invert_y);

	printf("\n%zu/%zu succeeded\n", ok_count, total_count);

	if (ok_count < total_count)
		return 1;

	return 0;
}

