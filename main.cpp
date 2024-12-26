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
	rtk_scene *low_uv_scene;
	rtk_scene *high_scene;
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

	std::mutex print_mutex;
};

template <> struct ufbx_converter<um_vec2> {
	static inline um_vec2 from(const ufbx_vec2 &v) { return { (float)v.x, (float)v.y }; }
};
template <> struct ufbx_converter<um_vec3> {
	static inline um_vec3 from(const ufbx_vec3 &v) { return { (float)v.x, (float)v.y, (float)v.y }; }
};

template<> struct std::default_delete<rtk_scene> {
	void operator()(rtk_scene *ptr) const { rtk_free_scene(ptr); }
};

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

void trace_tile(Tracer &tracer, TraceState &state, size_t x, size_t y)
{
	Target &target = tracer.target;

	size_t min_x = x * tracer.tile_size; 
	size_t min_y = y * tracer.tile_size; 
	size_t max_x = std::min(min_x + tracer.tile_size, target.width);
	size_t max_y = std::min(min_y + tracer.tile_size, target.height);

	float tile[Tracer::tile_size][Tracer::tile_size] = { };

	for (size_t x = min_x; x < max_x; x++) {
		for (size_t y = min_y; y < max_y; y++) {
			um_vec2 uv;
			uv.x = ((float)x + 0.5f) / (float)target.width;
			uv.y = ((float)y + 0.5f) / (float)target.height;


		}
	}

	for (size_t x = min_x; x < max_x; x++) {
		for (size_t y = min_y; y < max_y; y++) {
			target.at(x, y) = tile[y - min_y][x - min_x];
		}
	}
}

void trace_thread(Tracer &tracer, TraceState &state)
{
	size_t report_rate = tracer.tile_count / 32;

	for (;;) {
		size_t tile_index = state.tile_index.fetch_add(1u, std::memory_order_relaxed);
		if (tile_index >= tracer.tile_count) break;

		size_t tile_x = tile_index % tracer.tiles_x;
		size_t tile_y = tile_index % tracer.tiles_y;
		trace_tile(tracer, state, tile_x, tile_y);

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

	std::optional<Mesh> high_mesh_opt = create_mesh(entry.low_node);
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

	TraceState state;
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

int main(int argc, char **argv)
{
	std::string source_path;
	std::string output_path;

	std::string_view high_suffix = " High";
	std::string_view low_suffix = " Low";

	size_t resolution = 1024;
	size_t samples = 32;
	size_t threads = (size_t)std::thread::hardware_concurrency();

	im_arg_begin_c(argc, argv);
	while (im_arg_next()) {
		im_arg_help("--help", "Show help");

		if (im_arg("path", "Path for input FBX")) {
			source_path = im_arg_str(0);
		}
		if (im_arg("-o output", "Output path")) {
			output_path = im_arg_str(0);
		}

		im_arg_category("Options");
		if (im_arg("--resolution R", "Output texture resolution")) {
			resolution = (size_t)im_arg_int_range(0, 1, 1 << 18);
		}
		if (im_arg("--samples S", "Number of samples to use")) {
			samples = (size_t)im_arg_int_range(0, 1, INT_MAX);
		}
		if (im_arg("--high-suffix suffix", "Suffix for high-poly meshes")) {
			high_suffix = im_arg_str(0);
		}
		if (im_arg("--low-suffix suffix", "Suffix for low-poly meshes")) {
			low_suffix = im_arg_str(0);
		}

		im_arg_category("Performance");
		if (im_arg("--threads N", "Number of threads to use")) {
			threads = (size_t)im_arg_int_range(0, 1, 1024);
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

	Tracer tracer;
	tracer.target = Target{ resolution, resolution };
	tracer.samples = samples;
	tracer.thread_count = threads;

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

	printf("\n%zu/%zu succeeded\n", ok_count, total_count);

	if (ok_count < total_count)
		return 1;

	return 0;
}

