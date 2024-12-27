#include "alphabaker.h"
#include <stdio.h>

#define IM_ARG_IMPLEMENTATION
#include "external/im_arg.h"

static AlphaBaker_String to_option(const char *str)
{
	AlphaBaker_String s = { str, (int)strlen(str) };
	return s;
}

int main(int argc, char **argv)
{
	AlphaBaker_Options opts;
	AlphaBaker_defaults(&opts);

	im_arg_begin_c(argc, argv);
	while (im_arg_next()) {
		if (im_arg_empty()) {
			im_arg_show_help();
		}

		im_arg_helpf("\nUsage: alpha-baker input.fbx -o output.png\n");

		im_arg_helpf("\n"
			" Bake high-poly geometry into mask for a low-poly mesh.\n"
			" This tool matches the high and low poly meshes using a suffix (default ' High' and ' Low').\n"
			" For example, mesh 'Grass High' would be baked into mesh 'Grass Low'.\n");

		im_arg_category("Files");
		if (im_arg("-i path", "Path for input .fbx file")) {
			opts.source_path = to_option(im_arg_str(0));
		}
		if (im_arg("-o output", "Output .png path")) {
			opts.output_path = to_option(im_arg_str(0));
		}

		im_arg_category("Texture");
		if (im_arg("--resolution N", "Output texture resolution")) {
			opts.resolution = (size_t)im_arg_int_range(0, 1, 1 << 18);
		}
		if (im_arg("--invert-y", "Invert Y axis in the result")) {
			opts.invert_y = true;
		}

		im_arg_category("Baking");
		if (im_arg("--samples N", "Number of samples to use for baking")) {
			opts.samples = (size_t)im_arg_int_range(0, 1, INT_MAX);
		}
		if (im_arg("--high-suffix suffix", "Suffix for high-poly meshes")) {
			opts.high_suffix = to_option(im_arg_str(0));
		}
		if (im_arg("--low-suffix suffix", "Suffix for low-poly meshes")) {
			opts.low_suffix = to_option(im_arg_str(0));
		}
		if (im_arg("--distance range", "Ray distance from low-poly geometry")) {
			float dist = (float)im_arg_double_range(0, 0.0, 10000000.0);
			opts.ray_dist_back = dist;
			opts.ray_dist_front = dist;
		}
		if (im_arg("--distance-separate front back", "Ray distance range (separate distances for front and back)")) {
			opts.ray_dist_front = (float)im_arg_double_range(0, 0.0, 10000000.0);
			opts.ray_dist_back = (float)im_arg_double_range(0, 0.0, 10000000.0);
		}

		im_arg_category("Other");
		if (im_arg("--threads N", "Number of threads to use")) {
			opts.thread_count = (uint32_t)im_arg_int_range(0, 1, 1024);
		}
		im_arg_help("--help", "Show this help");

		im_arg_helpf("\n");
	}

	int result = AlphaBaker_bake(&opts);

	return result;
}

