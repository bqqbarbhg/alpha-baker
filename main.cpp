#include <stdio.h>
#include <stdint.h>
#include <stddef.h>

#include "external/ufbx.h"
#include "external/umath.h"
#include "external/stb_image.h"
#include "external/stb_image_write.h"
#include "external/im_arg.h"

int main(int argc, char **argv)
{
	const char *source_path = nullptr;
	const char *output_path = nullptr;

	im_arg_begin_c(argc, argv);
	while (im_arg_next()) {
		im_arg_help("--help", "Show help");

		if (im_arg("path", "Path for input FBX")) {
			source_path = im_arg_str(0);
		}
		if (im_arg("-o output", "Output path")) {
			output_path = im_arg_str(0);
		}
	}


}

