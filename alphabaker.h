
#if defined(__cplusplus)
extern "C" {
#endif

#ifndef ALPHABAKER_API
#define ALPHABAKER_API
#endif

typedef struct AlphaBaker_String {
	const char *data;
	int length;
} AlphaBaker_String;

typedef struct AlphaBaker_Options {
	AlphaBaker_String source_path;
	AlphaBaker_String output_path;

	AlphaBaker_String high_suffix;
	AlphaBaker_String low_suffix;

	int resolution;
	int samples;
	int invert_y;
	int thread_count;

	float ray_dist_front;
	float ray_dist_back;
} AlphaBaker_Options;

enum AlphaBaker_Result
{
	AlphaBaker_Success,
	AlphaBaker_PartialSuccess,

	AlphaBaker_InternalError,
	AlphaBaker_InvalidOptions,
	AlphaBaker_InputFileNotFound,
	AlphaBaker_BadInputFile,
	AlphaBaker_NoApplicableMeshes,
	AlphaBaker_FailedToReadInput,
	AlphaBaker_FailedToWriteOutput,
};

ALPHABAKER_API void AlphaBaker_defaults(AlphaBaker_Options *options);
ALPHABAKER_API int AlphaBaker_bake(const AlphaBaker_Options *options);

#if defined(__cplusplus)
}
#endif
