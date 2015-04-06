#ifndef MUFFT_H__
#define MUFFT_H__

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>

#define MUFFT_FORWARD (-1)
#define MUFFT_INVERSE  (1)

#define MUFFT_FLAG_CPU_ANY (0)
#define MUFFT_FLAG_CPU_NO_SIMD ((1 << 16) - 1)
#define MUFFT_FLAG_CPU_NO_AVX (1 << 0)
#define MUFFT_FLAG_CPU_NO_SSE3 (1 << 1)
#define MUFFT_FLAG_CPU_NO_SSE (1 << 2)

typedef struct mufft_plan_1d mufft_plan_1d;
mufft_plan_1d *mufft_create_plan_1d_c2c(unsigned N, int direction, unsigned flags);
void mufft_execute_plan_1d(mufft_plan_1d *plan, void *output, const void *input);
void mufft_free_plan_1d(mufft_plan_1d *plan);

typedef struct mufft_plan_2d mufft_plan_2d;
mufft_plan_2d *mufft_create_plan_2d_c2c(unsigned Nx, unsigned Ny, int direction, unsigned flags);
void mufft_execute_plan_2d(mufft_plan_2d *plan, void *output, const void *input);
void mufft_free_plan_2d(mufft_plan_2d *plan);

void *mufft_alloc(size_t size);
void *mufft_calloc(size_t size);
void mufft_free(void *ptr);

#ifdef __cplusplus
}
#endif

#endif

