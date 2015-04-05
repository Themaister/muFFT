#ifndef MUFFT_H__
#define MUFFT_H__

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>

#define MUFFT_FORWARD -1
#define MUFFT_INVERSE  1

typedef struct mufft_plan_1d mufft_plan_1d;

mufft_plan_1d *mufft_create_plan_1d_c2c(unsigned N, int direction);

void mufft_execute_plan_1d(mufft_plan_1d *plan, void *output, const void *input);

void mufft_free_plan_1d(mufft_plan_1d *plan);

void *mufft_alloc(size_t size);
void *mufft_calloc(size_t size);
void mufft_free(void *ptr);

#ifdef __cplusplus
}
#endif

#endif

