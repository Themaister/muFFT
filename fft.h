/* Copyright (C) 2015 Hans-Kristian Arntzen <maister@archlinux.us>
 *
 * Permission is hereby granted, free of charge,
 * to any person obtaining a copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation the rights to
 * use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
 * and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
 * WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

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

#define MUFFT_FLAG_FULL_R2C (1 << 16)
#define MUFFT_FLAG_ZERO_PAD_UPPER_HALF (1 << 17)

#define MUFFT_CONV_METHOD_MONO_MONO 0
#define MUFFT_CONV_METHOD_STEREO_MONO 1
#define MUFFT_CONV_BLOCK_FIRST 0
#define MUFFT_CONV_BLOCK_SECOND 1

typedef struct mufft_plan_1d mufft_plan_1d;
mufft_plan_1d *mufft_create_plan_1d_c2c(unsigned N, int direction, unsigned flags);
mufft_plan_1d *mufft_create_plan_1d_r2c(unsigned N, unsigned flags);
mufft_plan_1d *mufft_create_plan_1d_c2r(unsigned N, unsigned flags);
void mufft_execute_plan_1d(mufft_plan_1d *plan, void *output, const void *input);
void mufft_free_plan_1d(mufft_plan_1d *plan);

typedef struct mufft_plan_conv mufft_plan_conv;
mufft_plan_conv *mufft_create_plan_conv(unsigned N, unsigned flags, unsigned method);
void mufft_execute_conv_input(mufft_plan_conv *plan, unsigned block, const void *input);
void mufft_execute_conv_output(mufft_plan_conv *plan, void *output);
void mufft_free_plan_conv(mufft_plan_conv *plan);

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

