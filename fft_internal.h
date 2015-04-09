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

#ifndef MUFFT_INTERNAL_H__
#define MUFFT_INTERNAL_H__

#include "fft.h"
#include <math.h>
#include <complex.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#ifndef M_SQRT1_2
#define M_SQRT1_2 0.707106781186547524401
#endif

#define MUFFT_ALIGNMENT 64
#define SWAP(a, b) do { cfloat *tmp = b; b = a; a = tmp; } while(0)
#define ARRAY_SIZE(a) (sizeof(a) / sizeof((a)[0]))

#undef I
#define I _Complex_I

typedef complex float cfloat;
typedef void (*mufft_1d_func)(void * MUFFT_RESTRICT output, const void * MUFFT_RESTRICT input,
        const cfloat * MUFFT_RESTRICT twiddles, unsigned p, unsigned samples);
typedef void (*mufft_2d_func)(void * MUFFT_RESTRICT output, const void * MUFFT_RESTRICT input,
        const cfloat * MUFFT_RESTRICT twiddles, unsigned p, unsigned samples_x, unsigned samples_y);

typedef void (*mufft_r2c_resolve_func)(cfloat * MUFFT_RESTRICT output, const cfloat * MUFFT_RESTRICT input, const cfloat * MUFFT_RESTRICT twiddles, unsigned samples);
typedef void (*mufft_convolve_func)(cfloat * MUFFT_RESTRICT output, const cfloat * MUFFT_RESTRICT a, const cfloat * MUFFT_RESTRICT b,
        float normalization, unsigned samples);

#define MANGLE(name, arch) mufft_ ## name ## _ ## arch
#define FFT_CONVOLVE_FUNC(name, arch) void MANGLE(name, arch) (cfloat * MUFFT_RESTRICT output, const cfloat * MUFFT_RESTRICT a, const cfloat * MUFFT_RESTRICT b, float normalization, unsigned samples);
#define FFT_RESOLVE_FUNC(name, arch) void MANGLE(name, arch) (cfloat * MUFFT_RESTRICT output, const cfloat * MUFFT_RESTRICT input, const cfloat * MUFFT_RESTRICT twiddles, unsigned samples);
#define FFT_1D_FUNC(name, arch) void MANGLE(name, arch) (void * MUFFT_RESTRICT output, const void * MUFFT_RESTRICT input, const cfloat * MUFFT_RESTRICT twiddles, unsigned p, unsigned samples);
#define FFT_2D_FUNC(name, arch) void MANGLE(name, arch) (void * MUFFT_RESTRICT output, const void * MUFFT_RESTRICT input, const cfloat * MUFFT_RESTRICT twiddles, unsigned p, unsigned samples_x, unsigned samples_y);

#define DECLARE_FFT_CPU(arch) \
    FFT_CONVOLVE_FUNC(convolve, arch) \
    FFT_RESOLVE_FUNC(resolve_r2c, arch) \
    FFT_RESOLVE_FUNC(resolve_r2c_full, arch) \
    FFT_RESOLVE_FUNC(resolve_c2r, arch) \
    FFT_1D_FUNC(forward_radix8_p1, arch) \
    FFT_1D_FUNC(forward_radix4_p1, arch) \
    FFT_1D_FUNC(radix2_p1, arch) \
    FFT_1D_FUNC(forward_radix2_p2, arch) \
    FFT_1D_FUNC(radix2_half_p1, arch) \
    FFT_1D_FUNC(forward_half_radix8_p1, arch) \
    FFT_1D_FUNC(forward_half_radix4_p1, arch) \
    FFT_1D_FUNC(forward_radix2_p2, arch) \
    FFT_1D_FUNC(inverse_radix8_p1, arch) \
    FFT_1D_FUNC(inverse_radix4_p1, arch) \
    FFT_1D_FUNC(inverse_radix2_p2, arch) \
    FFT_1D_FUNC(radix8_generic, arch) \
    FFT_1D_FUNC(radix4_generic, arch) \
    FFT_1D_FUNC(radix2_generic, arch) \
    FFT_2D_FUNC(radix2_p1_vert, arch) \
    FFT_2D_FUNC(forward_radix8_p1_vert, arch) \
    FFT_2D_FUNC(forward_radix4_p1_vert, arch) \
    FFT_2D_FUNC(inverse_radix8_p1_vert, arch) \
    FFT_2D_FUNC(inverse_radix4_p1_vert, arch) \
    FFT_2D_FUNC(radix8_generic_vert, arch) \
    FFT_2D_FUNC(radix4_generic_vert, arch) \
    FFT_2D_FUNC(radix2_generic_vert, arch)

DECLARE_FFT_CPU(avx)
DECLARE_FFT_CPU(sse3)
DECLARE_FFT_CPU(sse)
DECLARE_FFT_CPU(c)

#define MUFFT_FLAG_MASK_CPU MUFFT_FLAG_CPU_NO_SIMD
#define MUFFT_FLAG_CPU_AVX MUFFT_FLAG_CPU_NO_AVX
#define MUFFT_FLAG_CPU_SSE3 MUFFT_FLAG_CPU_NO_SSE3
#define MUFFT_FLAG_CPU_SSE MUFFT_FLAG_CPU_NO_SSE

unsigned mufft_get_cpu_flags(void);

#define MUFFT_FLAG_DIRECTION_INVERSE (1 << 16)
#define MUFFT_FLAG_DIRECTION_FORWARD (1 << 17)
#define MUFFT_FLAG_DIRECTION_ANY 0

#define MUFFT_FLAG_R2C (1 << 17)
#define MUFFT_FLAG_C2R (1 << 18)
#define MUFFT_FLAG_NO_ZERO_PAD_UPPER_HALF (1 << 19)

#ifdef MUFFT_DEBUG
#define mufft_assert(x) do { if (!(x)) { abort(); } } while(0)
#else
#define mufft_assert(x) ((void)0)
#endif

#define MUFFT_PADDING_COMPLEX_SAMPLES 4

#endif

