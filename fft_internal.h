#ifndef MUFFT_INTERNAL_H__
#define MUFFT_INTERNAL_H__

#include "fft.h"
#include <math.h>
#include <complex.h>

#ifndef M_PI
#define M_PI 3.14159265359
#endif

#ifndef M_SQRT_2
#define M_SQRT_2 0.707106781186547524401
#endif

#define MUFFT_ALIGNMENT 64
#define SWAP(a, b) do { cfloat *tmp = b; b = a; a = tmp; } while(0)
#define ARRAY_SIZE(a) (sizeof(a) / sizeof((a)[0]))

#undef I
#define I _Complex_I

typedef complex float cfloat;
typedef void (*mufft_1d_func)(void *output, const void *input,
        const cfloat *twiddles, unsigned p, unsigned samples);
typedef void (*mufft_2d_func)(void *output, const void *input,
        const cfloat *twiddles, unsigned p, unsigned samples_x, unsigned samples_y);

#define MANGLE(name, arch) mufft_ ## name ## _ ## arch
#define FFT_1D_FUNC(name, arch) void MANGLE(name, arch) (void *output, const void *input, const cfloat *twiddles, unsigned p, unsigned samples);
#define FFT_2D_FUNC(name, arch) void MANGLE(name, arch) (void *output, const void *input, const cfloat *twiddles, unsigned p, unsigned samples_x, unsigned samples_y);

#define DECLARE_FFT_CPU(arch) \
    FFT_1D_FUNC(radix2_p1, arch) \
    FFT_1D_FUNC(forward_radix8_p1, arch) \
    FFT_1D_FUNC(forward_radix4_p1, arch) \
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

#define MUFFT_FLAG_DIRECTION_INVERSE (1 << 16)
#define MUFFT_FLAG_DIRECTION_FORWARD (1 << 17)
#define MUFFT_FLAG_DIRECTION_ANY 0

#ifdef MUFFT_DEBUG
#define mufft_assert(x) do { if (!(x)) { abort(); } } while(0)
#else
#define mufft_assert(x) ((void)0)
#endif

#endif

