#include <complex.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <fftw3.h>

#define Nx (8 * 8)
#define Ny (8 * 8)

#define ITERATIONS 1000000
#define RADIX2 0
#define RADIX4 0
#define RADIX8 1
#define FFTW 0
#define DEBUG 0
#define SIMD 1

#if SIMD
#if __AVX__
#include <immintrin.h>
#elif __SSE3__
#include <pmmintrin.h>
#elif __SSE__
#include <xmmintrin.h>
#endif
#endif

#ifndef M_PI
#define M_PI 3.14159265359
#endif

#ifndef M_SQRT_2
#define M_SQRT_2 0.707106781186547524401
#endif

#define SWAP(a, b) do { cfloat *tmp = b; b = a; a = tmp; } while(0)

#undef I
#define I _Complex_I

typedef complex float cfloat;

static inline cfloat twiddle(int direction, int k, int p)
{
   double phase = (M_PI * direction * k) / p;
   return cos(phase) + I * sin(phase);
}

#if SIMD

#if __AVX__
#define MM __m256
#define VSIZE 4 // Complex numbers per vector
#define permute_ps(a, x) _mm256_permute_ps(a, x)
#define moveldup_ps(x) _mm256_moveldup_ps(x)
#define movehdup_ps(x) _mm256_movehdup_ps(x)
#define xor_ps(a, b) _mm256_xor_ps(a, b)
#define add_ps(a, b) _mm256_add_ps(a, b)
#define sub_ps(a, b) _mm256_sub_ps(a, b)
#define mul_ps(a, b) _mm256_mul_ps(a, b)
#define addsub_ps(a, b) _mm256_addsub_ps(a, b)
#define load_ps(addr) _mm256_load_ps((const float*)(addr))
#define store_ps(addr, x) _mm256_store_ps((float*)(addr), x)
#define splat_const_complex(real, imag) _mm256_set_ps(imag, real, imag, real, imag, real, imag, real)
#define splat_const_dual_complex(a, b, real, imag) _mm256_set_ps(imag, real, b, a, imag, real, b, a)
#define splat_complex(addr) ((__m256)_mm256_broadcast_sd((const double*)(addr)))
#define unpacklo_pd(a, b) ((__m256)_mm256_unpacklo_pd((__m256d)(a), (__m256d)(b)))
#define unpackhi_pd(a, b) ((__m256)_mm256_unpackhi_pd((__m256d)(a), (__m256d)(b)))
#else
#define MM __m128
#define VSIZE 2 // Complex numbers per vector
#define permute_ps(a, x) _mm_shuffle_ps(a, a, x)
#define xor_ps(a, b) _mm_xor_ps(a, b)
#define add_ps(a, b) _mm_add_ps(a, b)
#define sub_ps(a, b) _mm_sub_ps(a, b)
#define mul_ps(a, b) _mm_mul_ps(a, b)
#define load_ps(addr) _mm_load_ps((const float*)(addr))
#define store_ps(addr, x) _mm_store_ps((float*)(addr), x)
#define splat_const_complex(real, imag) _mm_set_ps(imag, real, imag, real)
#define splat_const_dual_complex(a, b, real, imag) _mm_set_ps(imag, real, b, a)
#define unpacklo_pd(a, b) ((__m128)_mm_unpacklo_pd((__m128d)(a), (__m128d)(b)))
#define unpackhi_pd(a, b) ((__m128)_mm_unpackhi_pd((__m128d)(a), (__m128d)(b)))

#if __SSE3__
#define addsub_ps(a, b) _mm_addsub_ps(a, b)
#define moveldup_ps(x) _mm_moveldup_ps(x)
#define movehdup_ps(x) _mm_movehdup_ps(x)
#else
#define moveldup_ps(x) permute_ps(x, _MM_SHUFFLE(2, 2, 0, 0))
#define movehdup_ps(x) permute_ps(x, _MM_SHUFFLE(3, 3, 1, 1))
static inline __m128 addsub_ps(__m128 a, __m128 b)
{
   const MM flip_signs = splat_const_complex(-0.0f, 0.0f);
   return add_ps(a, xor_ps(b, flip_signs));
}
#endif

static inline __m128 splat_complex(const void *ptr)
{
   __m128d reg = _mm_load_sd((const double*)ptr);
   return (__m128)_mm_unpacklo_pd(reg, reg);
}
#endif

static inline MM cmul_ps(MM a, MM b)
{
   MM r3 = permute_ps(a, _MM_SHUFFLE(2, 3, 0, 1));
   MM r1 = moveldup_ps(b);
   MM R0 = mul_ps(a, r1);
   MM r2 = movehdup_ps(b);
   MM R1 = mul_ps(r2, r3);
   return addsub_ps(R0, R1);
}

#endif

static void __attribute__((noinline)) fft_forward_radix2_p1_vert(cfloat *output, const cfloat *input,
      const cfloat *twiddles, unsigned samples_x, unsigned samples_y)
{
   (void)twiddles;
#if SIMD
   unsigned half_stride = samples_x * (samples_x >> 1);
   unsigned half_lines = samples_y >> 1;

   for (unsigned line = 0; line < half_lines;
         line++, input += samples_x, output += samples_x << 1)
   {
      for (unsigned i = 0; i < samples_x; i += VSIZE)
      {
         MM a = load_ps(&input[i]);
         MM b = load_ps(&input[i + half_stride]);

         MM r0 = add_ps(a, b);
         MM r1 = sub_ps(a, b);

         store_ps(&output[i], r0);
         store_ps(&output[i + 1 * samples_x], r1);
      }
   }
#else
   unsigned half_stride = samples_x * (samples_x >> 1);
   unsigned half_lines = samples_y >> 1;

   for (unsigned line = 0; line < half_lines;
         line++, input += samples_x, output += samples_x << 1)
   {
      for (unsigned i = 0; i < samples_x; i++)
      {
         cfloat a = input[i];
         cfloat b = input[i + half_stride];

         cfloat r0 = a + b; // 0O + 0
         cfloat r1 = a - b; // 0O + 1

         output[i] = r0;
         output[i + 1 * samples_x] = r1;
      }
   }
#endif
}

static void __attribute__((noinline)) fft_forward_radix2_generic_vert(cfloat *output, const cfloat *input,
      const cfloat *twiddles, unsigned p, unsigned samples_x, unsigned samples_y)
{
#if SIMD
   unsigned half_stride = samples_x * (samples_x >> 1);
   unsigned half_lines = samples_y >> 1;
   unsigned out_stride = p * samples_x;

   for (unsigned line = 0; line < half_lines;
         line++, input += samples_x)
   {
      unsigned k = line & (p - 1);
      unsigned j = ((line << 1) - k) * samples_x;
      const MM w = splat_complex(&twiddles[k]);

      for (unsigned i = 0; i < samples_x; i += VSIZE)
      {
         MM a = load_ps(&input[i]);
         MM b = load_ps(&input[i + half_stride]);
         b = cmul_ps(b, w);

         MM r0 = add_ps(a, b);
         MM r1 = sub_ps(a, b);

         store_ps(&output[i + j], r0);
         store_ps(&output[i + j + 1 * out_stride], r1);
      }
   }
#else
   unsigned half_stride = samples_x * (samples_x >> 1);
   unsigned half_lines = samples_y >> 1;
   unsigned out_stride = p * samples_x;

   for (unsigned line = 0; line < half_lines;
         line++, input += samples_x)
   {
      unsigned k = line & (p - 1);
      unsigned j = ((line << 1) - k) * samples_x;

      for (unsigned i = 0; i < samples_x; i++)
      {
         cfloat a = input[i];
         cfloat b = twiddles[k] * input[i + half_stride];

         cfloat r0 = a + b; // 0O + 0
         cfloat r1 = a - b; // 0O + 1

         output[i + j] = r0;
         output[i + j + 1 * out_stride] = r1;
      }
   }
#endif
}

static void __attribute__((noinline)) fft_forward_radix4_p1_vert(cfloat *output, const cfloat *input,
      const cfloat *twiddles, unsigned samples_x, unsigned samples_y)
{
   (void)twiddles;
#if SIMD
   unsigned quarter_stride = samples_x * (samples_x >> 2);
   unsigned quarter_lines = samples_y >> 2;
   const MM flip_signs = splat_const_complex(0.0f, -0.0f);

   for (unsigned line = 0; line < quarter_lines;
         line++, input += samples_x, output += samples_x << 2)
   {
      for (unsigned i = 0; i < samples_x; i += VSIZE)
      {
         MM a = load_ps(&input[i]);
         MM b = load_ps(&input[i + quarter_stride]);
         MM c = load_ps(&input[i + 2 * quarter_stride]);
         MM d = load_ps(&input[i + 3 * quarter_stride]);

         MM r0 = add_ps(a, c);
         MM r1 = sub_ps(a, c);
         MM r2 = add_ps(b, d);
         MM r3 = sub_ps(b, d);
         r3 = xor_ps(permute_ps(r3, _MM_SHUFFLE(2, 3, 0, 1)), flip_signs);

         a = add_ps(r0, r2);
         b = add_ps(r1, r3);
         c = sub_ps(r0, r2);
         d = sub_ps(r1, r3);

         store_ps(&output[i], a);
         store_ps(&output[i + 1 * samples_x], b);
         store_ps(&output[i + 2 * samples_x], c);
         store_ps(&output[i + 3 * samples_x], d);
      }
   }
#else
   unsigned quarter_stride = samples_x * (samples_x >> 2);
   unsigned quarter_lines = samples_y >> 2;

   for (unsigned line = 0; line < quarter_lines;
         line++, input += samples_x, output += samples_x << 2)
   {
      for (unsigned i = 0; i < samples_x; i++)
      {
         cfloat a = input[i];
         cfloat b = input[i + quarter_stride];
         cfloat c = input[i + 2 * quarter_stride];
         cfloat d = input[i + 3 * quarter_stride];

         cfloat r0 = a + c; // 0O + 0
         cfloat r1 = a - c; // 0O + 1
         cfloat r2 = b + d; // 2O + 0
         cfloat r3 = b - d; // 2O + 1

         // p == 2 twiddles
         r3 *= -I;

         a = r0 + r2; // 0O + 0
         b = r1 + r3; // 0O + 1
         c = r0 - r2; // 00 + 2
         d = r1 - r3; // O0 + 3

         output[i] = a;
         output[i + 1 * samples_x] = b;
         output[i + 2 * samples_x] = c;
         output[i + 3 * samples_x] = d;
      }
   }
#endif
}

static void __attribute__((noinline)) fft_forward_radix8_p1_vert(cfloat *output, const cfloat *input,
      const cfloat *twiddles, unsigned samples_x, unsigned samples_y)
{
#if SIMD
   unsigned octa_stride = samples_x * (samples_x >> 3);
   unsigned octa_lines = samples_y >> 3;
   const MM flip_signs = splat_const_complex(0.0f, -0.0f);

   for (unsigned line = 0; line < octa_lines;
         line++, input += samples_x, output += samples_x << 3)
   {
      for (unsigned i = 0; i < samples_x; i += VSIZE)
      {
         MM a = load_ps(&input[i]);
         MM b = load_ps(&input[i + octa_stride]);
         MM c = load_ps(&input[i + 2 * octa_stride]);
         MM d = load_ps(&input[i + 3 * octa_stride]);
         MM e = load_ps(&input[i + 4 * octa_stride]);
         MM f = load_ps(&input[i + 5 * octa_stride]);
         MM g = load_ps(&input[i + 6 * octa_stride]);
         MM h = load_ps(&input[i + 7 * octa_stride]);

         MM r0 = add_ps(a, e);
         MM r1 = sub_ps(a, e);
         MM r2 = add_ps(b, f);
         MM r3 = sub_ps(b, f);
         MM r4 = add_ps(c, g);
         MM r5 = sub_ps(c, g);
         MM r6 = add_ps(d, h);
         MM r7 = sub_ps(d, h);
         r5 = xor_ps(permute_ps(r5, _MM_SHUFFLE(2, 3, 0, 1)), flip_signs);
         r7 = xor_ps(permute_ps(r7, _MM_SHUFFLE(2, 3, 0, 1)), flip_signs);

         a = add_ps(r0, r4);
         b = add_ps(r1, r5);
         c = sub_ps(r0, r4);
         d = sub_ps(r1, r5);
         e = add_ps(r2, r6);
         f = add_ps(r3, r7);
         g = sub_ps(r2, r6);
         h = sub_ps(r3, r7);
         f = cmul_ps(f, splat_complex(&twiddles[5]));
         g = cmul_ps(g, splat_complex(&twiddles[6]));
         h = cmul_ps(h, splat_complex(&twiddles[7]));

         r0 = add_ps(a, e);
         r1 = add_ps(b, f);
         r2 = add_ps(c, g);
         r3 = add_ps(d, h);
         r4 = sub_ps(a, e);
         r5 = sub_ps(b, f);
         r6 = sub_ps(c, g);
         r7 = sub_ps(d, h);

         store_ps(&output[i], r0);
         store_ps(&output[i + 1 * samples_x], r1);
         store_ps(&output[i + 2 * samples_x], r2);
         store_ps(&output[i + 3 * samples_x], r3);
         store_ps(&output[i + 4 * samples_x], r4);
         store_ps(&output[i + 5 * samples_x], r5);
         store_ps(&output[i + 6 * samples_x], r6);
         store_ps(&output[i + 7 * samples_x], r7);
      }
   }
#else
   unsigned octa_stride = samples_x * (samples_x >> 3);
   unsigned octa_lines = samples_y >> 3;

   for (unsigned line = 0; line < octa_lines;
         line++, input += samples_x, output += samples_x << 3)
   {
      for (unsigned i = 0; i < samples_x; i++)
      {
         cfloat a = input[i];
         cfloat b = input[i + octa_stride];
         cfloat c = input[i + 2 * octa_stride];
         cfloat d = input[i + 3 * octa_stride];
         cfloat e = input[i + 4 * octa_stride];
         cfloat f = input[i + 5 * octa_stride];
         cfloat g = input[i + 6 * octa_stride];
         cfloat h = input[i + 7 * octa_stride];

         cfloat r0 = a + e; // 0O + 0
         cfloat r1 = a - e; // 0O + 1
         cfloat r2 = b + f; // 2O + 0
         cfloat r3 = b - f; // 2O + 1
         cfloat r4 = c + g; // 4O + 0
         cfloat r5 = c - g; // 4O + 1
         cfloat r6 = d + h; // 60 + 0
         cfloat r7 = d - h; // 6O + 1

         // p == 2 twiddles
         r5 *= -I;
         r7 *= -I;

         a = r0 + r4; // 0O + 0
         b = r1 + r5; // 0O + 1
         c = r0 - r4; // 00 + 2
         d = r1 - r5; // O0 + 3
         e = r2 + r6; // 4O + 0
         f = r3 + r7; // 4O + 1
         g = r2 - r6; // 4O + 2
         h = r3 - r7; // 4O + 3

         // p == 4 twiddles
         e *= twiddles[4];
         f *= twiddles[5];
         g *= twiddles[6];
         h *= twiddles[7];

         output[i] = a + e;
         output[i + 1 * samples_x] = b + f;
         output[i + 2 * samples_x] = c + g;
         output[i + 3 * samples_x] = d + h;
         output[i + 4 * samples_x] = a - e;
         output[i + 5 * samples_x] = b - f;
         output[i + 6 * samples_x] = c - g;
         output[i + 7 * samples_x] = d - h;
      }
   }
#endif
}

static void __attribute__((noinline)) fft_forward_radix4_generic_vert(cfloat *output, const cfloat *input,
      const cfloat *twiddles, unsigned p, unsigned samples_x, unsigned samples_y)
{
#if SIMD
   unsigned quarter_stride = samples_x * (samples_x >> 2);
   unsigned quarter_lines = samples_y >> 2;
   unsigned out_stride = p * samples_x;

   for (unsigned line = 0; line < quarter_lines; line++, input += samples_x)
   {
      unsigned k = line & (p - 1);
      unsigned j = (((line - k) << 2) + k) * samples_x;

      for (unsigned i = 0; i < samples_x; i += VSIZE)
      {
         const MM w = splat_complex(&twiddles[k]);
         MM a = load_ps(&input[i]);
         MM b = load_ps(&input[i + quarter_stride]);
         MM c = cmul_ps(load_ps(&input[i + 2 * quarter_stride]), w);
         MM d = cmul_ps(load_ps(&input[i + 3 * quarter_stride]), w);

         MM r0 = add_ps(a, c);
         MM r1 = sub_ps(a, c);
         MM r2 = add_ps(b, d);
         MM r3 = sub_ps(b, d);

         MM w0 = splat_complex(&twiddles[p + k]);
         MM w1 = splat_complex(&twiddles[p + k + p]);
         r2 = cmul_ps(r2, w0);
         r3 = cmul_ps(r3, w1);

         a = add_ps(r0, r2);
         b = add_ps(r1, r3);
         c = sub_ps(r0, r2);
         d = sub_ps(r1, r3);

         store_ps(&output[i + j + 0 * out_stride], a);
         store_ps(&output[i + j + 1 * out_stride], b);
         store_ps(&output[i + j + 2 * out_stride], c);
         store_ps(&output[i + j + 3 * out_stride], d);
      }
   }
#else
   unsigned quarter_stride = samples_x * (samples_x >> 2);
   unsigned quarter_lines = samples_y >> 2;
   unsigned out_stride = p * samples_x;

   for (unsigned line = 0; line < quarter_lines; line++, input += samples_x)
   {
      unsigned k = line & (p - 1);
      unsigned j = (((line - k) << 2) + k) * samples_x;

      for (unsigned i = 0; i < samples_x; i++)
      {
         cfloat a = input[i];
         cfloat b = input[i + quarter_stride];
         cfloat c = twiddles[k] * input[i + 2 * quarter_stride];
         cfloat d = twiddles[k] * input[i + 3 * quarter_stride];

         cfloat r0 = a + c; // 0O + 0
         cfloat r1 = a - c; // 0O + 1
         cfloat r2 = b + d; // 2O + 0
         cfloat r3 = b - d; // 2O + 1

         r2 *= twiddles[p + k];
         r3 *= twiddles[p + k + p];

         a = r0 + r2; // 0O + 0
         b = r1 + r3; // 0O + 1
         c = r0 - r2; // 00 + 2
         d = r1 - r3; // O0 + 3

         output[i + j + 0 * out_stride] = a;
         output[i + j + 1 * out_stride] = b;
         output[i + j + 2 * out_stride] = c;
         output[i + j + 3 * out_stride] = d;
      }
   }
#endif
}

static void __attribute__((noinline)) fft_forward_radix8_generic_vert(cfloat *output, const cfloat *input,
      const cfloat *twiddles, unsigned p, unsigned samples_x, unsigned samples_y)
{
#if SIMD
   unsigned octa_stride = samples_x * (samples_x >> 3);
   unsigned octa_lines = samples_y >> 3;
   unsigned out_stride = p * samples_x;

   for (unsigned line = 0; line < octa_lines; line++, input += samples_x)
   {
      unsigned k = line & (p - 1);
      unsigned j = (((line - k) << 3) + k) * samples_x;

      for (unsigned i = 0; i < samples_x; i += VSIZE)
      {
         const MM w = splat_complex(&twiddles[k]);
         MM a = load_ps(&input[i]);
         MM b = load_ps(&input[i + octa_stride]);
         MM c = load_ps(&input[i + 2 * octa_stride]);
         MM d = load_ps(&input[i + 3 * octa_stride]);
         MM e = cmul_ps(load_ps(&input[i + 4 * octa_stride]), w);
         MM f = cmul_ps(load_ps(&input[i + 5 * octa_stride]), w);
         MM g = cmul_ps(load_ps(&input[i + 6 * octa_stride]), w);
         MM h = cmul_ps(load_ps(&input[i + 7 * octa_stride]), w);

         MM r0 = add_ps(a, e);
         MM r1 = sub_ps(a, e);
         MM r2 = add_ps(b, f);
         MM r3 = sub_ps(b, f);
         MM r4 = add_ps(c, g);
         MM r5 = sub_ps(c, g);
         MM r6 = add_ps(d, h);
         MM r7 = sub_ps(d, h);

         MM w0 = splat_complex(&twiddles[p + k]);
         MM w1 = splat_complex(&twiddles[p + k + p]);
         r4 = cmul_ps(r4, w0);
         r5 = cmul_ps(r5, w1);
         r6 = cmul_ps(r6, w0);
         r7 = cmul_ps(r7, w1);

         a = add_ps(r0, r4);
         b = add_ps(r1, r5);
         c = sub_ps(r0, r4);
         d = sub_ps(r1, r5);
         e = add_ps(r2, r6);
         f = add_ps(r3, r7);
         g = sub_ps(r2, r6);
         h = sub_ps(r3, r7);

         e = cmul_ps(e, splat_complex(&twiddles[3 * p + k]));
         f = cmul_ps(f, splat_complex(&twiddles[3 * p + k + p]));
         g = cmul_ps(g, splat_complex(&twiddles[3 * p + k + 2 * p]));
         h = cmul_ps(h, splat_complex(&twiddles[3 * p + k + 3 * p]));

         r0 = add_ps(a, e);
         r1 = add_ps(b, f);
         r2 = add_ps(c, g);
         r3 = add_ps(d, h);
         r4 = sub_ps(a, e);
         r5 = sub_ps(b, f);
         r6 = sub_ps(c, g);
         r7 = sub_ps(d, h);

         store_ps(&output[i + j + 0 * out_stride], r0);
         store_ps(&output[i + j + 1 * out_stride], r1);
         store_ps(&output[i + j + 2 * out_stride], r2);
         store_ps(&output[i + j + 3 * out_stride], r3);
         store_ps(&output[i + j + 4 * out_stride], r4);
         store_ps(&output[i + j + 5 * out_stride], r5);
         store_ps(&output[i + j + 6 * out_stride], r6);
         store_ps(&output[i + j + 7 * out_stride], r7);
      }
   }
#else
   unsigned octa_stride = samples_x * (samples_x >> 3);
   unsigned octa_lines = samples_y >> 3;
   unsigned out_stride = p * samples_x;

   for (unsigned line = 0; line < octa_lines; line++, input += samples_x)
   {
      unsigned k = line & (p - 1);
      unsigned j = (((line - k) << 3) + k) * samples_x;

      for (unsigned i = 0; i < samples_x; i++)
      {
         cfloat a = input[i];
         cfloat b = input[i + octa_stride];
         cfloat c = input[i + 2 * octa_stride];
         cfloat d = input[i + 3 * octa_stride];
         cfloat e = twiddles[k] * input[i + 4 * octa_stride];
         cfloat f = twiddles[k] * input[i + 5 * octa_stride];
         cfloat g = twiddles[k] * input[i + 6 * octa_stride];
         cfloat h = twiddles[k] * input[i + 7 * octa_stride];

         cfloat r0 = a + e; // 0O + 0
         cfloat r1 = a - e; // 0O + 1
         cfloat r2 = b + f; // 2O + 0
         cfloat r3 = b - f; // 2O + 1
         cfloat r4 = c + g; // 4O + 0
         cfloat r5 = c - g; // 4O + 1
         cfloat r6 = d + h; // 60 + 0
         cfloat r7 = d - h; // 6O + 1

         r4 *= twiddles[p + k];
         r5 *= twiddles[p + k + p];
         r6 *= twiddles[p + k];
         r7 *= twiddles[p + k + p];

         a = r0 + r4; // 0O + 0
         b = r1 + r5; // 0O + 1
         c = r0 - r4; // 00 + 2
         d = r1 - r5; // O0 + 3
         e = r2 + r6; // 4O + 0
         f = r3 + r7; // 4O + 1
         g = r2 - r6; // 4O + 2
         h = r3 - r7; // 4O + 3

         // p == 4 twiddles
         e *= twiddles[3 * p + k];
         f *= twiddles[3 * p + k + p];
         g *= twiddles[3 * p + k + 2 * p];
         h *= twiddles[3 * p + k + 3 * p];

         output[i + j + 0 * out_stride] = a + e;
         output[i + j + 1 * out_stride] = b + f;
         output[i + j + 2 * out_stride] = c + g;
         output[i + j + 3 * out_stride] = d + h;
         output[i + j + 4 * out_stride] = a - e;
         output[i + j + 5 * out_stride] = b - f;
         output[i + j + 6 * out_stride] = c - g;
         output[i + j + 7 * out_stride] = d - h;
      }
   }
#endif
}

static void __attribute__((noinline)) fft_forward_radix8_p1(cfloat *output, const cfloat *input,
      const cfloat *twiddles, unsigned samples)
{
   (void)twiddles;
#if SIMD
   const MM flip_signs = splat_const_complex(0.0f, -0.0f);
   const MM w_f = splat_const_complex(+M_SQRT_2, -M_SQRT_2);
   const MM w_h = splat_const_complex(-M_SQRT_2, -M_SQRT_2);

   unsigned octa_samples = samples >> 3;
   for (unsigned i = 0; i < octa_samples; i += VSIZE)
   {
      MM a = load_ps(&input[i]);
      MM b = load_ps(&input[i + octa_samples]);
      MM c = load_ps(&input[i + 2 * octa_samples]);
      MM d = load_ps(&input[i + 3 * octa_samples]);
      MM e = load_ps(&input[i + 4 * octa_samples]);
      MM f = load_ps(&input[i + 5 * octa_samples]);
      MM g = load_ps(&input[i + 6 * octa_samples]);
      MM h = load_ps(&input[i + 7 * octa_samples]);

      MM r0 = add_ps(a, e); // 0O + 0
      MM r1 = sub_ps(a, e); // 0O + 1
      MM r2 = add_ps(b, f); // 0O + 0
      MM r3 = sub_ps(b, f); // 0O + 1
      MM r4 = add_ps(c, g); // 0O + 0
      MM r5 = sub_ps(c, g); // 0O + 1
      MM r6 = add_ps(d, h); // 0O + 0
      MM r7 = sub_ps(d, h); // 0O + 1
      r5 = xor_ps(permute_ps(r5, _MM_SHUFFLE(2, 3, 0, 1)), flip_signs);
      r7 = xor_ps(permute_ps(r7, _MM_SHUFFLE(2, 3, 0, 1)), flip_signs);

      a = add_ps(r0, r4);
      b = add_ps(r1, r5);
      c = sub_ps(r0, r4);
      d = sub_ps(r1, r5);
      e = add_ps(r2, r6);
      f = add_ps(r3, r7);
      g = sub_ps(r2, r6);
      h = sub_ps(r3, r7);

      f = cmul_ps(f, w_f);
      g = xor_ps(permute_ps(g, _MM_SHUFFLE(2, 3, 0, 1)), flip_signs); // -j
      h = cmul_ps(h, w_h);

      MM o0 = add_ps(a, e); // { 0, 8, ... }
      MM o1 = add_ps(b, f); // { 1, 9, ... }
      MM o2 = add_ps(c, g); // { 2, 10, ... }
      MM o3 = add_ps(d, h); // { 3, 11, ... }
      MM o4 = sub_ps(a, e); // { 4, 12, ... }
      MM o5 = sub_ps(b, f); // { 5, 13, ... }
      MM o6 = sub_ps(c, g); // { 6, 14, ... }
      MM o7 = sub_ps(d, h); // { 7, 15, ... }

      MM o0o1_lo = unpacklo_pd(o0, o1); // { 0, 1, 16, 17 }
      MM o0o1_hi = unpackhi_pd(o0, o1); // { 8, 9, 24, 25 }
      MM o2o3_lo = unpacklo_pd(o2, o3); // { 2, 3, 18, 19 }
      MM o2o3_hi = unpackhi_pd(o2, o3); // { 10, 11, 26, 27 }
      MM o4o5_lo = unpacklo_pd(o4, o5); // { 4, 5, 20, 21 }
      MM o4o5_hi = unpackhi_pd(o4, o5); // { 12, 13, 28, 29 }
      MM o6o7_lo = unpacklo_pd(o6, o7); // { 6, 7, 22, 23 }
      MM o6o7_hi = unpackhi_pd(o6, o7); // { 14, 15, 30, 31 }

#if VSIZE == 4
      o0 = _mm256_permute2f128_ps(o0o1_lo, o2o3_lo, (2 << 4) | (0 << 0)); // { 0, 1, 2, 3 }
      o1 = _mm256_permute2f128_ps(o4o5_lo, o6o7_lo, (2 << 4) | (0 << 0)); // { 4, 5, 6, 7 }
      o2 = _mm256_permute2f128_ps(o0o1_hi, o2o3_hi, (2 << 4) | (0 << 0)); // { 8, 9, 10, 11 }
      o3 = _mm256_permute2f128_ps(o4o5_hi, o6o7_hi, (2 << 4) | (0 << 0)); // { 12, 13, 14, 15 }
      o4 = _mm256_permute2f128_ps(o0o1_lo, o2o3_lo, (3 << 4) | (1 << 0)); // ...
      o5 = _mm256_permute2f128_ps(o4o5_lo, o6o7_lo, (3 << 4) | (1 << 0));
      o6 = _mm256_permute2f128_ps(o0o1_hi, o2o3_hi, (3 << 4) | (1 << 0));
      o7 = _mm256_permute2f128_ps(o4o5_hi, o6o7_hi, (3 << 4) | (1 << 0));
#else
      o0 = o0o1_lo;
      o1 = o2o3_lo;
      o2 = o4o5_lo;
      o3 = o6o7_lo;
      o4 = o0o1_hi;
      o5 = o2o3_hi;
      o6 = o4o5_hi;
      o7 = o6o7_hi;
#endif

      unsigned j = i << 3;
      store_ps(&output[j + 0 * VSIZE], o0);
      store_ps(&output[j + 1 * VSIZE], o1);
      store_ps(&output[j + 2 * VSIZE], o2);
      store_ps(&output[j + 3 * VSIZE], o3);
      store_ps(&output[j + 4 * VSIZE], o4);
      store_ps(&output[j + 5 * VSIZE], o5);
      store_ps(&output[j + 6 * VSIZE], o6);
      store_ps(&output[j + 7 * VSIZE], o7);
   }
#else
   unsigned octa_samples = samples >> 3;
   for (unsigned i = 0; i < octa_samples; i++)
   {
      cfloat a = input[i];
      cfloat b = input[i + octa_samples];
      cfloat c = input[i + 2 * octa_samples];
      cfloat d = input[i + 3 * octa_samples];
      cfloat e = input[i + 4 * octa_samples];
      cfloat f = input[i + 5 * octa_samples];
      cfloat g = input[i + 6 * octa_samples];
      cfloat h = input[i + 7 * octa_samples];

      cfloat r0 = a + e; // 0O + 0
      cfloat r1 = a - e; // 0O + 1
      cfloat r2 = b + f; // 2O + 0
      cfloat r3 = b - f; // 2O + 1
      cfloat r4 = c + g; // 4O + 0
      cfloat r5 = c - g; // 4O + 1
      cfloat r6 = d + h; // 60 + 0
      cfloat r7 = d - h; // 6O + 1

      // p == 2 twiddles
      r5 *= -I;
      r7 *= -I;

      a = r0 + r4; // 0O + 0
      b = r1 + r5; // 0O + 1
      c = r0 - r4; // 00 + 2
      d = r1 - r5; // O0 + 3
      e = r2 + r6; // 4O + 0
      f = r3 + r7; // 4O + 1
      g = r2 - r6; // 4O + 2
      h = r3 - r7; // 4O + 3

      // p == 4 twiddles
      e *= twiddles[4];
      f *= twiddles[5];
      g *= twiddles[6];
      h *= twiddles[7];

      unsigned j = i << 3;
      output[j + 0] = a + e;
      output[j + 1] = b + f;
      output[j + 2] = c + g;
      output[j + 3] = d + h;
      output[j + 4] = a - e;
      output[j + 5] = b - f;
      output[j + 6] = c - g;
      output[j + 7] = d - h;
   }
#endif
}

static void __attribute__((noinline)) fft_forward_radix8_generic(cfloat *output, const cfloat *input,
      const cfloat *twiddles, unsigned p, unsigned samples)
{
#if SIMD
   unsigned octa_samples = samples >> 3;
   for (unsigned i = 0; i < octa_samples; i += VSIZE)
   {
      unsigned k = i & (p - 1);
      const MM w = load_ps(&twiddles[k]);
      MM a = load_ps(&input[i]);
      MM b = load_ps(&input[i + octa_samples]);
      MM c = load_ps(&input[i + 2 * octa_samples]);
      MM d = load_ps(&input[i + 3 * octa_samples]);
      MM e = load_ps(&input[i + 4 * octa_samples]);
      MM f = load_ps(&input[i + 5 * octa_samples]);
      MM g = load_ps(&input[i + 6 * octa_samples]);
      MM h = load_ps(&input[i + 7 * octa_samples]);

      e = cmul_ps(e, w);
      f = cmul_ps(f, w);
      g = cmul_ps(g, w);
      h = cmul_ps(h, w);

      MM r0 = add_ps(a, e); // 0O + 0
      MM r1 = sub_ps(a, e); // 0O + 1
      MM r2 = add_ps(b, f); // 0O + 0
      MM r3 = sub_ps(b, f); // 0O + 1
      MM r4 = add_ps(c, g); // 0O + 0
      MM r5 = sub_ps(c, g); // 0O + 1
      MM r6 = add_ps(d, h); // 0O + 0
      MM r7 = sub_ps(d, h); // 0O + 1

      const MM w0 = load_ps(&twiddles[p + k]);
      const MM w1 = load_ps(&twiddles[2 * p + k]);
      r4 = cmul_ps(r4, w0);
      r5 = cmul_ps(r5, w1);
      r6 = cmul_ps(r6, w0);
      r7 = cmul_ps(r7, w1);

      a = add_ps(r0, r4);
      b = add_ps(r1, r5);
      c = sub_ps(r0, r4);
      d = sub_ps(r1, r5);
      e = add_ps(r2, r6);
      f = add_ps(r3, r7);
      g = sub_ps(r2, r6);
      h = sub_ps(r3, r7);

      const MM we = load_ps(&twiddles[3 * p + k]);
      const MM wf = load_ps(&twiddles[3 * p + k + p]);
      const MM wg = load_ps(&twiddles[3 * p + k + 2 * p]);
      const MM wh = load_ps(&twiddles[3 * p + k + 3 * p]);
      e = cmul_ps(e, we);
      f = cmul_ps(f, wf);
      g = cmul_ps(g, wg);
      h = cmul_ps(h, wh);

      MM o0 = add_ps(a, e);
      MM o1 = add_ps(b, f);
      MM o2 = add_ps(c, g);
      MM o3 = add_ps(d, h);
      MM o4 = sub_ps(a, e);
      MM o5 = sub_ps(b, f);
      MM o6 = sub_ps(c, g);
      MM o7 = sub_ps(d, h);

      unsigned j = ((i - k) << 3) + k;
      store_ps(&output[j + 0 * p], o0);
      store_ps(&output[j + 1 * p], o1);
      store_ps(&output[j + 2 * p], o2);
      store_ps(&output[j + 3 * p], o3);
      store_ps(&output[j + 4 * p], o4);
      store_ps(&output[j + 5 * p], o5);
      store_ps(&output[j + 6 * p], o6);
      store_ps(&output[j + 7 * p], o7);
   }
#else
   unsigned octa_samples = samples >> 3;
   for (unsigned i = 0; i < octa_samples; i++)
   {
      unsigned k = i & (p - 1);
      cfloat a = input[i];
      cfloat b = input[i + octa_samples];
      cfloat c = input[i + 2 * octa_samples];
      cfloat d = input[i + 3 * octa_samples];
      cfloat e = twiddles[k] * input[i + 4 * octa_samples];
      cfloat f = twiddles[k] * input[i + 5 * octa_samples];
      cfloat g = twiddles[k] * input[i + 6 * octa_samples];
      cfloat h = twiddles[k] * input[i + 7 * octa_samples];

      cfloat r0 = a + e; // 0O + 0
      cfloat r1 = a - e; // 0O + 1
      cfloat r2 = b + f; // 2O + 0
      cfloat r3 = b - f; // 2O + 1
      cfloat r4 = c + g; // 4O + 0
      cfloat r5 = c - g; // 4O + 1
      cfloat r6 = d + h; // 60 + 0
      cfloat r7 = d - h; // 6O + 1

      r4 *= twiddles[p + k];
      r5 *= twiddles[p + k + p];
      r6 *= twiddles[p + k];
      r7 *= twiddles[p + k + p];

      a = r0 + r4; // 0O + 0
      b = r1 + r5; // 0O + 1
      c = r0 - r4; // 00 + 2
      d = r1 - r5; // O0 + 3
      e = r2 + r6; // 4O + 0
      f = r3 + r7; // 4O + 1
      g = r2 - r6; // 4O + 2
      h = r3 - r7; // 4O + 3

      // p == 4 twiddles
      e *= twiddles[3 * p + k];
      f *= twiddles[3 * p + k + p];
      g *= twiddles[3 * p + k + 2 * p];
      h *= twiddles[3 * p + k + 3 * p];

      unsigned j = ((i - k) << 3) + k;
      output[j + 0 * p] = a + e;
      output[j + 1 * p] = b + f;
      output[j + 2 * p] = c + g;
      output[j + 3 * p] = d + h;
      output[j + 4 * p] = a - e;
      output[j + 5 * p] = b - f;
      output[j + 6 * p] = c - g;
      output[j + 7 * p] = d - h;
   }
#endif
}

static void __attribute__((noinline)) fft_forward_radix4_p1(cfloat *output, const cfloat *input,
      unsigned samples)
{
#if SIMD
   const MM flip_signs = splat_const_complex(0.0f, -0.0f);
   unsigned quarter_samples = samples >> 2;

   for (unsigned i = 0; i < quarter_samples; i += VSIZE)
   {
      MM a = load_ps(&input[i]);
      MM b = load_ps(&input[i + quarter_samples]);
      MM c = load_ps(&input[i + 2 * quarter_samples]);
      MM d = load_ps(&input[i + 3 * quarter_samples]);

      MM r0 = add_ps(a, c);
      MM r1 = sub_ps(a, c);
      MM r2 = add_ps(b, d);
      MM r3 = sub_ps(b, d);
      r3 = xor_ps(permute_ps(r3, _MM_SHUFFLE(2, 3, 0, 1)), flip_signs);

      MM o0 = add_ps(r0, r2); // { 0, 4, 8, 12 }
      MM o1 = add_ps(r1, r3); // { 1, 5, 9, 13 }
      MM o2 = sub_ps(r0, r2); // { 2, 6, 10, 14 }
      MM o3 = sub_ps(r1, r3); // { 3, 7, 11, 15 }

      // Transpose
      MM o0o1_lo = unpacklo_pd(o0, o1); // { 0, 1, 8, 9 }
      MM o0o1_hi = unpackhi_pd(o0, o1); // { 4, 5, 12, 13 }
      MM o2o3_lo = unpacklo_pd(o2, o3); // { 2, 3, 10, 11 }
      MM o2o3_hi = unpackhi_pd(o2, o3); // { 6, 7, 14, 15 }
#if VSIZE == 4
      o0 = _mm256_permute2f128_ps(o0o1_lo, o2o3_lo, (2 << 4) | (0 << 0));  // { 0, 1, 2, 3 }
      o1 = _mm256_permute2f128_ps(o0o1_hi, o2o3_hi, (2 << 4) | (0 << 0));  // { 4, 5, 6, 7 }
      o2 = _mm256_permute2f128_ps(o0o1_lo, o2o3_lo, (3 << 4) | (1 << 0));  // { 8, 9, 10, 11 }
      o3 = _mm256_permute2f128_ps(o0o1_hi, o2o3_hi, (3 << 4) | (1 << 0));  // { 12, 13, 14, 15 }
#else
      o0 = o0o1_lo;
      o1 = o2o3_lo;
      o2 = o0o1_hi;
      o3 = o2o3_hi;
#endif

      unsigned j = i << 2;
      store_ps(&output[j + 0 * VSIZE], o0);
      store_ps(&output[j + 1 * VSIZE], o1);
      store_ps(&output[j + 2 * VSIZE], o2);
      store_ps(&output[j + 3 * VSIZE], o3);
   }
#else
   unsigned quarter_samples = samples >> 2;
   for (unsigned i = 0; i < quarter_samples; i++)
   {
      cfloat a = input[i];
      cfloat b = input[i + quarter_samples];
      cfloat c = input[i + 2 * quarter_samples];
      cfloat d = input[i + 3 * quarter_samples];

      cfloat r0 = a + c;
      cfloat r1 = a - c;
      cfloat r2 = b + d;
      cfloat r3 = b - d;
      r3 *= -I;

      unsigned j = i << 2;
      output[j + 0] = r0 + r2;
      output[j + 1] = r1 + r3;
      output[j + 2] = r0 - r2;
      output[j + 3] = r1 - r3;
   }
#endif
}

static void __attribute__((noinline)) fft_forward_radix4_generic(cfloat *output, const cfloat *input,
      const cfloat *twiddles, unsigned p, unsigned samples)
{
#if SIMD
   unsigned quarter_samples = samples >> 2;

   for (unsigned i = 0; i < quarter_samples; i += VSIZE)
   {
      unsigned k = i & (p - 1);

      MM w = load_ps(&twiddles[k]);
      MM w0 = load_ps(&twiddles[p + k]);
      MM w1 = load_ps(&twiddles[2 * p + k]);

      MM a = load_ps(&input[i]);
      MM b = load_ps(&input[i + quarter_samples]);
      MM c = load_ps(&input[i + 2 * quarter_samples]);
      MM d = load_ps(&input[i + 3 * quarter_samples]);

      c = cmul_ps(c, w);
      d = cmul_ps(d, w);

      MM r0 = add_ps(a, c);
      MM r1 = sub_ps(a, c);
      MM r2 = add_ps(b, d);
      MM r3 = sub_ps(b, d);

      r2 = cmul_ps(r2, w0);
      r3 = cmul_ps(r3, w1);

      MM o0 = add_ps(r0, r2);
      MM o1 = sub_ps(r0, r2);
      MM o2 = add_ps(r1, r3);
      MM o3 = sub_ps(r1, r3);

      unsigned j = ((i - k) << 2) + k;
      store_ps(&output[j + 0], o0);
      store_ps(&output[j + 1 * p], o2);
      store_ps(&output[j + 2 * p], o1);
      store_ps(&output[j + 3 * p], o3);
   }
#else
   unsigned quarter_samples = samples >> 2;
   for (unsigned i = 0; i < quarter_samples; i++)
   {
      unsigned k = i & (p - 1);

      cfloat a = input[i];
      cfloat b = input[i + quarter_samples];
      cfloat c = twiddles[k] * input[i + 2 * quarter_samples];
      cfloat d = twiddles[k] * input[i + 3 * quarter_samples];

      // DFT-2
      cfloat r0 = a + c;
      cfloat r1 = a - c;
      cfloat r2 = b + d;
      cfloat r3 = b - d;

      r2 *= twiddles[p + k];
      r3 *= twiddles[p + k + p];

      // DFT-2
      cfloat o0 = r0 + r2;
      cfloat o1 = r0 - r2;
      cfloat o2 = r1 + r3;
      cfloat o3 = r1 - r3;

      unsigned j = ((i - k) << 2) + k;
      output[j +     0] = o0;
      output[j + 1 * p] = o2;
      output[j + 2 * p] = o1;
      output[j + 3 * p] = o3;
   }
#endif
}

static void __attribute__((noinline)) fft_forward_radix2_p1(cfloat *output, const cfloat *input,
      unsigned samples)
{
#if SIMD
   unsigned half_samples = samples >> 1;
   for (unsigned i = 0; i < half_samples; i += VSIZE)
   {
      MM a = load_ps(&input[i]);
      MM b = load_ps(&input[i + half_samples]);

      MM r0 = add_ps(a, b); // { 0, 2, 4, 6 }
      MM r1 = sub_ps(a, b); // { 1, 3, 5, 7 }
      a = unpacklo_pd(r0, r1); // { 0, 1, 4, 5 }
      b = unpackhi_pd(r0, r1); // { 2, 3, 6, 7 }
#if VSIZE == 4
      r0 = _mm256_permute2f128_ps(a, b, (2 << 4) | (0 << 0)); // { 0, 1, 2, 3 }
      r1 = _mm256_permute2f128_ps(a, b, (3 << 4) | (1 << 0)); // { 4, 5, 6, 7 }
#else
      r0 = a;
      r1 = b;
#endif

      unsigned j = i << 1;
      store_ps(&output[j + 0 * VSIZE], r0);
      store_ps(&output[j + 1 * VSIZE], r1);
   }
#else
   unsigned half_samples = samples >> 1;
   for (unsigned i = 0; i < half_samples; i++)
   {
      cfloat a = input[i];
      cfloat b = input[i + half_samples]; 

      unsigned j = i << 1;
      output[j + 0] = a + b;
      output[j + 1] = a - b;
   }
#endif
}

static void __attribute__((noinline)) fft_forward_radix2_p2(cfloat *output, const cfloat *input,
      const cfloat *twiddles, unsigned samples)
{
   (void)twiddles;
#if SIMD
   unsigned half_samples = samples >> 1;
   const MM flip_signs = splat_const_dual_complex(0.0f, 0.0f, 0.0f, -0.0f);

   for (unsigned i = 0; i < half_samples; i += VSIZE)
   {
      MM a = load_ps(&input[i]);
      MM b = load_ps(&input[i + half_samples]);
      b = xor_ps(permute_ps(b, _MM_SHUFFLE(2, 3, 1, 0)), flip_signs);

      MM r0 = add_ps(a, b); // { c0, c1, c4, c5 }
      MM r1 = sub_ps(a, b); // { c2, c3, c6, c7 }
#if VSIZE == 4
      a = _mm256_permute2f128_ps(r0, r1, (2 << 4) | (0 << 0));
      b = _mm256_permute2f128_ps(r0, r1, (3 << 4) | (1 << 0));
#else
      a = r0;
      b = r1;
#endif

      unsigned j = i << 1;
      store_ps(&output[j + 0], a);
      store_ps(&output[j + VSIZE], b);
   }
#else
   unsigned half_samples = samples >> 1;
   for (unsigned i = 0; i < half_samples; i++)
   {
      unsigned k = i & (2 - 1);
      cfloat a = input[i];
      cfloat b = twiddles[k] * input[i + half_samples];

      unsigned j = (i << 1) - k;
      output[j + 0] = a + b;
      output[j + 2] = a - b;
   }
#endif
}

static void __attribute__((noinline)) fft_forward_radix2_generic(cfloat *output, const cfloat *input,
      const cfloat *twiddles, unsigned p, unsigned samples)
{
#if SIMD
   unsigned half_samples = samples >> 1;

   for (unsigned i = 0; i < half_samples; i += VSIZE)
   {
      unsigned k = i & (p - 1);

      MM w = load_ps(&twiddles[k]);
      MM a = load_ps(&input[i]);
      MM b = load_ps(&input[i + half_samples]);
      b = cmul_ps(b, w);

      MM r0 = add_ps(a, b);
      MM r1 = sub_ps(a, b);

      unsigned j = (i << 1) - k;
      store_ps(&output[j + 0], r0);
      store_ps(&output[j + p], r1);
   }
#else
   unsigned half_samples = samples >> 1;
   for (unsigned i = 0; i < half_samples; i++)
   {
      unsigned k = i & (p - 1);
      cfloat a = input[i];
      cfloat b = twiddles[k] * input[i + half_samples];

      unsigned j = (i << 1) - k;
      output[j + 0] = a + b;
      output[j + p] = a - b;
   }
#endif
}

cfloat *allocate(size_t count)
{
   void *ptr = NULL;
   if (posix_memalign(&ptr, 64, count * sizeof(cfloat)) < 0)
      return NULL;
   return (cfloat*)ptr;
}

int main()
{
   cfloat *twiddles = allocate(Nx * Ny);
   cfloat *input = allocate(Nx * Ny);
   cfloat *tmp0 = allocate(Nx * Ny);
   cfloat *tmp1 = allocate(Nx * Ny);

   cfloat *pt = twiddles;
   for (unsigned p = 1; p < Nx; p <<= 1)
   {
      for (unsigned k = 0; k < p; k++)
         pt[k] = twiddle(-1, k, p);
      pt += p;
      if (p == 2)
         pt++;
   }

   srand(0);
   for (unsigned i = 0; i < Nx * Ny; i++)
   {
      float real = (float)rand() / RAND_MAX - 0.5f;
      float imag = (float)rand() / RAND_MAX - 0.5f;
      input[i] = real + I * imag;
   }

#if RADIX2
   // Radix-2

   for (unsigned i = 0; i < ITERATIONS; i++)
   {
      cfloat *out = NULL;
      cfloat *in = NULL;

      // Horizontals
      for (unsigned y = 0; y < Ny; y++)
      {
         pt = twiddles;

         fft_forward_radix2_p1(tmp0 + y * Nx, input + y * Nx, Nx);
         pt += 1;
         fft_forward_radix2_p2(tmp1 + y * Nx, tmp0 + y * Nx, pt, Nx);
         pt += 3;
         out = tmp0;
         in = tmp1;

         for (unsigned p = 4; p < Nx; p <<= 1)
         {
            fft_forward_radix2_generic(out + y * Nx, in + y * Nx, pt, p, Nx);
            SWAP(out, in);
            pt += p;
         }
      }

      // Verticals
      pt = twiddles;

      fft_forward_radix2_p1_vert(out, in, pt, Nx, Ny);
      pt += 1;
      SWAP(out, in);

      fft_forward_radix2_generic_vert(out, in, pt, 2, Nx, Ny);
      pt += 3;
      SWAP(out, in);

      for (unsigned p = 4; p < Ny; p <<= 1)
      {
         fft_forward_radix2_generic_vert(out, in, pt, p, Nx, Ny);
         pt += p;
         SWAP(out, in);
      }

#if DEBUG
      for (unsigned i = 0; i < Nx * Ny; i++)
         printf("Radix-2 FFT[%03u] = (%+8.3f, %+8.3f)\n", i, crealf(in[i]), cimagf(in[i]));
#endif
   }
#endif

#if RADIX4
   // Radix-4

   for (unsigned i = 0; i < ITERATIONS; i++)
   {
      cfloat *out = NULL;
      cfloat *in = NULL;

      // Horizontals
      for (unsigned y = 0; y < Ny; y++)
      {
         pt = twiddles;

         fft_forward_radix4_p1(tmp0 + y * Nx, input + y * Nx, Nx);
         pt += 4;
         out = tmp1;
         in = tmp0;

         for (unsigned p = 4; p < Nx; p <<= 2)
         {
            fft_forward_radix4_generic(out + y * Nx, in + y * Nx, pt, p, Nx);
            SWAP(out, in);
            pt += p * 3;
         }
      }

      // Verticals
      pt = twiddles;
      fft_forward_radix4_p1_vert(out, in, pt, Nx, Ny);
      pt += 4;
      SWAP(out, in);

      for (unsigned p = 4; p < Ny; p <<= 2)
      {
         fft_forward_radix4_generic_vert(out, in, pt, p, Nx, Ny);
         pt += p * 3;
         SWAP(out, in);
      }

#if DEBUG
      for (unsigned i = 0; i < Nx * Ny; i++)
         printf("Radix-4 FFT[%03u] = (%+8.3f, %+8.3f)\n", i, crealf(in[i]), cimagf(in[i]));
#endif
   }
#endif

#if RADIX8
   // Radix-8

   for (unsigned i = 0; i < ITERATIONS; i++)
   {
      cfloat *out = NULL;
      cfloat *in = NULL;

      // Horizontals
      for (unsigned y = 0; y < Ny; y++)
      {
         pt = twiddles;

         fft_forward_radix8_p1(tmp0 + y * Nx, input + y * Nx, pt, Nx);
         pt += 8;
         out = tmp1;
         in = tmp0;

         for (unsigned p = 8; p < Nx; p <<= 3)
         {
            fft_forward_radix8_generic(out + y * Nx, in + y * Nx, pt, p, Nx);
            SWAP(out, in);
            pt += p * 7;
         }
      }

      // Verticals
      pt = twiddles;
      fft_forward_radix8_p1_vert(out, in, pt, Nx, Ny);
      pt += 8;
      SWAP(out, in);

      for (unsigned p = 8; p < Ny; p <<= 3)
      {
         fft_forward_radix8_generic_vert(out, in, pt, p, Nx, Ny);
         pt += p * 7;
         SWAP(out, in);
      }

#if DEBUG
      for (unsigned i = 0; i < Nx * Ny; i++)
         printf("Radix-8 FFT[%03u] = (%+8.3f, %+8.3f)\n", i, crealf(in[i]), cimagf(in[i]));
#endif
   }
#endif

#if FFTW
   cfloat *in, *out;
   fftwf_plan p;
   in = (cfloat*)fftwf_malloc(sizeof(fftw_complex) * Nx * Ny);
   out = (cfloat*)fftwf_malloc(sizeof(fftw_complex) * Nx * Ny);

   p = fftwf_plan_dft_2d(Nx, Ny, (fftwf_complex*)in, (fftwf_complex*)out, FFTW_FORWARD, FFTW_MEASURE);
   if (!p)
      return 1;

   memcpy(in, input, Nx * Ny * sizeof(fftw_complex));
   for (unsigned i = 0; i < ITERATIONS; i++)
   {
      fftwf_execute(p);
#if DEBUG
      for (unsigned i = 0; i < Nx * Ny; i++)
         printf("FFTW FFT[%03u] = (%+8.3f, %+8.3f)\n", i, crealf(out[i]), cimagf(out[i]));
#endif
   }
   fftwf_destroy_plan(p);
   fftwf_free(in);
   fftwf_free(out);
#endif

   free(tmp0);
   free(tmp1);
   free(input);
   free(twiddles);
}

