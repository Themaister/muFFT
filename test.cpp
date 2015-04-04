
#include <complex>
#include <cmath>
#include <string.h>

#include <fftw3.h>

#define N (8 * 8 * 8 * 8 * 8)
#define ITERATIONS 10000
#define RADIX2 0
#define RADIX4 0
#define RADIX8 0
#define FFTW 1
#define DEBUG 0
#define SIMD 1

#if SIMD
#include <immintrin.h>
#endif

#ifndef M_SQRT_2
#define M_SQRT_2 0.707106781186547524401
#endif

using namespace std;

static inline complex<float> twiddle(int direction, int k, int p)
{
   double phase = (M_PI * direction * k) / p;
   return complex<float>(cos(phase), sin(phase));
}

#if SIMD
static inline __m256 _mm256_cmul_ps(__m256 a, __m256 b)
{
   auto r3 = _mm256_permute_ps(a, _MM_SHUFFLE(2, 3, 0, 1));
   auto r1 = _mm256_moveldup_ps(b);
   auto R0 = _mm256_mul_ps(a, r1);
   auto r2 = _mm256_movehdup_ps(b);
   auto R1 = _mm256_mul_ps(r2, r3);
   return _mm256_addsub_ps(R0, R1);
}
#endif

static void __attribute__((noinline)) fft_forward_radix8_p1(complex<float> *output, const complex<float> *input,
      const complex<float> *twiddles, unsigned samples)
{
#if SIMD
   const auto flip_signs = _mm256_set_ps(-0.0f, 0.0f, -0.0f, 0.0f, -0.0f, 0.0f, -0.0f, 0.0f);
   const auto w_f = _mm256_set_ps(-M_SQRT_2, +M_SQRT_2, -M_SQRT_2, +M_SQRT_2, -M_SQRT_2, +M_SQRT_2, -M_SQRT_2, +M_SQRT_2);
   const auto w_h = _mm256_set_ps(-M_SQRT_2, -M_SQRT_2, -M_SQRT_2, -M_SQRT_2, -M_SQRT_2, -M_SQRT_2, -M_SQRT_2, -M_SQRT_2);

   unsigned octa_samples = samples >> 3;
   for (unsigned i = 0; i < octa_samples; i += 4)
   {
      auto a = _mm256_load_ps((const float*)&input[i]);
      auto b = _mm256_load_ps((const float*)&input[i + octa_samples]);
      auto c = _mm256_load_ps((const float*)&input[i + 2 * octa_samples]);
      auto d = _mm256_load_ps((const float*)&input[i + 3 * octa_samples]);
      auto e = _mm256_load_ps((const float*)&input[i + 4 * octa_samples]);
      auto f = _mm256_load_ps((const float*)&input[i + 5 * octa_samples]);
      auto g = _mm256_load_ps((const float*)&input[i + 6 * octa_samples]);
      auto h = _mm256_load_ps((const float*)&input[i + 7 * octa_samples]);

      auto r0 = _mm256_add_ps(a, e); // 0O + 0
      auto r1 = _mm256_sub_ps(a, e); // 0O + 1
      auto r2 = _mm256_add_ps(b, f); // 0O + 0
      auto r3 = _mm256_sub_ps(b, f); // 0O + 1
      auto r4 = _mm256_add_ps(c, g); // 0O + 0
      auto r5 = _mm256_sub_ps(c, g); // 0O + 1
      auto r6 = _mm256_add_ps(d, h); // 0O + 0
      auto r7 = _mm256_sub_ps(d, h); // 0O + 1
      r5 = _mm256_xor_ps(_mm256_permute_ps(r5, _MM_SHUFFLE(2, 3, 0, 1)), flip_signs);
      r7 = _mm256_xor_ps(_mm256_permute_ps(r7, _MM_SHUFFLE(2, 3, 0, 1)), flip_signs);

      a = _mm256_add_ps(r0, r4);
      b = _mm256_add_ps(r1, r5);
      c = _mm256_sub_ps(r0, r4);
      d = _mm256_sub_ps(r1, r5);
      e = _mm256_add_ps(r2, r6);
      f = _mm256_add_ps(r3, r7);
      g = _mm256_sub_ps(r2, r6);
      h = _mm256_sub_ps(r3, r7);

      f = _mm256_cmul_ps(f, w_f);
      g = _mm256_xor_ps(_mm256_permute_ps(g, _MM_SHUFFLE(2, 3, 0, 1)), flip_signs); // -j
      h = _mm256_cmul_ps(h, w_h);

      auto o0 = _mm256_add_ps(a, e); // { 0, 8, ... }
      auto o1 = _mm256_add_ps(b, f); // { 1, 9, ... }
      auto o2 = _mm256_add_ps(c, g); // { 2, 10, ... }
      auto o3 = _mm256_add_ps(d, h); // { 3, 11, ... }
      auto o4 = _mm256_sub_ps(a, e); // { 4, 12, ... }
      auto o5 = _mm256_sub_ps(b, f); // { 5, 13, ... }
      auto o6 = _mm256_sub_ps(c, g); // { 6, 14, ... }
      auto o7 = _mm256_sub_ps(d, h); // { 7, 15, ... }

      auto o0o1_lo = (__m256)_mm256_unpacklo_pd((__m256d)o0, (__m256d)o1); // { 0, 1, 16, 17 }
      auto o0o1_hi = (__m256)_mm256_unpackhi_pd((__m256d)o0, (__m256d)o1); // { 8, 9, 24, 25 }
      auto o2o3_lo = (__m256)_mm256_unpacklo_pd((__m256d)o2, (__m256d)o3); // { 2, 3, 18, 19 }
      auto o2o3_hi = (__m256)_mm256_unpackhi_pd((__m256d)o2, (__m256d)o3); // { 10, 11, 26, 27 }
      auto o4o5_lo = (__m256)_mm256_unpacklo_pd((__m256d)o4, (__m256d)o5); // { 4, 5, 20, 21 }
      auto o4o5_hi = (__m256)_mm256_unpackhi_pd((__m256d)o4, (__m256d)o5); // { 12, 13, 28, 29 }
      auto o6o7_lo = (__m256)_mm256_unpacklo_pd((__m256d)o6, (__m256d)o7); // { 6, 7, 22, 23 }
      auto o6o7_hi = (__m256)_mm256_unpackhi_pd((__m256d)o6, (__m256d)o7); // { 14, 15, 30, 31 }
      o0 = _mm256_permute2f128_ps(o0o1_lo, o2o3_lo, (2 << 4) | (0 << 0)); // { 0, 1, 2, 3 }
      o1 = _mm256_permute2f128_ps(o4o5_lo, o6o7_lo, (2 << 4) | (0 << 0)); // { 4, 5, 6, 7 }
      o2 = _mm256_permute2f128_ps(o0o1_hi, o2o3_hi, (2 << 4) | (0 << 0)); // { 8, 9, 10, 11 }
      o3 = _mm256_permute2f128_ps(o4o5_hi, o6o7_hi, (2 << 4) | (0 << 0)); // { 12, 13, 14, 15 }
      o4 = _mm256_permute2f128_ps(o0o1_lo, o2o3_lo, (3 << 4) | (1 << 0)); // ...
      o5 = _mm256_permute2f128_ps(o4o5_lo, o6o7_lo, (3 << 4) | (1 << 0));
      o6 = _mm256_permute2f128_ps(o0o1_hi, o2o3_hi, (3 << 4) | (1 << 0));
      o7 = _mm256_permute2f128_ps(o4o5_hi, o6o7_hi, (3 << 4) | (1 << 0));

      unsigned j = i << 3;
      _mm256_store_ps((float*)&output[j +  0], o0);
      _mm256_store_ps((float*)&output[j +  4], o1);
      _mm256_store_ps((float*)&output[j +  8], o2);
      _mm256_store_ps((float*)&output[j + 12], o3);
      _mm256_store_ps((float*)&output[j + 16], o4);
      _mm256_store_ps((float*)&output[j + 20], o5);
      _mm256_store_ps((float*)&output[j + 24], o6);
      _mm256_store_ps((float*)&output[j + 28], o7);
   }
#else
   unsigned octa_samples = samples >> 3;
   for (unsigned i = 0; i < octa_samples; i++)
   {
      auto a = input[i];
      auto b = input[i + octa_samples];
      auto c = input[i + 2 * octa_samples];
      auto d = input[i + 3 * octa_samples];
      auto e = input[i + 4 * octa_samples];
      auto f = input[i + 5 * octa_samples];
      auto g = input[i + 6 * octa_samples];
      auto h = input[i + 7 * octa_samples];

      auto r0 = a + e; // 0O + 0
      auto r1 = a - e; // 0O + 1
      auto r2 = b + f; // 2O + 0
      auto r3 = b - f; // 2O + 1
      auto r4 = c + g; // 4O + 0
      auto r5 = c - g; // 4O + 1
      auto r6 = d + h; // 60 + 0
      auto r7 = d - h; // 6O + 1

      // p == 2 twiddles
      r5 = complex<float>(r5.imag(), -r5.real());
      r7 = complex<float>(r7.imag(), -r7.real());

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

static void __attribute__((noinline)) fft_forward_radix8_generic(complex<float> *output, const complex<float> *input,
      const complex<float> *twiddles, unsigned p, unsigned samples)
{
#if SIMD
   unsigned octa_samples = samples >> 3;
   for (unsigned i = 0; i < octa_samples; i += 4)
   {
      unsigned k = i & (p - 1);
      const auto w = _mm256_load_ps((const float*)&twiddles[k]);
      auto a = _mm256_load_ps((const float*)&input[i]);
      auto b = _mm256_load_ps((const float*)&input[i + octa_samples]);
      auto c = _mm256_load_ps((const float*)&input[i + 2 * octa_samples]);
      auto d = _mm256_load_ps((const float*)&input[i + 3 * octa_samples]);
      auto e = _mm256_load_ps((const float*)&input[i + 4 * octa_samples]);
      auto f = _mm256_load_ps((const float*)&input[i + 5 * octa_samples]);
      auto g = _mm256_load_ps((const float*)&input[i + 6 * octa_samples]);
      auto h = _mm256_load_ps((const float*)&input[i + 7 * octa_samples]);

      e = _mm256_cmul_ps(e, w);
      f = _mm256_cmul_ps(f, w);
      g = _mm256_cmul_ps(g, w);
      h = _mm256_cmul_ps(h, w);

      auto r0 = _mm256_add_ps(a, e); // 0O + 0
      auto r1 = _mm256_sub_ps(a, e); // 0O + 1
      auto r2 = _mm256_add_ps(b, f); // 0O + 0
      auto r3 = _mm256_sub_ps(b, f); // 0O + 1
      auto r4 = _mm256_add_ps(c, g); // 0O + 0
      auto r5 = _mm256_sub_ps(c, g); // 0O + 1
      auto r6 = _mm256_add_ps(d, h); // 0O + 0
      auto r7 = _mm256_sub_ps(d, h); // 0O + 1

      const auto w0 = _mm256_load_ps((const float*)&twiddles[p + k]);
      const auto w1 = _mm256_load_ps((const float*)&twiddles[2 * p + k]);
      r4 = _mm256_cmul_ps(r4, w0);
      r5 = _mm256_cmul_ps(r5, w1);
      r6 = _mm256_cmul_ps(r6, w0);
      r7 = _mm256_cmul_ps(r7, w1);

      a = _mm256_add_ps(r0, r4);
      b = _mm256_add_ps(r1, r5);
      c = _mm256_sub_ps(r0, r4);
      d = _mm256_sub_ps(r1, r5);
      e = _mm256_add_ps(r2, r6);
      f = _mm256_add_ps(r3, r7);
      g = _mm256_sub_ps(r2, r6);
      h = _mm256_sub_ps(r3, r7);

      const auto we = _mm256_load_ps((const float*)&twiddles[3 * p + k]);
      const auto wf = _mm256_load_ps((const float*)&twiddles[3 * p + k + p]);
      const auto wg = _mm256_load_ps((const float*)&twiddles[3 * p + k + 2 * p]);
      const auto wh = _mm256_load_ps((const float*)&twiddles[3 * p + k + 3 * p]);
      e = _mm256_cmul_ps(e, we);
      f = _mm256_cmul_ps(f, wf);
      g = _mm256_cmul_ps(g, wg);
      h = _mm256_cmul_ps(h, wh);

      auto o0 = _mm256_add_ps(a, e);
      auto o1 = _mm256_add_ps(b, f);
      auto o2 = _mm256_add_ps(c, g);
      auto o3 = _mm256_add_ps(d, h);
      auto o4 = _mm256_sub_ps(a, e);
      auto o5 = _mm256_sub_ps(b, f);
      auto o6 = _mm256_sub_ps(c, g);
      auto o7 = _mm256_sub_ps(d, h);

      unsigned j = ((i - k) << 3) + k;
      _mm256_store_ps((float*)&output[j + 0 * p], o0);
      _mm256_store_ps((float*)&output[j + 1 * p], o1);
      _mm256_store_ps((float*)&output[j + 2 * p], o2);
      _mm256_store_ps((float*)&output[j + 3 * p], o3);
      _mm256_store_ps((float*)&output[j + 4 * p], o4);
      _mm256_store_ps((float*)&output[j + 5 * p], o5);
      _mm256_store_ps((float*)&output[j + 6 * p], o6);
      _mm256_store_ps((float*)&output[j + 7 * p], o7);
   }
#else
   unsigned octa_samples = samples >> 3;
   for (unsigned i = 0; i < octa_samples; i++)
   {
      unsigned k = i & (p - 1);
      auto a = input[i];
      auto b = input[i + octa_samples];
      auto c = input[i + 2 * octa_samples];
      auto d = input[i + 3 * octa_samples];
      auto e = twiddles[k] * input[i + 4 * octa_samples];
      auto f = twiddles[k] * input[i + 5 * octa_samples];
      auto g = twiddles[k] * input[i + 6 * octa_samples];
      auto h = twiddles[k] * input[i + 7 * octa_samples];

      auto r0 = a + e; // 0O + 0
      auto r1 = a - e; // 0O + 1
      auto r2 = b + f; // 2O + 0
      auto r3 = b - f; // 2O + 1
      auto r4 = c + g; // 4O + 0
      auto r5 = c - g; // 4O + 1
      auto r6 = d + h; // 60 + 0
      auto r7 = d - h; // 6O + 1

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

static void __attribute__((noinline)) fft_forward_radix4_p1(complex<float> *output, const complex<float> *input,
      unsigned samples)
{
#if SIMD
   const auto flip_signs = _mm256_set_ps(-0.0f, 0.0f, -0.0f, 0.0f, -0.0f, 0.0f, -0.0f, 0.0f);
   unsigned quarter_samples = samples >> 2;

   for (unsigned i = 0; i < quarter_samples; i += 4)
   {
      auto a = _mm256_load_ps((const float*)&input[i]);
      auto b = _mm256_load_ps((const float*)&input[i + quarter_samples]);
      auto c = _mm256_load_ps((const float*)&input[i + 2 * quarter_samples]);
      auto d = _mm256_load_ps((const float*)&input[i + 3 * quarter_samples]);

      auto r0 = _mm256_add_ps(a, c);
      auto r1 = _mm256_sub_ps(a, c);
      auto r2 = _mm256_add_ps(b, d);
      auto r3 = _mm256_sub_ps(b, d);
      r3 = _mm256_xor_ps(_mm256_permute_ps(r3, _MM_SHUFFLE(2, 3, 0, 1)), flip_signs);

      auto o0 = _mm256_add_ps(r0, r2); // { 0, 4, 8, 12 }
      auto o1 = _mm256_add_ps(r1, r3); // { 1, 5, 9, 13 }
      auto o2 = _mm256_sub_ps(r0, r2); // { 2, 6, 10, 14 }
      auto o3 = _mm256_sub_ps(r1, r3); // { 3, 7, 11, 15 }

      // Transpose
      auto o0o1_lo = (__m256)_mm256_unpacklo_pd((__m256d)o0, (__m256d)o1); // { 0, 1, 8, 9 }
      auto o0o1_hi = (__m256)_mm256_unpackhi_pd((__m256d)o0, (__m256d)o1); // { 4, 5, 12, 13 }
      auto o2o3_lo = (__m256)_mm256_unpacklo_pd((__m256d)o2, (__m256d)o3); // { 2, 3, 10, 11 }
      auto o2o3_hi = (__m256)_mm256_unpackhi_pd((__m256d)o2, (__m256d)o3); // { 6, 7, 14, 15 }
      o0 = _mm256_permute2f128_ps(o0o1_lo, o2o3_lo, (2 << 4) | (0 << 0));  // { 0, 1, 2, 3 }
      o1 = _mm256_permute2f128_ps(o0o1_hi, o2o3_hi, (2 << 4) | (0 << 0));  // { 4, 5, 6, 7 }
      o2 = _mm256_permute2f128_ps(o0o1_lo, o2o3_lo, (3 << 4) | (1 << 0));  // { 8, 9, 10, 11 }
      o3 = _mm256_permute2f128_ps(o0o1_hi, o2o3_hi, (3 << 4) | (1 << 0));  // { 12, 13, 14, 15 }

      unsigned j = i << 2;
      _mm256_store_ps((float*)&output[j +  0], o0);
      _mm256_store_ps((float*)&output[j +  4], o1);
      _mm256_store_ps((float*)&output[j +  8], o2);
      _mm256_store_ps((float*)&output[j + 12], o3);
   }
#else
   unsigned quarter_samples = samples >> 2;
   for (unsigned i = 0; i < quarter_samples; i++)
   {
      auto a = input[i];
      auto b = input[i + quarter_samples];
      auto c = input[i + 2 * quarter_samples];
      auto d = input[i + 3 * quarter_samples];

      auto r0 = a + c;
      auto r1 = a - c;
      auto r2 = b + d;
      auto r3 = b - d;
      r3 = complex<float>(r3.imag(), -r3.real());

      unsigned j = i << 2;
      output[j + 0] = r0 + r2;
      output[j + 1] = r1 + r3;
      output[j + 2] = r0 - r2;
      output[j + 3] = r1 - r3;
   }
#endif
}

static void __attribute__((noinline)) fft_forward_radix4_generic(complex<float> *output, const complex<float> *input,
      const complex<float> *twiddles, unsigned p, unsigned samples)
{
#if SIMD
   unsigned quarter_samples = samples >> 2;

   for (unsigned i = 0; i < quarter_samples; i += 4)
   {
      unsigned k = i & (p - 1);

      auto w = _mm256_load_ps((const float*)&twiddles[k]);
      auto w0 = _mm256_load_ps((const float*)&twiddles[p + k]);
      auto w1 = _mm256_load_ps((const float*)&twiddles[2 * p + k]);

      auto a = _mm256_load_ps((const float*)&input[i]);
      auto b = _mm256_load_ps((const float*)&input[i + quarter_samples]);
      auto c = _mm256_load_ps((const float*)&input[i + 2 * quarter_samples]);
      auto d = _mm256_load_ps((const float*)&input[i + 3 * quarter_samples]);

      c = _mm256_cmul_ps(c, w);
      d = _mm256_cmul_ps(d, w);

      auto r0 = _mm256_add_ps(a, c);
      auto r1 = _mm256_sub_ps(a, c);
      auto r2 = _mm256_add_ps(b, d);
      auto r3 = _mm256_sub_ps(b, d);

      r2 = _mm256_cmul_ps(r2, w0);
      r3 = _mm256_cmul_ps(r3, w1);

      auto o0 = _mm256_add_ps(r0, r2);
      auto o1 = _mm256_sub_ps(r0, r2);
      auto o2 = _mm256_add_ps(r1, r3);
      auto o3 = _mm256_sub_ps(r1, r3);

      unsigned j = ((i - k) << 2) + k;
      _mm256_store_ps((float*)&output[j + 0], o0);
      _mm256_store_ps((float*)&output[j + 1 * p], o2);
      _mm256_store_ps((float*)&output[j + 2 * p], o1);
      _mm256_store_ps((float*)&output[j + 3 * p], o3);
   }
#else
   unsigned quarter_samples = samples >> 2;
   for (unsigned i = 0; i < quarter_samples; i++)
   {
      unsigned k = i & (p - 1);

      auto a = input[i];
      auto b = input[i + quarter_samples];
      auto c = twiddles[k] * input[i + 2 * quarter_samples];
      auto d = twiddles[k] * input[i + 3 * quarter_samples];

      // DFT-2
      auto r0 = a + c;
      auto r1 = a - c;
      auto r2 = b + d;
      auto r3 = b - d;

      r2 *= twiddles[p + k];
      r3 *= twiddles[p + k + p];

      // DFT-2
      auto o0 = r0 + r2;
      auto o1 = r0 - r2;
      auto o2 = r1 + r3;
      auto o3 = r1 - r3;

      unsigned j = ((i - k) << 2) + k;
      output[j +     0] = o0;
      output[j + 1 * p] = o2;
      output[j + 2 * p] = o1;
      output[j + 3 * p] = o3;
   }
#endif
}

static void __attribute__((noinline)) fft_forward_radix2_p1(complex<float> *output, const complex<float> *input,
      unsigned samples)
{
#if SIMD
   unsigned half_samples = samples >> 1;
   for (unsigned i = 0; i < half_samples; i += 4)
   {
      auto a = _mm256_load_ps((const float*)&input[i]);
      auto b = _mm256_load_ps((const float*)&input[i + half_samples]);

      auto r0 = _mm256_add_ps(a, b); // { 0, 2, 4, 6 }
      auto r1 = _mm256_sub_ps(a, b); // { 1, 3, 5, 7 }
      a = (__m256)_mm256_unpacklo_pd((__m256d)r0, (__m256d)r1); // { 0, 1, 4, 5 }
      b = (__m256)_mm256_unpackhi_pd((__m256d)r0, (__m256d)r1); // { 2, 3, 6, 7 }
      r0 = _mm256_permute2f128_ps(a, b, (2 << 4) | (0 << 0)); // { 0, 1, 2, 3 }
      r1 = _mm256_permute2f128_ps(a, b, (3 << 4) | (1 << 0)); // { 4, 5, 6, 7 }

      unsigned j = i << 1;
      _mm256_store_ps((float*)&output[j + 0], r0);
      _mm256_store_ps((float*)&output[j + 4], r1);
   }
#else
   unsigned half_samples = samples >> 1;
   for (unsigned i = 0; i < half_samples; i++)
   {
      auto a = input[i];
      auto b = input[i + half_samples]; 

      unsigned j = i << 1;
      output[j + 0] = a + b;
      output[j + 1] = a - b;
   }
#endif
}

static void __attribute__((noinline)) fft_forward_radix2_p2(complex<float> *output, const complex<float> *input,
      const complex<float> *twiddles, unsigned samples)
{
#if SIMD
   unsigned half_samples = samples >> 1;
   const auto flip_signs = _mm256_set_ps(-0.0f, 0.0f, 0.0f, 0.0f, -0.0f, 0.0f, 0.0f, 0.0f);

   for (unsigned i = 0; i < half_samples; i += 4)
   {
      auto a = _mm256_load_ps((const float*)&input[i]);
      auto b = _mm256_load_ps((const float*)&input[i + half_samples]);
      b = _mm256_xor_ps(_mm256_permute_ps(b, _MM_SHUFFLE(2, 3, 1, 0)), flip_signs);

      auto r0 = _mm256_add_ps(a, b); // { c0, c1, c4, c5 }
      auto r1 = _mm256_sub_ps(a, b); // { c2, c3, c6, c7 }
      a = _mm256_permute2f128_ps(r0, r1, (2 << 4) | (0 << 0));
      b = _mm256_permute2f128_ps(r0, r1, (3 << 4) | (1 << 0));

      unsigned j = i << 1;
      _mm256_store_ps((float*)&output[j + 0], a);
      _mm256_store_ps((float*)&output[j + 4], b);
   }
#else
   unsigned half_samples = samples >> 1;
   for (unsigned i = 0; i < half_samples; i++)
   {
      unsigned k = i & (2 - 1);
      auto a = input[i];
      auto b = twiddles[k] * input[i + half_samples];

      unsigned j = (i << 1) - k;
      output[j + 0] = a + b;
      output[j + 2] = a - b;
   }
#endif
}

static void __attribute__((noinline)) fft_forward_radix2_generic(complex<float> *output, const complex<float> *input,
      const complex<float> *twiddles, unsigned p, unsigned samples)
{
#if SIMD
   unsigned half_samples = samples >> 1;

   for (unsigned i = 0; i < half_samples; i += 4)
   {
      unsigned k = i & (p - 1);

      auto w = _mm256_load_ps((const float*)&twiddles[k]);
      auto a = _mm256_load_ps((const float*)&input[i]);
      auto b = _mm256_load_ps((const float*)&input[i + half_samples]);
      b = _mm256_cmul_ps(b, w);

      auto r0 = _mm256_add_ps(a, b);
      auto r1 = _mm256_sub_ps(a, b);

      unsigned j = (i << 1) - k;
      _mm256_store_ps((float*)&output[j + 0], r0);
      _mm256_store_ps((float*)&output[j + p], r1);
   }
#else
   unsigned half_samples = samples >> 1;
   for (unsigned i = 0; i < half_samples; i++)
   {
      unsigned k = i & (p - 1);
      auto a = input[i];
      auto b = twiddles[k] * input[i + half_samples];

      unsigned j = (i << 1) - k;
      output[j + 0] = a + b;
      output[j + p] = a - b;
   }
#endif
}

int main()
{
   alignas(64) complex<float> twiddles[N];
   alignas(64) complex<float> input[N];
   alignas(64) complex<float> tmp0[N];
   alignas(64) complex<float> tmp1[N];

   auto *pt = twiddles;
   for (unsigned p = 1; p < N; p <<= 1)
   {
      for (unsigned k = 0; k < p; k++)
         pt[k] = twiddle(-1, k, p);
      pt += p;
      if (p == 2)
         pt++;
   }

   srand(0);
   for (unsigned i = 0; i < N; i++)
   {
      float real = float(rand()) / RAND_MAX - 0.5f;
      float imag = float(rand()) / RAND_MAX - 0.5f;
      input[i] = complex<float>(real, imag);
   }

#if RADIX2
   // Radix-2

   for (unsigned i = 0; i < ITERATIONS; i++)
   {
      pt = twiddles;

      fft_forward_radix2_p1(tmp0, input, N);
      pt += 1;
      fft_forward_radix2_p2(tmp1, tmp0, pt, N);
      pt += 3;

      auto *out = tmp0;
      auto *in = tmp1;
      for (unsigned p = 4; p < N; p <<= 1)
      {
         fft_forward_radix2_generic(out, in, pt, p, N);
         pt += p;
         swap(out, in);
      }

#if DEBUG
      for (unsigned i = 0; i < N; i++)
         printf("Radix-2 FFT[%03u] = (%+8.3f, %+8.3f)\n", i, in[i].real(), in[i].imag());
#endif
   }
#endif

#if RADIX4
   // Radix-4

   for (unsigned i = 0; i < ITERATIONS; i++)
   {
      pt = twiddles;

      fft_forward_radix4_p1(tmp0, input, N);
      pt += 4;
      auto *out = tmp1;
      auto *in = tmp0;

      for (unsigned p = 4; p < N; p <<= 2)
      {
         fft_forward_radix4_generic(out, in, pt, p, N);
         swap(out, in);
         pt += p * 3;
      }

#if DEBUG
      for (unsigned i = 0; i < N; i++)
         printf("Radix-4 FFT[%03u] = (%+8.3f, %+8.3f)\n", i, in[i].real(), in[i].imag());
#endif
   }
#endif

#if RADIX8
   // Radix-8

   for (unsigned i = 0; i < ITERATIONS; i++)
   {
      pt = twiddles;

      fft_forward_radix8_p1(tmp0, input, pt, N);
      pt += 8;
      auto *out = tmp1;
      auto *in = tmp0;

      for (unsigned p = 8; p < N; p <<= 3)
      {
         fft_forward_radix8_generic(out, in, pt, p, N);
         swap(out, in);
         pt += p * 7;
      }

#if DEBUG
      for (unsigned i = 0; i < N; i++)
         printf("Radix-8 FFT[%03u] = (%+8.3f, %+8.3f)\n", i, in[i].real(), in[i].imag());
#endif
   }
#endif

#if FFTW
   complex<float> *in, *out;
   fftwf_plan p;
   in = (complex<float>*)fftwf_malloc(sizeof(fftw_complex) * N);
   out = (complex<float>*)fftwf_malloc(sizeof(fftw_complex) * N);

   p = fftwf_plan_dft_1d(N, (fftwf_complex*)in, (fftwf_complex*)out, FFTW_FORWARD, FFTW_MEASURE);
   if (!p)
      return 1;

   memcpy(in, input, sizeof(input));
   for (unsigned i = 0; i < ITERATIONS; i++)
   {
      fftwf_execute(p);
#if DEBUG
      for (unsigned i = 0; i < N; i++)
         printf("FFTW FFT[%03u] = (%+8.3f, %+8.3f)\n", i, out[i].real(), out[i].imag());
#endif
   }
   fftwf_destroy_plan(p);
   fftwf_free(in);
   fftwf_free(out);
#endif
}

