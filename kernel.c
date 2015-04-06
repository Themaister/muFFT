#include "fft_internal.h"

void mufft_forward_radix2_p1_c(void *output_, const void *input_,
      const cfloat *twiddles, unsigned p, unsigned samples)
{
   cfloat *output = output_;
   const cfloat *input = input_;
   (void)twiddles;
   (void)p;

   unsigned half_samples = samples >> 1;
   for (unsigned i = 0; i < half_samples; i++)
   {
      cfloat a = input[i];
      cfloat b = input[i + half_samples]; 

      unsigned j = i << 1;
      output[j + 0] = a + b;
      output[j + 1] = a - b;
   }
}

void mufft_forward_radix2_p2_c(void *output_, const void *input_,
      const cfloat *twiddles, unsigned p, unsigned samples)
{
   cfloat *output = output_;
   const cfloat *input = input_;
   (void)twiddles;
   (void)p;

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
}

void mufft_radix2_generic_c(void *output_, const void *input_,
      const cfloat *twiddles, unsigned p, unsigned samples)
{
   cfloat *output = output_;
   const cfloat *input = input_;

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
}


void mufft_forward_radix4_p1_c(void *output_, const void *input_,
      const cfloat *twiddles, unsigned p, unsigned samples)
{
   cfloat *output = output_;
   const cfloat *input = input_;
   (void)twiddles;
   (void)p;

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
}

void mufft_radix4_generic_c(void *output_, const void *input_,
      const cfloat *twiddles, unsigned p, unsigned samples)
{
   cfloat *output = output_;
   const cfloat *input = input_;

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
}

void mufft_forward_radix8_p1_c(void *output_, const void *input_,
      const cfloat *twiddles, unsigned p, unsigned samples)
{
   cfloat *output = output_;
   const cfloat *input = input_;
   (void)twiddles;
   (void)p;

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
}

void mufft_radix8_generic_c(void *output_, const void *input_,
      const cfloat *twiddles, unsigned p, unsigned samples)
{
   cfloat *output = output_;
   const cfloat *input = input_;

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
}

void mufft_forward_radix2_p1_vert_c(void *output_, const void *input_,
      const cfloat *twiddles, unsigned p, unsigned samples_x, unsigned samples_y)
{
   cfloat *output = output_;
   const cfloat *input = input_;
   (void)twiddles;
   (void)p;

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
}

void mufft_radix2_generic_vert_c(void *output_, const void *input_,
      const cfloat *twiddles, unsigned p, unsigned samples_x, unsigned samples_y)
{
   cfloat *output = output_;
   const cfloat *input = input_;

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
}

void mufft_forward_radix4_p1_vert_c(void *output_, const void *input_,
      const cfloat *twiddles, unsigned p, unsigned samples_x, unsigned samples_y)
{
   cfloat *output = output_;
   const cfloat *input = input_;
   (void)twiddles;
   (void)p;

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
}

void mufft_radix4_generic_vert_c(void *output_, const void *input_,
      const cfloat *twiddles, unsigned p, unsigned samples_x, unsigned samples_y)
{
   cfloat *output = output_;
   const cfloat *input = input_;

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
}

void mufft_forward_radix8_p1_vert_c(void *output_, const void *input_,
      const cfloat *twiddles, unsigned p, unsigned samples_x, unsigned samples_y)
{
   cfloat *output = output_;
   const cfloat *input = input_;
   (void)p;

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
}

void mufft_radix8_generic_vert_c(void *output_, const void *input_,
      const cfloat *twiddles, unsigned p, unsigned samples_x, unsigned samples_y)
{
   cfloat *output = output_;
   const cfloat *input = input_;

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
}

