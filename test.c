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

#define MUFFT_DEBUG

#include "fft.h"
#include "fft_internal.h"
#include <complex.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include <fftw3.h> // Used as a reference.

static void test_fft_2d(unsigned Nx, unsigned Ny, int direction, unsigned flags)
{
    complex float *input = mufft_alloc(Nx * Ny * sizeof(complex float));
    complex float *output = mufft_alloc(Nx * Ny * sizeof(complex float));
    complex float *input_fftw = fftwf_malloc(Nx * Ny * sizeof(fftwf_complex));
    complex float *output_fftw = fftwf_malloc(Nx * Ny * sizeof(fftwf_complex));

    srand(0);
    for (unsigned i = 0; i < Nx * Ny; i++)
    {
        float real = (float)rand() / RAND_MAX - 0.5f;
        float imag = (float)rand() / RAND_MAX - 0.5f;
        input[i] = real + I * imag;
    }

    fftwf_plan plan = fftwf_plan_dft_2d(Ny, Nx, input_fftw, output_fftw,
            direction, FFTW_ESTIMATE);
    mufft_assert(plan != NULL);
    memcpy(input_fftw, input, Nx * Ny * sizeof(complex float));

    mufft_plan_2d *muplan = mufft_create_plan_2d_c2c(Nx, Ny, direction, flags);
    mufft_assert(muplan != NULL);

    fftwf_execute(plan);
    mufft_execute_plan_2d(muplan, output, input);

    const float epsilon = 0.000001f * sqrtf(Nx * Ny);
    for (unsigned i = 0; i < Nx * Ny; i++)
    {
        float delta = cabsf(output[i] - output_fftw[i]);
        mufft_assert(delta < epsilon);
    }

    mufft_free(input);
    mufft_free(output);
    mufft_free_plan_2d(muplan);
    fftwf_free(input_fftw);
    fftwf_free(output_fftw);
    fftwf_destroy_plan(plan);
}

static void test_fft_1d(unsigned N, int direction, unsigned flags)
{
    complex float *input = mufft_alloc(N * sizeof(complex float));
    complex float *output = mufft_alloc(N * sizeof(complex float));
    complex float *input_fftw = fftwf_malloc(N * sizeof(fftwf_complex));
    complex float *output_fftw = fftwf_malloc(N * sizeof(fftwf_complex));

    srand(0);
    for (unsigned i = 0; i < N; i++)
    {
        float real = (float)rand() / RAND_MAX - 0.5f;
        float imag = (float)rand() / RAND_MAX - 0.5f;
        input[i] = real + _Complex_I * imag;
    }

    fftwf_plan plan = fftwf_plan_dft_1d(N, input_fftw, output_fftw,
            direction, FFTW_ESTIMATE);
    mufft_assert(plan != NULL);
    memcpy(input_fftw, input, N * sizeof(complex float));

    mufft_plan_1d *muplan = mufft_create_plan_1d_c2c(N, direction, flags);
    mufft_assert(muplan != NULL);

    fftwf_execute(plan);
    mufft_execute_plan_1d(muplan, output, input);

    const float epsilon = 0.000001f * sqrtf(N);
    for (unsigned i = 0; i < N; i++)
    {
        float delta = cabsf(output[i] - output_fftw[i]);
        mufft_assert(delta < epsilon);
    }

    mufft_free(input);
    mufft_free(output);
    mufft_free_plan_1d(muplan);
    fftwf_free(input_fftw);
    fftwf_free(output_fftw);
    fftwf_destroy_plan(plan);
}

static void test_fft_1d_c2r(unsigned N, unsigned flags)
{
    unsigned fftN = N / 2 + 1;
    complex float *input = mufft_alloc(fftN * sizeof(complex float));
    float *output = mufft_alloc(N * sizeof(float));
    complex float *input_fftw = fftwf_malloc(fftN * sizeof(fftwf_complex));
    float *output_fftw = fftwf_malloc(N * sizeof(float));

    srand(0);
    input[0] = (float)rand() / RAND_MAX - 0.5f;
    for (unsigned i = 1; i < N / 2; i++)
    {
        float real = (float)rand() / RAND_MAX - 0.5f;
        float imag = (float)rand() / RAND_MAX - 0.5f;
        input[i] = real + _Complex_I * imag;
    }
    input[N / 2] = (float)rand() / RAND_MAX - 0.5f;

    fftwf_plan plan = fftwf_plan_dft_c2r_1d(N, input_fftw, output_fftw, FFTW_ESTIMATE);
    mufft_assert(plan != NULL);
    memcpy(input_fftw, input, fftN * sizeof(complex float));

    mufft_plan_1d *muplan = mufft_create_plan_1d_c2r(N, flags);
    mufft_assert(muplan != NULL);

    fftwf_execute(plan);
    mufft_execute_plan_1d(muplan, output, input);

    const float epsilon = 0.000001f * sqrtf(N);
    for (unsigned i = 0; i < N; i++)
    {
        float delta = output[i] - output_fftw[i];
        mufft_assert(delta < epsilon);
    }

    mufft_free(input);
    mufft_free(output);
    mufft_free_plan_1d(muplan);
    fftwf_free(input_fftw);
    fftwf_free(output_fftw);
    fftwf_destroy_plan(plan);
}

static void test_fft_1d_r2c(unsigned N, unsigned flags)
{
    unsigned fftN = N / 2 + 1;
    float *input = mufft_alloc(N * sizeof(float));
    complex float *output = mufft_alloc(N * sizeof(complex float));
    float *input_fftw = fftwf_malloc(N * sizeof(float));
    complex float *output_fftw = fftwf_malloc(fftN * sizeof(fftwf_complex));

    srand(0);
    for (unsigned i = 0; i < N; i++)
    {
        float real = (float)rand() / RAND_MAX - 0.5f;
        input[i] = real;
    }

    fftwf_plan plan = fftwf_plan_dft_r2c_1d(N, input_fftw, output_fftw, FFTW_ESTIMATE);
    mufft_assert(plan != NULL);
    memcpy(input_fftw, input, N * sizeof(float));

    mufft_plan_1d *muplan_full = mufft_create_plan_1d_r2c(N, flags | MUFFT_FLAG_FULL_R2C);
    mufft_assert(muplan_full != NULL);
    mufft_plan_1d *muplan = mufft_create_plan_1d_r2c(N, flags);
    mufft_assert(muplan != NULL);

    fftwf_execute(plan);
    mufft_execute_plan_1d(muplan, output, input);

    const float epsilon = 0.000001f * sqrtf(N);
    for (unsigned i = 0; i < fftN; i++)
    {
        float delta = cabsf(output[i] - output_fftw[i]);
        mufft_assert(delta < epsilon);
    }

    mufft_execute_plan_1d(muplan_full, output, input);

    for (unsigned i = 0; i < fftN; i++)
    {
        float delta = cabsf(output[i] - output_fftw[i]);
        mufft_assert(delta < epsilon);
    }

    // Verify stuff is properly conjugated as well.
    for (unsigned i = 1; i < N / 2; i++)
    {
        complex float a = output[i];
        complex float b = conjf(output[N - i]);
        float delta = cabsf(a - b);
        mufft_assert(delta < epsilon);
    }

    mufft_free(input);
    mufft_free(output);
    mufft_free_plan_1d(muplan);
    mufft_free_plan_1d(muplan_full);
    fftwf_free(input_fftw);
    fftwf_free(output_fftw);
    fftwf_destroy_plan(plan);
}

static void test_fft_1d_r2c_half(unsigned N, unsigned flags)
{
    unsigned fftN = N / 2 + 1;
    float *input = mufft_alloc((N / 2) * sizeof(float));
    complex float *output = mufft_alloc(N * sizeof(complex float));
    float *input_fftw = fftwf_malloc(N * sizeof(float));
    complex float *output_fftw = fftwf_malloc(fftN * sizeof(fftwf_complex));

    memset(input_fftw, 0, N * sizeof(float));

    srand(0);
    for (unsigned i = 0; i < N / 2; i++)
    {
        float real = (float)rand() / RAND_MAX - 0.5f;
        input[i] = real;
    }

    fftwf_plan plan = fftwf_plan_dft_r2c_1d(N, input_fftw, output_fftw, FFTW_ESTIMATE);
    mufft_assert(plan != NULL);
    memcpy(input_fftw, input, (N / 2) * sizeof(float));

    mufft_plan_1d *muplan_full = mufft_create_plan_1d_r2c(N, flags | MUFFT_FLAG_FULL_R2C | MUFFT_FLAG_ZERO_PAD_UPPER_HALF);
    mufft_assert(muplan_full != NULL);
    mufft_plan_1d *muplan = mufft_create_plan_1d_r2c(N, flags | MUFFT_FLAG_ZERO_PAD_UPPER_HALF);
    mufft_assert(muplan != NULL);

    fftwf_execute(plan);
    mufft_execute_plan_1d(muplan, output, input);

    const float epsilon = 0.000001f * sqrtf(N);
    for (unsigned i = 0; i < fftN; i++)
    {
        float delta = cabsf(output[i] - output_fftw[i]);
        mufft_assert(delta < epsilon);
    }

    mufft_execute_plan_1d(muplan_full, output, input);

    for (unsigned i = 0; i < fftN; i++)
    {
        float delta = cabsf(output[i] - output_fftw[i]);
        mufft_assert(delta < epsilon);
    }

    // Verify stuff is properly conjugated as well.
    for (unsigned i = 1; i < N / 2; i++)
    {
        complex float a = output[i];
        complex float b = conjf(output[N - i]);
        float delta = cabsf(a - b);
        mufft_assert(delta < epsilon);
    }

    mufft_free(input);
    mufft_free(output);
    mufft_free_plan_1d(muplan);
    mufft_free_plan_1d(muplan_full);
    fftwf_free(input_fftw);
    fftwf_free(output_fftw);
    fftwf_destroy_plan(plan);
}

int main(void)
{
    for (unsigned N = 2; N < 128 * 1024; N <<= 1)
    {
        for (unsigned flags = 0; flags < 8; flags++)
        {
            test_fft_1d(N, -1, flags);
            test_fft_1d(N, +1, flags);
        }
    }

    for (unsigned N = 4; N < 128 * 1024; N <<= 1)
    {
        for (unsigned flags = 0; flags < 8; flags++)
        {
            test_fft_1d_r2c(N, flags);
            test_fft_1d_r2c_half(N, flags);
            test_fft_1d_c2r(N, flags);
        }
    }

    for (unsigned Ny = 2; Ny < 1024; Ny <<= 1)
    {
        for (unsigned Nx = 2; Nx < 1024; Nx <<= 1)
        {
            for (unsigned flags = 0; flags < 8; flags++)
            {
                test_fft_2d(Nx, Ny, -1, flags);
                test_fft_2d(Nx, Ny, +1, flags);
            }
        }
    }

    fftwf_cleanup();
}

