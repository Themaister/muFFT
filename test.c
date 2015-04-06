#include "fft.h"
#include <complex.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#include <fftw3.h> // Used as a reference.

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
    memcpy(input_fftw, input, N * sizeof(complex float));

    fftwf_plan plan = fftwf_plan_dft_1d(N, input_fftw, output_fftw,
            direction, FFTW_ESTIMATE);
    assert(plan != NULL);

    mufft_plan_1d *muplan = mufft_create_plan_1d_c2c(N, direction, flags);
    assert(muplan != NULL);

    fftwf_execute(plan);
    mufft_execute_plan_1d(muplan, output, input);

    const float epsilon = 0.0000001f * N;
    for (unsigned i = 0; i < N; i++)
    {
        complex float delta = cabsf(output[i] - output_fftw[i]);
        assert(crealf(delta) < epsilon);
        assert(cimagf(delta) < epsilon);
    }

    mufft_free(input);
    mufft_free(output);
    mufft_free_plan_1d(muplan);
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
}

