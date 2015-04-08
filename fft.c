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

#include "fft.h"
#include "fft_internal.h"
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <stdint.h>
#include <assert.h>

struct mufft_step_base
{
    void (*func)(void);
    unsigned radix;
    unsigned p;
    unsigned twiddle_offset;
};

struct mufft_step_1d
{
    mufft_1d_func func;
    unsigned radix;
    unsigned p;
    unsigned twiddle_offset;
};

struct mufft_step_2d
{
    mufft_2d_func func;
    unsigned radix;
    unsigned p;
    unsigned twiddle_offset;
};

struct mufft_plan_1d
{
    struct mufft_step_1d *steps;
    unsigned num_steps;
    unsigned N;

    cfloat *tmp_buffer;
    cfloat *twiddles;

    mufft_r2c_resolve_func r2c_resolve;
    mufft_r2c_resolve_func c2r_resolve;
    cfloat *r2c_twiddles;
};

struct mufft_plan_2d
{
    struct mufft_step_1d *steps_x;
    unsigned num_steps_x;
    struct mufft_step_2d *steps_y;
    unsigned num_steps_y;
    unsigned Nx, Ny;

    cfloat *tmp_buffer;
    cfloat *twiddles_x;
    cfloat *twiddles_y;
};

static cfloat twiddle(int direction, int k, int p)
{
    double phase = (M_PI * direction * k) / p;
    return cos(phase) + I * sin(phase);
}

static cfloat *build_twiddles(unsigned N, int direction)
{
    cfloat *twiddles = mufft_alloc(N * sizeof(cfloat));
    if (twiddles == NULL)
    {
        return NULL;
    }

    cfloat *pt = twiddles;

    for (unsigned p = 1; p < N; p <<= 1)
    {
        for (unsigned k = 0; k < p; k++)
        {
            pt[k] = twiddle(direction, k, p);
        }
        pt += p == 2 ? 3 : p; // Make sure that twiddles for p == 4 and up are aligned properly for AVX.
    }

    return twiddles;
}

struct fft_step_base
{
    void (*func)(void);
    unsigned radix;
};

struct fft_step_1d
{
    mufft_1d_func func;
    unsigned radix;
    unsigned minimum_elements;
    unsigned fixed_p;
    unsigned minimum_p;
    unsigned flags;
};

struct fft_step_2d
{
    mufft_2d_func func;
    unsigned radix;
    unsigned minimum_elements_x;
    unsigned minimum_elements_y;
    unsigned fixed_p;
    unsigned minimum_p;
    unsigned flags;
};

struct fft_r2c_resolve_step
{
    mufft_r2c_resolve_func func;
    unsigned minimum_elements;
    unsigned flags;
};

struct fft_r2c_resolve_step fft_r2c_resolve_table[] = {
#define STAMP_CPU_RESOLVE(arch, ext, min_x) \
    { .flags = arch | MUFFT_FLAG_FULL_R2C, \
        .func = mufft_resolve_r2c_full_ ## ext, .minimum_elements = 2 * min_x }, \
    { .flags = arch | MUFFT_FLAG_R2C, \
        .func = mufft_resolve_r2c_ ## ext, .minimum_elements = 2 * min_x }, \
    { .flags = arch | MUFFT_FLAG_C2R, \
        .func = mufft_resolve_c2r_ ## ext, .minimum_elements = 2 * min_x }

#ifdef MUFFT_HAVE_AVX
    STAMP_CPU_RESOLVE(MUFFT_FLAG_CPU_AVX, avx, 4),
#endif
#ifdef MUFFT_HAVE_SSE3
    STAMP_CPU_RESOLVE(MUFFT_FLAG_CPU_SSE3, sse3, 2),
#endif
#ifdef MUFFT_HAVE_SSE
    STAMP_CPU_RESOLVE(MUFFT_FLAG_CPU_SSE, sse, 2),
#endif
    STAMP_CPU_RESOLVE(0, c, 1),
};

static const struct fft_step_1d fft_1d_table[] = {
#define STAMP_CPU_1D(arch, ext, min_x) \
    { .flags = arch | MUFFT_FLAG_DIRECTION_FORWARD | MUFFT_FLAG_NO_ZERO_PAD_UPPER_HALF, \
        .func = mufft_forward_radix8_p1_ ## ext, .minimum_elements = 8 * min_x, .radix = 8, .fixed_p = 1, .minimum_p = -1u }, \
    { .flags = arch | MUFFT_FLAG_DIRECTION_FORWARD | MUFFT_FLAG_NO_ZERO_PAD_UPPER_HALF, \
        .func = mufft_forward_radix4_p1_ ## ext, .minimum_elements = 4 * min_x, .radix = 4, .fixed_p = 1, .minimum_p = -1u }, \
    { .flags = arch | MUFFT_FLAG_DIRECTION_ANY | MUFFT_FLAG_NO_ZERO_PAD_UPPER_HALF, \
        .func = mufft_radix2_p1_ ## ext, .minimum_elements = 2 * min_x, .radix = 2, .fixed_p = 1, .minimum_p = -1u }, \
    { .flags = arch | MUFFT_FLAG_DIRECTION_FORWARD | MUFFT_FLAG_ZERO_PAD_UPPER_HALF, \
        .func = mufft_forward_half_radix8_p1_ ## ext, .minimum_elements = 8 * min_x, .radix = 8, .fixed_p = 1, .minimum_p = -1u }, \
    { .flags = arch | MUFFT_FLAG_DIRECTION_FORWARD | MUFFT_FLAG_ZERO_PAD_UPPER_HALF, \
        .func = mufft_forward_half_radix4_p1_ ## ext, .minimum_elements = 4 * min_x, .radix = 4, .fixed_p = 1, .minimum_p = -1u }, \
    { .flags = arch | MUFFT_FLAG_DIRECTION_ANY | MUFFT_FLAG_ZERO_PAD_UPPER_HALF, \
        .func = mufft_radix2_half_p1_ ## ext, .minimum_elements = 2 * min_x, .radix = 2, .fixed_p = 1, .minimum_p = -1u }, \
    { .flags = arch | MUFFT_FLAG_DIRECTION_FORWARD, \
        .func = mufft_forward_radix2_p2_ ## ext, .minimum_elements = 2 * min_x, .radix = 2, .fixed_p = 2, .minimum_p = -1u }, \
    { .flags = arch | MUFFT_FLAG_DIRECTION_INVERSE, \
        .func = mufft_inverse_radix8_p1_ ## ext, .minimum_elements = 8 * min_x, .radix = 8, .fixed_p = 1, .minimum_p = -1u }, \
    { .flags = arch | MUFFT_FLAG_DIRECTION_INVERSE, \
        .func = mufft_inverse_radix4_p1_ ## ext, .minimum_elements = 4 * min_x, .radix = 4, .fixed_p = 1, .minimum_p = -1u }, \
    { .flags = arch | MUFFT_FLAG_DIRECTION_INVERSE, \
        .func = mufft_inverse_radix2_p2_ ## ext, .minimum_elements = 2 * min_x, .radix = 2, .fixed_p = 2, .minimum_p = -1u }, \
    { .flags = arch | MUFFT_FLAG_DIRECTION_ANY, \
        .func = mufft_radix8_generic_ ## ext, .minimum_elements = 8 * min_x, .radix = 8, .minimum_p = 8 }, \
    { .flags = arch | MUFFT_FLAG_DIRECTION_ANY, \
        .func = mufft_radix4_generic_ ## ext, .minimum_elements = 4 * min_x, .radix = 4, .minimum_p = 4 }, \
    { .flags = arch | MUFFT_FLAG_DIRECTION_ANY, \
        .func = mufft_radix2_generic_ ## ext, .minimum_elements = 2 * min_x, .radix = 2, .minimum_p = 4 }

#ifdef MUFFT_HAVE_AVX
    STAMP_CPU_1D(MUFFT_FLAG_CPU_AVX, avx, 4),
#endif
#ifdef MUFFT_HAVE_SSE3
    STAMP_CPU_1D(MUFFT_FLAG_CPU_SSE3, sse3, 2),
#endif
#ifdef MUFFT_HAVE_SSE
    STAMP_CPU_1D(MUFFT_FLAG_CPU_SSE, sse, 2),
#endif
    STAMP_CPU_1D(0, c, 1),
};

static const struct fft_step_2d fft_2d_table[] = {
#define STAMP_CPU_2D(arch, ext, min_x) \
    { .flags = arch | MUFFT_FLAG_DIRECTION_FORWARD, \
        .func = mufft_forward_radix8_p1_vert_ ## ext, .minimum_elements_x = min_x, .minimum_elements_y = 8, .radix = 8, .fixed_p = 1, .minimum_p = -1u }, \
    { .flags = arch | MUFFT_FLAG_DIRECTION_FORWARD, \
        .func = mufft_forward_radix4_p1_vert_ ## ext, .minimum_elements_x = min_x, .minimum_elements_y = 4, .radix = 4, .fixed_p = 1, .minimum_p = -1u }, \
    { .flags = arch | MUFFT_FLAG_DIRECTION_ANY, \
        .func = mufft_radix2_p1_vert_ ## ext, .minimum_elements_x = min_x, .minimum_elements_y = 2, .radix = 2, .fixed_p = 1, .minimum_p = -1u }, \
    { .flags = arch | MUFFT_FLAG_DIRECTION_INVERSE, \
        .func = mufft_inverse_radix8_p1_vert_ ## ext, .minimum_elements_x = min_x, .minimum_elements_y = 8, .radix = 8, .fixed_p = 1, .minimum_p = -1u }, \
    { .flags = arch | MUFFT_FLAG_DIRECTION_INVERSE, \
        .func = mufft_inverse_radix4_p1_vert_ ## ext, .minimum_elements_x = min_x, .minimum_elements_y = 4, .radix = 4, .fixed_p = 1, .minimum_p = -1u }, \
    { .flags = arch | MUFFT_FLAG_DIRECTION_ANY, \
        .func = mufft_radix8_generic_vert_ ## ext, .minimum_elements_x = min_x, .minimum_elements_y = 8, .radix = 8, .minimum_p = 8 }, \
    { .flags = arch | MUFFT_FLAG_DIRECTION_ANY, \
        .func = mufft_radix4_generic_vert_ ## ext, .minimum_elements_x = min_x, .minimum_elements_y = 4, .radix = 4, .minimum_p = 4 }, \
    { .flags = arch | MUFFT_FLAG_DIRECTION_ANY, \
        .func = mufft_radix2_generic_vert_ ## ext, .minimum_elements_x = min_x, .minimum_elements_y = 2, .radix = 2, .minimum_p = 2 }

#ifdef MUFFT_HAVE_AVX
    STAMP_CPU_2D(MUFFT_FLAG_CPU_AVX, avx, 4),
#endif
#ifdef MUFFT_HAVE_SSE3
    STAMP_CPU_2D(MUFFT_FLAG_CPU_SSE3, sse3, 2),
#endif
#ifdef MUFFT_HAVE_SSE
    STAMP_CPU_2D(MUFFT_FLAG_CPU_SSE, sse, 2),
#endif
    STAMP_CPU_2D(0, c, 1),
};

static bool add_step(struct mufft_step_base **steps, unsigned *num_steps,
        const struct fft_step_base *step, unsigned p)
{
    unsigned twiddle_offset = 0;
    if (*num_steps != 0)
    {
        struct mufft_step_base prev = (*steps)[*num_steps - 1];
        twiddle_offset = prev.twiddle_offset +
            (prev.p == 2 ? 3 : (prev.p * (prev.radix - 1)));

        // We skipped radix2 kernels, we have to add the padding twiddle here.
        if (p >= 4 && prev.p == 1)
        {
            twiddle_offset++;
        }
    }

    struct mufft_step_base *new_steps = realloc(*steps, (*num_steps + 1) * sizeof(*new_steps));
    if (new_steps == NULL)
    {
        return false;
    }

    *steps = new_steps;
    (*steps)[*num_steps] = (struct mufft_step_base) {
        .func = step->func,
        .radix = step->radix,
        .p = p,
        .twiddle_offset = twiddle_offset,
    };
    (*num_steps)++;
    return true;
}

static bool build_plan_1d(struct mufft_step_1d **steps, unsigned *num_steps, unsigned N, int direction, unsigned flags)
{
    unsigned radix = N;
    unsigned p = 1;

    unsigned step_flags = 0;
    switch (direction)
    {
        case MUFFT_FORWARD:
            step_flags |= MUFFT_FLAG_DIRECTION_FORWARD;
            break;

        case MUFFT_INVERSE:
            step_flags |= MUFFT_FLAG_DIRECTION_INVERSE;
            break;
    }

    // Add CPU flags. Just accept any CPU for now, but mask out flags we don't want.
    step_flags |= MUFFT_FLAG_MASK_CPU & ~(MUFFT_FLAG_CPU_NO_SIMD & flags);
    step_flags |= (flags & MUFFT_FLAG_ZERO_PAD_UPPER_HALF) != 0 ?
        MUFFT_FLAG_ZERO_PAD_UPPER_HALF : MUFFT_FLAG_NO_ZERO_PAD_UPPER_HALF;

    while (radix > 1)
    {
        bool found = false;

        // Find first (optimal?) routine which can do work.
        for (unsigned i = 0; i < ARRAY_SIZE(fft_1d_table); i++)
        {
            const struct fft_step_1d *step = &fft_1d_table[i];

            if (radix % step->radix == 0 &&
                    N >= step->minimum_elements &&
                    (step_flags & step->flags) == step->flags &&
                    (p >= step->minimum_p || p == step->fixed_p))
            {
                // Ugly casting, but add_step_1d and add_step_2d are ABI-wise exactly the same, and we don't have templates :(
                if (add_step((struct mufft_step_base**)steps, num_steps, (const struct fft_step_base*)step, p))
                {
                    found = true;
                    radix /= step->radix;
                    p *= step->radix;
                    break;
                }
            }
        }

        if (!found)
        {
            return false;
        }
    }

    return true;
}

static bool build_plan_2d(struct mufft_step_2d **steps, unsigned *num_steps, unsigned Nx, unsigned Ny, int direction, unsigned flags)
{
    unsigned radix = Ny;
    unsigned p = 1;

    unsigned step_flags = 0;
    switch (direction)
    {
        case MUFFT_FORWARD:
            step_flags |= MUFFT_FLAG_DIRECTION_FORWARD;
            break;

        case MUFFT_INVERSE:
            step_flags |= MUFFT_FLAG_DIRECTION_INVERSE;
            break;
    }
    // Add CPU flags. Just accept any CPU for now, but mask out flags we don't want.
    step_flags |= MUFFT_FLAG_MASK_CPU & ~(MUFFT_FLAG_CPU_NO_SIMD & flags);

    while (radix > 1)
    {
        bool found = false;

        // Find first (optimal?) routine which can do work.
        for (unsigned i = 0; i < ARRAY_SIZE(fft_2d_table); i++)
        {
            const struct fft_step_2d *step = &fft_2d_table[i];

            if (radix % step->radix == 0 &&
                    Ny >= step->minimum_elements_y &&
                    Nx >= step->minimum_elements_x &&
                    (step_flags & step->flags) == step->flags &&
                    (p >= step->minimum_p || p == step->fixed_p))
            {
                // Ugly casting, but add_step_1d and add_step_2d are ABI-wise exactly the same,
                // and we don't have templates :(
                if (add_step((struct mufft_step_base**)steps, num_steps, (const struct fft_step_base*)step, p))
                {
                    found = true;
                    radix /= step->radix;
                    p *= step->radix;
                    break;
                }
            }
        }

        if (!found)
        {
            return false;
        }
    }

    return true;
}

// The real-to-complex transform is implemented with a N / 2 complex transform with a
// final butterfly which extracts real/imag parts of the complex transform.
// See http://www.engineeringproductivitytools.com/stuff/T0001/PT10.HTM for details on algorithm.
mufft_plan_1d *mufft_create_plan_1d_r2c(unsigned N, unsigned flags)
{
    if ((N & (N - 1)) != 0 || N == 1)
    {
        return NULL;
    }

    unsigned complex_n = N / 2;

    mufft_plan_1d *plan = mufft_create_plan_1d_c2c(complex_n, MUFFT_FORWARD, flags);
    if (plan == NULL)
    {
        goto error;
    }

    plan->r2c_twiddles = mufft_alloc(complex_n * sizeof(cfloat));
    if (plan->r2c_twiddles == NULL)
    {
        goto error;
    }

    for (unsigned i = 0; i < complex_n; i++)
    {
        plan->r2c_twiddles[i] = -I * twiddle(-1, i, complex_n);
    }

    // Add CPU flags. Just accept any CPU for now, but mask out flags we don't want.
    unsigned resolve_flags = MUFFT_FLAG_MASK_CPU & ~(MUFFT_FLAG_CPU_NO_SIMD & flags);
    resolve_flags |= (flags & MUFFT_FLAG_FULL_R2C) != 0 ?
        MUFFT_FLAG_FULL_R2C : MUFFT_FLAG_R2C;

    for (unsigned i = 0; i < ARRAY_SIZE(fft_r2c_resolve_table); i++)
    {
        const struct fft_r2c_resolve_step *step = &fft_r2c_resolve_table[i];
        if ((step->flags & resolve_flags) == step->flags &&
                N >= step->minimum_elements)
        {
            plan->r2c_resolve = step->func;
            break;
        }
    }

    if (plan->r2c_resolve == NULL)
    {
        goto error;
    }

    return plan;

error:
    mufft_free_plan_1d(plan);
    return NULL;
}

mufft_plan_1d *mufft_create_plan_1d_c2r(unsigned N, unsigned flags)
{
    if ((N & (N - 1)) != 0 || N == 1)
    {
        return NULL;
    }

    unsigned complex_n = N / 2;

    mufft_plan_1d *plan = mufft_create_plan_1d_c2c(complex_n, MUFFT_INVERSE, flags);
    if (plan == NULL)
    {
        goto error;
    }

    plan->r2c_twiddles = mufft_alloc(complex_n * sizeof(cfloat));
    if (plan->r2c_twiddles == NULL)
    {
        goto error;
    }

    for (unsigned i = 0; i < complex_n; i++)
    {
        plan->r2c_twiddles[i] = I * twiddle(+1, i, complex_n);
    }

    // Add CPU flags. Just accept any CPU for now, but mask out flags we don't want.
    unsigned resolve_flags = MUFFT_FLAG_MASK_CPU & ~(MUFFT_FLAG_CPU_NO_SIMD & flags);
    resolve_flags |= MUFFT_FLAG_C2R;

    for (unsigned i = 0; i < ARRAY_SIZE(fft_r2c_resolve_table); i++)
    {
        const struct fft_r2c_resolve_step *step = &fft_r2c_resolve_table[i];
        if ((step->flags & resolve_flags) == step->flags &&
                N >= step->minimum_elements)
        {
            plan->c2r_resolve = step->func;
            break;
        }
    }

    if (plan->c2r_resolve == NULL)
    {
        goto error;
    }

    return plan;

error:
    mufft_free_plan_1d(plan);
    return NULL;
}


mufft_plan_1d *mufft_create_plan_1d_c2c(unsigned N, int direction, unsigned flags)
{
    if ((N & (N - 1)) != 0 || N == 1)
    {
        return NULL;
    }

    mufft_plan_1d *plan = mufft_calloc(sizeof(*plan));
    if (plan == NULL)
    {
        goto error;
    }

    plan->twiddles = build_twiddles(N, direction);
    if (plan->twiddles == NULL)
    {
        goto error;
    }

    plan->tmp_buffer = mufft_alloc(N * sizeof(cfloat));
    if (plan->tmp_buffer == NULL)
    {
        goto error;
    }

    if (!build_plan_1d(&plan->steps, &plan->num_steps, N, direction, flags))
    {
        goto error;
    }

    plan->N = N;
    return plan;

error:
    mufft_free_plan_1d(plan);
    return NULL;
}

mufft_plan_2d *mufft_create_plan_2d_c2c(unsigned Nx, unsigned Ny, int direction, unsigned flags)
{
    if ((Nx & (Nx - 1)) != 0 || (Ny & (Ny - 1)) != 0 || Nx == 1 || Ny == 1)
    {
        return NULL;
    }

    mufft_plan_2d *plan = mufft_calloc(sizeof(*plan));
    if (plan == NULL)
    {
        goto error;
    }

    plan->twiddles_x = build_twiddles(Nx, direction);
    plan->twiddles_y = build_twiddles(Ny, direction);
    if (plan->twiddles_x == NULL || plan->twiddles_y == NULL)
    {
        goto error;
    }

    plan->tmp_buffer = mufft_alloc(Nx * Ny * sizeof(cfloat));
    if (plan->tmp_buffer == NULL)
    {
        goto error;
    }

    if (!build_plan_1d(&plan->steps_x, &plan->num_steps_x, Nx, direction, flags))
    {
        goto error;
    }

    if (!build_plan_2d(&plan->steps_y, &plan->num_steps_y, Nx, Ny, direction, flags))
    {
        goto error;
    }

    plan->Nx = Nx;
    plan->Ny = Ny;
    return plan;

error:
    mufft_free_plan_2d(plan);
    return NULL;
}


void mufft_execute_plan_1d(mufft_plan_1d *plan, void *output, const void *input)
{
    const cfloat *pt = plan->twiddles;
    cfloat *out = output;
    cfloat *in = plan->tmp_buffer;
    unsigned N = plan->N;

    // If we're doing real-to-complex, we need an extra step.
    unsigned steps = plan->num_steps + (plan->r2c_resolve != NULL);

    // We want final step to write to output.
    if ((steps & 1) == 1)
    {
        SWAP(out, in);
    }

    const struct mufft_step_1d *first_step = &plan->steps[0];
    if (plan->c2r_resolve != NULL)
    {
        plan->c2r_resolve(out, input, plan->r2c_twiddles, N);
        first_step->func(in, out, pt, 1, N);
    }
    else
    {
        first_step->func(in, input, pt, 1, N);
    }

    for (unsigned i = 1; i < plan->num_steps; i++)
    {
        const struct mufft_step_1d *step = &plan->steps[i];
        step->func(out, in, pt + step->twiddle_offset, step->p, N);
        SWAP(out, in);
    }

    // Do Real-to-complex butterfly resolve.
    if (plan->r2c_resolve != NULL)
    {
        plan->r2c_resolve(out, in, plan->r2c_twiddles, N);
    }
}

void mufft_execute_plan_2d(mufft_plan_2d *plan, void *output, const void *input_)
{
    const cfloat *ptx = plan->twiddles_x;
    const cfloat *pty = plan->twiddles_y;
    const cfloat *input = input_;

    unsigned Nx = plan->Nx;
    unsigned Ny = plan->Ny;

    cfloat *hout = output;
    cfloat *hin = plan->tmp_buffer;
    if ((plan->num_steps_y & 1) == 0)
    {
        SWAP(hout, hin);
    }

    cfloat *out = hin;
    cfloat *in = hout;
    if ((plan->num_steps_x & 1) == 1)
    {
        SWAP(out, in);
    }

    // First, horizontal transforms over all lines individually.
    for (unsigned y = 0; y < Ny; y++)
    {
        cfloat *tin = in;
        cfloat *tout = out;

        const struct mufft_step_1d *first_step = &plan->steps_x[0];
        first_step->func(tin + y * Nx, input + y * Nx, ptx, 1, Nx);

        for (unsigned i = 1; i < plan->num_steps_x; i++)
        {
            const struct mufft_step_1d *step = &plan->steps_x[i];
            step->func(tout + y * Nx, tin + y * Nx, ptx + step->twiddle_offset, step->p, Nx);
            SWAP(tout, tin);
        }

        mufft_assert(tin == hin);
    }

    // Vertical transforms.
    const struct mufft_step_2d *first_step = &plan->steps_y[0];
    first_step->func(hout, hin, pty, 1, Nx, Ny);
    SWAP(hout, hin);

    for (unsigned i = 1; i < plan->num_steps_y; i++)
    {
        const struct mufft_step_2d *step = &plan->steps_y[i];
        step->func(hout, hin, pty + step->twiddle_offset, step->p, Nx, Ny);
        SWAP(hout, hin);
    }

    mufft_assert(hin == output);
}

void mufft_free_plan_1d(mufft_plan_1d *plan)
{
    if (plan == NULL)
    {
        return;
    }
    free(plan->steps);
    mufft_free(plan->tmp_buffer);
    mufft_free(plan->twiddles);
    mufft_free(plan->r2c_twiddles);
    mufft_free(plan);
}

void mufft_free_plan_2d(mufft_plan_2d *plan)
{
    if (plan == NULL)
    {
        return;
    }
    free(plan->steps_x);
    free(plan->steps_y);
    mufft_free(plan->tmp_buffer);
    mufft_free(plan->twiddles_x);
    mufft_free(plan->twiddles_y);
    mufft_free(plan);
}

void *mufft_alloc(size_t size)
{
#if defined(_ISOC11_SOURCE)
    return aligned_alloc(MUFFT_ALIGNMENT, size);
#elif (_POSIX_C_SOURCE >= 200112L) || (_XOPEN_SOURCE >= 600)
    void *ptr = NULL;
    if (posix_memalign(&ptr, MUFFT_ALIGNMENT, size) < 0)
    {
        return NULL;
    }
    return ptr;
#else
    // Align stuff ourselves. Kinda ugly, but will work anywhere.
    void **place;
    uintptr_t addr = 0;
    void *ptr = malloc(MUFFT_ALIGNMENT + size + sizeof(uintptr_t));

    if (ptr == NULL)
    {
        return NULL;
    }

    addr = ((uintptr_t)ptr + sizeof(uintptr_t) + MUFFT_ALIGNMENT)
        & ~(MUFFT_ALIGNMENT - 1);
    place = (void**)addr;
    place[-1] = ptr;

    return (void*)addr;
#endif
}

void *mufft_calloc(size_t size)
{
    void *ptr = mufft_alloc(size);
    if (ptr != NULL)
    {
        memset(ptr, 0, size);
    }
    return ptr;
}

void mufft_free(void *ptr)
{
#if !defined(_ISOC11_SOURCE) && !((_POSIX_C_SOURCE >= 200112L) || (_XOPEN_SOURCE >= 600))
    if (ptr != NULL)
    {
        void **p = (void**)ptr;
        free(p[-1]);
    }
#else
    free(ptr);
#endif
}

