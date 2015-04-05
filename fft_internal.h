#ifndef MUFFT_INTERNAL_H__
#define MUFFT_INTERNAL_H__

#include <math.h>
#include <complex.h>

#ifndef M_PI
#define M_PI 3.14159265359
#endif

#ifndef M_SQRT_2
#define M_SQRT_2 0.707106781186547524401
#endif

#define SWAP(a, b) do { cfloat *tmp = b; b = a; a = tmp; } while(0)

#define ARRAY_SIZE(a) (sizeof(a) / sizeof((a)[0]))

#undef I
#define I _Complex_I

typedef complex float cfloat;

#endif

