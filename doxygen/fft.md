# The Fast Fourier Transform {#FFT}

## Introduction

First, a brief introduction of what the Fourier Transform is and how we get to the Fast Fourier Transform, the practical application of the transform.
If you are familiar with fourier transforms, you can safely skip this section.
Don't expect rigorous derivations of the transform, there are plenty of resources online explaining the fourier transform.

## The Fourier Transform

The Fourier Transform is a mathematical transform which converts time-domain signals into the frequency domain (and back, as we will see later).
Its original definition is an infinite integral.
The main idea is that all signals can be expressed as sum of infinite number of sinusoids.

    X(w) = integrate from -inf to +inf: x(t)exp(-iwt) dt

where `w` is angular frequency `2 * pi * frequency` and `i` is the imaginary constant.

### Imaginary numbers and imaginary exponents

To recap imaginary numbers, they are two-dimensional numbers which consist of a real part, and an imaginary part.

    c = a + ib

The imaginary constant is defined as sqrt(-1).
Arithmetic with it is fairly obvious.

    (a + ib) + (c + id) = a + c + i(b + d)
    (a + ib) - (c + id) = a - c + i(b - d)
    (a + ib) * (c + id) = ac + ib + id + i^2 bd = ac - bd + i(b + d)

The conjugate is a common operation. It simply flips the sign of the imaginary component. It's often noted with trailing asterisk.

    (a + ib)* = a - ib

Complex exponentials are fairly interesting.
Recall Taylor expansions of trancendental functions.

    exp(x) = 1 + x/1! + x^2/2! + x^3/3! + x^4/4! + ...
    cos(x) = 1 - x^2/2! + x^4/4! - x^6/6! + ...
    sin(x) = x - x^3/3! + x^5/5! - x^7/7! + ...

By working out the taylor expansions for `exp(ix)` we find an interesting result ...

    exp(ix) = cos(x) + isin(x)
    exp(-ix) = cos(x) - isin(x) = exp(ix)*

Basically, the complex exponential is a complex oscillator.
This makes sense, since the fourier transform is essentially correlating the input signal against various oscillators, which allows us to extract frequency components.
Complex oscillators can distinguish between positive frequencies and negative frequencies (which will become relevant in a bit),
but other than that they're pretty much ye olde sinusoids.

### Continous frequency domain to discrete frequency domain

To make the fourier transform practical, we need to assume that our signal is repeating in some way.
Lets assume we have a signal x(t) defined from t = 0 to T. x(t + T) = x(t).
It is fairly easy to show that we can now only have frequencies which are a multiple of (1 / T) in our signal.
If we try to reconstruct the repeating x(t) pattern using sinusoids, we must use sinusoids which also repeat in the same pattern, i.e. frequencies multiple of 1 / T.

### Infinite discrete frequencies to finite discrete frequencies

The final change to make things computable is to make our time domain discrete as well (i.e. sampling).
When we make the time domain discrete we repeat the frequency spectrum with a period of (1 / sampling interval).
If our sampling interval is D, we get aliasing where

    X(w) = X(w + 2 * pi * (1 / D))

If you know your Nyquist you might be confused because you know that the maximum reconstructible frequency is (sampling frequency / 2) and not (sampling frequency), but don't fear.
Let's assume sampling interval is 1 s, i.e. 1 Hz and we sample a signal that is 0.6 Hz.

    X(2 * pi * 0.6) = X(2 * pi * (-0.4))

The 0.6 Hz component is aliased into -0.4 Hz (negative frequency). We can think of our repeated frequency spectrum as [0, 1/D] or [-0.5/D, +0.5/D], the end result is the same.
For real input signals negative and positive frequencies are pretty much the same thing (just inverted phase), so we rarely consider negative frequencies but they are distinct for complex signals.


## The Discrete Time Fourier Transform

The FFT is a well known transform. Its main purpose is to transform time-domain samples into the frequency domain.

##

