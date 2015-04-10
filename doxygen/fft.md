# The Fast Fourier Transform {#FFT}

## Introduction

First, a brief introduction of what the Fourier Transform is and how we get to the Fast Fourier Transform, the practical application of the transform.
If you are familiar with fourier transforms, you can safely skip this section.
Don't expect rigorous derivations of the transform, there are plenty of resources online explaining the fourier transform.

## The Fourier Transform

The Fourier Transform is a mathematical transform which converts time-domain signals into the frequency domain (and back, as we will see later).
Related transforms are the discrete cosine transform (DCT) and discrete sine transform (DST) which do essentially the same thing.
Especially DCT has enjoyed extensive use in audio, image and video compression over the last decades.

The original definition of the fourier transform is an infinite integral.
The main idea is that all signals can be expressed as sum of infinite number of sinusoids and we can find the sinusoid frequency components with the transform:

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
We have Taylor expansions of some trancendental functions.

    exp(x) = 1 + x/1! + x^2/2! + x^3/3! + x^4/4! + ...
    cos(x) = 1 - x^2/2! + x^4/4! - x^6/6! + ...
    sin(x) = x - x^3/3! + x^5/5! - x^7/7! + ...

By working out the taylor expansions for `exp(ix)` we find an interesting result ...

    exp(ix) = cos(x) + isin(x)
    exp(-ix) = cos(x) - isin(x) = exp(ix)*

Basically, the complex exponential is a complex valued oscillator.
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

### The Discrete Time Fourier Transform (DFT)

Once we have made the signal repeating and made time discrete, we are now ready for our fourier transform since we have a finite number of frequencies (well, infinite, but the frequency components repeat due to aliasing so we don't care) in our signal.

Recall the integral

    X(w) = integrate from -inf to +inf: x(t)exp(-iwt) dt

Since we made the signal repeating, we change it to a bounded integral

    X(w) = integrate from 0 to T: x(t)exp(-iwt) dt

And since we made time discrete we can change the integral to a discrete sum:

    X(2 * pi * k / T) = sum n from 0 to T / D - 1:
        x(n * D) * exp(-i * (2 * pi * k / T) * (n * D)) = x(n * D) * exp(-i * 2 * pi * k * n * D / T)

T / D is the number of samples we have in our repeating signal, let's call it N to make it clearer and
let's call make our sampled input signal an array, x[n] = x(n * D). Finally, we only have N different frequencies due to our repeating spectrum, where k = [0, N - 1], so let's make that an array as well, X[k] = X(2 * pi * k / T).

    X[k] = sum n from 0 to N - 1: x[n] * exp(-i * 2 * pi * k * n / N)

And there we have it, the Discrete Time Fourier Transform.

### The inverse Discrete Time Fourier Transform

Going from frequency domain to time domain is very similar, we only need to slightly alter the formula.
Note that the phase of the exponential is inverted, otherwise the math looks very similar (the same).

    x[n] = sum k from 0 to N - 1: (1 / N) * X[k] * exp(+i * 2 * pi * k * n / N)

The 1 / N term here is just a normalization factor.
It is also needed in the continous time transform, but it was omitted intentionally.
muFFT omits this normalization step.
We don't always care about every coefficient being scaled.

### Speed of the DFT

A big problem with the naive DFT algorithm is its complexity. To evaluate every frequency coefficient,
we need O(n^2) operations, fortunately there are more efficient ways of doing this which algorithm is called the fast fourier transform (duh).

### Interpreting the results of the DFT

The DFT represents the strength of each frequency component in the input signal as well as its phase.
Both phase and amplitude are neatly encoded in a single complex number (one reason why complex numbers are so vitally important to any math dealing with waves of sorts!)

Every amplitude with angle can be expressed as a complex number

    ComplexNumber(amplitude, phase) = amplitude * exp(+i * phase)

### Fourier Transform of real input data

By far, the most common input data to the fourier transform is real input data.
Note that the definition of the fourier transform allows x[n] to be complex, but we will see that there are some redundancies in the DFT when x[n] is real.

    X[k]     = sum n: x[n] * exp(-i * 2 * pi * k * n / N)
    X[N - k] = sum n: x[n] * exp(-i * 2 * pi * (N - k) * n / N)
             = sum n: x[n] * exp(-i * 2 * pi * -k * n / N) * exp(-i * 2 * pi * N / N)
             = sum n: x[n] * exp(-i * 2 * pi * -k * n / N) * 1
             = sum n: x[n] * exp(+i * 2 * pi * k * n / N)
             = (sum n: x[n] * exp(-i * 2 * pi * k * n / N))*
             = (X[k])* // This only works because x[n] is real because x[n]* = x[n]!
             = X[k]*

And we have an interesting result.
The frequency components are perfectly mirrored around X[N / 2] except for the trivial change that the frequency component is conjugated.

We only really need to compute X[0] up to and including X[N / 2]. Due to the symmetry

    X[N / 2] = X[N / 2]*
    
X[N / 2] must be real as well, but this is a minor point.

## Fast Fourier Transform

The Fast Fourier Transform is an optimization which allows us to get O(nlogn) performance rather than DFT O(n^2).

This optimization is realized by finding some recursive patterns in the computation.
First, let's have a quick look at the DFT formula again.

    X[k] = sum n: x[n] * exp(-i * 2 * pi * k * n / N)

Writing the full exponential gets a bit annoying so let

    W(x, N) = exp(-i * 2 * pi * x / N)

so that we can rewrite the sum as

    X[k] = sum n: x[n] * W(k * n, N)

Lets split this sum into even n and odd n

    X[k] = sum n: x[2n] * W(k * 2n, N) + x[2n + 1] * W(k * (2n + 1), N)

Since W(k + 1, N) = W(k, N) * W(1, N), we can factor out some stuff

    X[k] = sum n: x[2n] * W(k * 2n, N)    + W(k, N) * x[2n + 1] * W(k * n, N / 2)
         = sum n: x[2n] * W(k * n, N / 2) + W(k, N) * x[2n + 1] * W(k * n, N / 2)
         = Xeven[k] + W(k, N) * Xodd[k]

Some interesting things happen if we try X[k + N / 2]

    X[k + N / 2]
         = sum n: x[2n] * W((k + N / 2) * n, N / 2) + W((k + N / 2), N) * x[2n + 1] * W((k + N / 2) * n, N / 2)
         = sum n: x[2n] * W(k * n, N / 2) - W(k, N) * x[2n + 1] * W(k * n, N / 2)
         = Xeven[k] - W(k, N) * Xodd[k]

All the exponentials either repeat themselves or simply invert.
We can now essentially compute two frequency samples by taking two smaller DFTs and either add or subtract the right hand side.

This definition is recursive as well. We can keep splitting Xeven and Xodd into even and odd DFTs and do this same optimization over and over until we end up with a N = 2 DFT.
This assumes that the transform size is power-of-two which is often a reasonable assumption.
Note that it is possible to split into any factor you want, every third, every fourth, every fifth and so on, and the math will look very similar.

Let's try to illustrate this for a DFT of length 8.

               _______
    x[0] ---- |       |--------------\----/-----X[0]
    x[2] ---- | DFT   |-------------\-\--/-/----X[1]
    x[4] ---- | N = 4 |------------\-\-\/-/-/---X[2]
    x[6] ---- |_______|-----------\-\-\/\/-/-/--X[3]
               _______             \ \/\/\/ /
    x[1] ---- |       |-- W(0, 8) --\/\/\/\/----X[4]
    x[3] ---- | DFT   |-- W(1, 8) ---\/\/\/-----X[5]
    x[5] ---- | N = 4 |-- W(2, 8) ----\/\/------X[6]
    x[7] ---- |_______|-- W(3, 8) -----\/-------X[7]

