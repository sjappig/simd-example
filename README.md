# Optimizing C++ with x86 SIMD

Nowadays modern CPUs support so-called SIMD (single instruction, multiple data [1]) architecture, where single instructions
are capable of processing multiple data elements in parellel. SIMD capability came to normal desktops with Intel's MMX
and AMD's 3DNow! - I remember seeing advertisements of e.g. Intel Pentium MMX -processors in the end of 90's, without
understanding what it meant but knowing that it would improve my gaming experience in PC. And that was pretty much what
they were made for: boost the calculations needed for awesome graphics. Their usability is not limited to graphics
though; most computing intensive tasks that can be parallelized benefits from those.

The SIMD extensions are all about introducing new registers and instructions for manipulating those registers. As an
example, MMX introduced eight 64 bit registers (MM0-MM7) and operations to manipulate those their values as they would
be e.g. four 16 bit integers.

In this text we try to boost our simple C++ program by using SIMD intrisics manually.

# Problem to solve

The problem here is chosen to be one that should have high chance of getting performance boost from SIMD: direct
convolution. The convolution is a fundamental operation with many applications. See e.g. [2] if it is not familiar to
you already. The values here are chosen pretty arbitrarily but chosen to be integers with 16 bit range for the sake of
simplicity. The data generation was done with Octave:

    octave:3> h = [-1 2 10 2 -1];
    octave:4>
    octave:5> y = conv(h,x);

Our target with C++ is now to calculate y when h and x are given.

# Naive C++ implementation

To get the first benchmark (and to build all the necessary validations and measuring) we start with the most
straight-forward implementation of convolution one can have:

    int16_t* naive(const int16_t* x, int16_t* y, size_t yLen) {
        for (auto t = 0; t < yLen; ++t) {
            y[t] = 0;
            for (auto i = 0; i < data::hLen; ++i) {
                y[t] += data::h[i] * x[(data::hLen - 1) - i + t];
            }
        }
        return y;
    }

The data is zero-padded where needed to prevent prevent buffer overflows. The mechanisms around the actual convolution
implementations can be found in the source code, and include choosing the implementation based on a command line flag,
optional validation (also based on a command line flag) of the output and measuring needed cycles per run. Note that the
implementation has potential integer overflow in it, which we however ignore here.

# First SIMD implementation

We start our SIMD experiments with 128 bit register introduced in SSE [3]. Instructions used here require capability up to
SSE4.1 (_mm_extract_epi_32) [4]. As this is our first version, we start by just getting rid of the inner loop of our naive
implementation.

Using 128 bit registers means that we have enough space for eigth 16 bit values; however, since our filter has only
five values, we waste three elements in each round with this implementation. If you wonder the argument order in the
_mm_set_epi16, the explanation is that it takes the least significant word as the last one. Hence we get one register
with first 80 bits filled with filter weights, and other register with 80 bits filled with the reversed input elements.

The implementation uses then the multiply-add to multiply the filter weights with input and do the first pair-wise
additions. Note how the outcome of the multiply-add for two 16 bit integers is one 32 bit integer. Unfortunately the
multiply-add does not do the whole summation, so we have to do two more pair-wise additions (_mm_hadd_epi32), before the
sum over the whole range can be read from the first 32 bits of the register.

    int16_t* dumbSse(const int16_t* x, int16_t* y, size_t yLen) {
        const __m128i mFilter = _mm_set_epi16(0, 0, 0, data::h[4], data::h[3], data::h[2], data::h[1], data::h[0]);
        for (auto t = 0; t < yLen; ++t) {
            __m128i input = _mm_set_epi16(0, 0, 0, x[t], x[t + 1], x[t + 2], x[t + 3], x[t + 4]);
            __m128i tmp = _mm_madd_epi16(mFilter, input);
            tmp = _mm_hadd_epi32(tmp, tmp);
            tmp = _mm_hadd_epi32(tmp, tmp);
            y[t] = _mm_extract_epi32(tmp, 0);
        }
        return y;
    }

# Benchmarking the first SIMD version

We start with compiler optimizations turned off and enabling all intruction subsets supported by the local machine
(-march=native).

    jaripekkary@jaripekkary-Latitude-E7440:~/projects/simd-example$ make
    clang++ -march=native -g -O0   -c -o src/data.o src/data.cpp
    clang++ -march=native -g -O0   -c -o src/main.o src/main.cpp
    clang++  src/data.o src/main.o   -o simd-example
    jaripekkary@jaripekkary-Latitude-E7440:~/projects/simd-example$ ./simd-example --naive
    Cycles per convolution (averaged over 13821 runs): 72354.4
    jaripekkary@jaripekkary-Latitude-E7440:~/projects/simd-example$ ./simd-example --dumbSse
    Cycles per convolution (averaged over 13497 runs): 74093.6

The results seem highly discouraging! We are not even able to beat non-optimized for-loop. Luckily, some investigation
and trials reveal that _mm_set_epi16 is rather slow, and there are load-instruction _mm_loadu_si128 that should be more
efficient.

    int16_t* sse(const int16_t* x, int16_t* y, size_t yLen) {
        // filter has to be zero-padded and should be reversed: however, our filter is symmetrical,
        // so the reverse is not really needed
        const __m128i mFilter = _mm_loadu_si128((__m128i*)&data::h[0]);
        for (auto t = 0; t < yLen; ++t) {
            __m128i input =_mm_loadu_si128((__m128i*)&x[t]);
            __m128i tmp = _mm_madd_epi16(mFilter, input);
            tmp = _mm_hadd_epi32(tmp, tmp);
            tmp = _mm_hadd_epi32(tmp, tmp);
            y[t] = _mm_extract_epi32(tmp, 3);
        }
        return y;
    }

jaripekkary@jaripekkary-Latitude-E7440:~/projects/simd-example$ ./simd-example --naive
Cycles per convolution (averaged over 14045 runs): 71202.9
jaripekkary@jaripekkary-Latitude-E7440:~/projects/simd-example$ ./simd-example --sse
Cycles per convolution (averaged over 23754 runs): 42099.6

Finally, this looks more promising. However, comparing non-optimized versions is only our first step. Let's see how
these compare when we optimize for speed.

    jaripekkary@jaripekkary-Latitude-E7440:~/projects/simd-example$ make clean && make -e OPTIMIZATION_LEVEL=fast
    rm -f src/*.o
    rm -f simd-example
    clang++ -march=native -g -Ofast   -c -o src/data.o src/data.cpp
    clang++ -march=native -g -Ofast   -c -o src/main.o src/main.cpp
    clang++  src/data.o src/main.o   -o simd-example
    jaripekkary@jaripekkary-Latitude-E7440:~/projects/simd-example$ ./simd-example --naive
    Cycles per convolution (averaged over 726030 runs): 1377.35
    jaripekkary@jaripekkary-Latitude-E7440:~/projects/simd-example$ ./simd-example --sse
    Cycles per convolution (averaged over 115116 runs): 8686.93

Compiler beats us hands down. We can see that also our SSE-version has benefit from the optimizations, but clearly the
use of the intrinsics has prevented some of the most powerful optimizations. To understand the situation better, we have
to take a look at the generated assembly.

jaripekkary@jaripekkary-Latitude-E7440:~/projects/simd-example$ make -e OPTIMIZATION_LEVEL=fast main.s
clang++ -march=native -g -Ofast -S -fverbose-asm src/main.cpp

Looking at the main.s and the naive-function there, we can already conclude a lot by looking how the filter is used
there:

	movzwl	_ZN4data1hE(%rip), %r8d
	movzwl	_ZN4data1hE+2(%rip), %r9d
	movzwl	_ZN4data1hE+4(%rip), %r10d
	movzwl	_ZN4data1hE+6(%rip), %r11d
	movzwl	_ZN4data1hE+8(%rip), %ecx

    ...

	vmovd	%r8d, %xmm0
	vpbroadcastw	%xmm0, %ymm0
	vmovd	%r9d, %xmm1
	vpbroadcastw	%xmm1, %ymm1
	vmovd	%r10d, %xmm2
	vpbroadcastw	%xmm2, %ymm2
	vmovd	%r11d, %xmm3
	vpbroadcastw	%xmm3, %ymm3
	vmovd	%ecx, %xmm4
	vpbroadcastw	%xmm4, %ymm4

So the compiler is already using SIMD, and it is even using the newer subset AVX which has 256 bit registers. It is also
organizing the filter weights in a smarter way, using one register per weight and "broadcasting" the weight to fill the
whole register. This makes it possible to process 16 input elements (16 * 16 = 256) parallel in a few steps: multiplying
the input with each weight-filled register and adding those up. Using these ideas, we can create our own AVX
SIMD-version (note that we also have to take care that the input and the output arrays have enough space so that the
last round does not overflow).

    int16_t* smartAvx2(const int16_t* x, int16_t* y, size_t yLen) {
        __m256i mFilter[data::hLen];
        for (auto i = 0; i < data::hLen; ++i) {
            mFilter[i] = _mm256_set1_epi16(data::h[i]);
        }
        __m256i input[data::hLen];
        for (auto t = 0; t < yLen; t += 16) {
            for (auto i = 0; i < data::hLen; ++i) {
                input[i] = _mm256_loadu_si256((__m256i*)&x[t + i]);
                input[i] = _mm256_mullo_epi16(input[i], mFilter[i]);
                if (i > 0) {
                    input[0] = _mm256_add_epi16(input[0], input[i]);
                }
            }
            _mm256_storeu_si256((__m256i*)&y[t], input[0]);
        }
        return y;
    }

And now we are actually beating the optimized naive-version:

    jaripekkary@jaripekkary-Latitude-E7440:~/projects/simd-example$ ./simd-example --naive
    Cycles per convolution (averaged over 733863 runs): 1362.65
    jaripekkary@jaripekkary-Latitude-E7440:~/projects/simd-example$ ./simd-example --smartAvx2
    Cycles per convolution (averaged over 1743057 runs): 573.705

However, the results do not make sense; we should have similar results as the optimized version, as we use the same
approach with same level of parallelization. It turns out that the compiler does not know if the input and the output
arrays will overlap, and it has to therefore produce assembly that is writing and reading from those arrays after each
calculation step.

        #DEBUG_VALUE: naive:y <- $rax
        #DEBUG_VALUE: t <- 0
        #DEBUG_VALUE: naive:yLen <- $rdx
        #DEBUG_VALUE: naive:x <- $rdi
        .loc	7 17 14                 # src/main.cpp:17:14
        vmovups	%ymm5, (%rax,%rbx,2)
    .Ltmp19:
        .loc	7 19 32                 # src/main.cpp:19:32
        vpmullw	8(%rdi,%rbx,2), %ymm0, %ymm6
        .loc	7 19 18 is_stmt 0       # src/main.cpp:19:18
        vmovdqu	%ymm6, (%rax,%rbx,2)
        .loc	7 19 32                 # src/main.cpp:19:32
        vpmullw	6(%rdi,%rbx,2), %ymm1, %ymm7
        .loc	7 19 18                 # src/main.cpp:19:18
        vpaddw	%ymm6, %ymm7, %ymm6
        vmovdqu	%ymm6, (%rax,%rbx,2)

For this we luckily have simple solution. We can aid the compiler by marking pointers as restricted, if we can guarantee
that they will not overlap:

    int16_t* naive(const int16_t* __restrict__ x, int16_t* __restrict__ y, size_t yLen)

With that change, the results are matching:

    jaripekkary@jaripekkary-Latitude-E7440:~/projects/simd-example$ ./simd-example --naive
    Cycles per convolution (averaged over 1749345 runs): 571.643
    jaripekkary@jaripekkary-Latitude-E7440:~/projects/simd-example$ ./simd-example --smartAvx2
    Cycles per convolution (averaged over 1751001 runs): 571.102

# Conclusions

We managed to produce same performance with our highly-optimized-non-readable smartAvx2-function as we got from the
naive-function when optimizations were turned on and the compiler was aided a bit by using restricted pointers. This is
a very good example of the perils of premature optimization: if this would have been a real-life project, deciding first
to optimize the code with intrinsics would have been a huge waste of time, when the most readable version would provide
the same performance with very small modifications. As the often-heard wisdom goes: Trust you compiler!

Since our problem to solve here was as basic operation as convolution, it is very likely that the compiler has been
tested with that during its development and the optimization for convolution is top notch. It might very well be that
the compiler would fail with some other, more complex cases; however, one should still always start with the most
readable "naive"-version, and see if the compiler magic would be enough.

# References

[1] https://en.wikipedia.org/wiki/SIMD

[2] https://betterexplained.com/articles/intuitive-convolution/

[3] https://en.wikipedia.org/wiki/Streaming_SIMD_Extensions

[4] https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html
