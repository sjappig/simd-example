#include <iostream>
#include <immintrin.h>
#include <string.h>
#include "data.hpp"

namespace data {
int16_t y[data::yLen + 16]{};
}

// h[0] * x[4 + t] +
// h[1] * x[3 + t] +
// h[2] * x[2 + t] +
// h[3] * x[1 + t] +
// h[4] * x[0 + t] = y[t]
int16_t* naive(const int16_t* __restrict__ x, int16_t* __restrict__ y, size_t yLen) {
    for (auto t = 0; t < yLen; ++t) {
        y[t] = 0;
        for (auto i = 0; i < data::hLen; ++i) {
            y[t] += data::h[i] * x[(data::hLen - 1) - i + t];
        }
    }
    return y;
}

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

// filter[0] * x[t + 0] + filter[1] * x[t + 1] + ... + filter[4] * x[t + 4]
// filter[0] * x[t + 1] + filter[1] * x[t + 2] + ... + filter[4] * x[t + 5]
// filter[0] * x[t + 2] + filter[1] * x[t + 3] + ... + filter[4] * x[t + 6]
// ...
// filter[0] * x[t + 7] + filter[1] * x[t + 8] + ... + filter[4] * x[t + 11]
int16_t* smartSse(const int16_t* x, int16_t* y, size_t yLen) {
    __m128i mFilter[data::hLen];
    // should revert the filter
    for (auto i = 0; i < data::hLen; ++i) {
        mFilter[i] = _mm_set1_epi16(data::h[i]);
    }
    __m128i input[data::hLen];
    // data has to be zero-padded
    for (auto t = 0; t < yLen; t += 8) {
        for (auto i = 0; i < data::hLen; ++i) {
            input[i] = _mm_loadu_si128((__m128i*)&x[t + i]);
            input[i] = _mm_mullo_epi16(input[i], mFilter[i]);
            if (i > 0) {
                input[0] = _mm_add_epi16(input[0], input[i]);
            }
        }
        _mm_storeu_si128((__m128i*)&y[t], input[0]);
    }
    return y;
}

int16_t* smartAvx2(const int16_t* __restrict__ x, int16_t* __restrict__ y, size_t yLen) {
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

bool validate(int16_t* y, size_t yLen) {
    for (auto t = 0; t < yLen; ++t) {
        if (data::yExpected[t] != y[t]) {
            std::cerr << "Error at index " << t << ", got " << static_cast<int>(y[t]) << ", expected " << static_cast<int>(data::yExpected[t]) << std::endl;
            return false;
        }
    }
    return true;
}

int main(int argc, char** argv) {
    int16_t* (*targetFunction)(const int16_t*, int16_t*, size_t) = nullptr;

    bool shouldValidate = false;
    if (argc > 1)
    {
        std::string firstArg{argv[1]};

        if (firstArg == "--naive") {
            targetFunction = naive;
        }
        else if (firstArg == "--dumbSse") {
            targetFunction = dumbSse;
        }
        else if (firstArg == "--sse") {
            targetFunction = sse;
        }
        else if (firstArg == "--smartSse") {
            targetFunction = smartSse;
        }
        else if (firstArg == "--smartAvx2") {
            targetFunction = smartAvx2;
        }

        if (argc > 2) {
            if (std::string{argv[2]} == "--validate") {
                shouldValidate = true;
            }
            else {
                targetFunction = nullptr;
            }
        }
    }

    if (not targetFunction) {
        std::cerr << "usage: " << argv[0] << " --naive|--dumbSse|--sse|--smartSse|--smartAvx2 [--validate]" << std::endl;
        return -1;
    }

    uint64_t startTicks = __rdtsc();
    uint64_t durationCycles = 0;
    uint64_t runCount = 0;
    do
    {
        auto* result = targetFunction(&data::x[0], &data::y[0], data::yLen);
        if (shouldValidate and not validate(result, data::yLen)) {
            return -1;
        }
        ++runCount;
        durationCycles = __rdtsc() - startTicks;
    } while (durationCycles < 1000000000);

    std::cout << "Cycles per convolution (averaged over " << runCount << " runs): " << durationCycles/(double)runCount;
    if (shouldValidate) {
        std::cout << " (incl. validation)";
    }
    std::cout << std::endl;

    return 0;
}

