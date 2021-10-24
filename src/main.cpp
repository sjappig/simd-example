#include <iostream>
#include <immintrin.h>
#include <string.h>
#include "data.hpp"

namespace data {
int16_t y[data::yLen + 16]{};
}

// filter[0] * x[4 + t] +
// filter[1] * x[3 + t] +
// filter[2] * x[2 + t] +
// filter[3] * x[1 + t] +
// filter[4] * x[0 + t] = result[t]
int16_t* naive(const int16_t* x, int16_t* y, size_t yLen, const int16_t* filter) {
    for (auto t = 0; t < yLen; ++t) {
        y[t] = 0;
        for (auto i = 0; i < data::filterLen; ++i) {
            y[t] += filter[i] * x[(data::filterLen - 1) + t - i];
        }
    }
    return y;
}

int16_t* dumbSse2(const int16_t* x, int16_t* y, size_t yLen, const int16_t* filter) {
    const __m128i mFilter = _mm_set_epi16(0, 0, 0, filter[4], filter[3], filter[2], filter[2], filter[1]);
    for (auto t = 0; t < yLen; ++t) {
        __m128i input = _mm_set_epi16(x[t], x[t + 1], x[t + 2], x[t + 3], x[t + 4], 0, 0, 0);
        __m128i tmp = _mm_madd_epi16(mFilter, input);
        tmp = _mm_hadd_epi32(tmp, tmp);
        tmp = _mm_hadd_epi32(tmp, tmp);
        y[t] = _mm_extract_epi32(tmp, 3);
    }
    return y;
}

int16_t* sse2(const int16_t* x, int16_t* y, size_t yLen, const int16_t* filter) {
    // filter has to be zero-padded and reversed
    const __m128i mFilter =_mm_loadu_si128((__m128i*)filter);
    for (auto t = 0; t < yLen; ++t) {
        __m128i input =_mm_loadu_si128((__m128i*)&x[t]);
        __m128i tmp = _mm_madd_epi16(mFilter, input);
        tmp = _mm_hadd_epi32(tmp, tmp);
        tmp = _mm_hadd_epi32(tmp, tmp);
        y[t] = _mm_extract_epi32(tmp, 3);
    }
    return y;
}

int16_t* avx2(const int16_t* x, int16_t* y, size_t yLen, const int16_t* filter) {
    const __m256i mFilter = _mm256_loadu2_m128i((__m128i*)filter, (__m128i*)filter);
    for (auto t = 0; t < yLen; t += 2) {
        __m256i input = _mm256_loadu2_m128i((__m128i*)&x[t + 1], (__m128i*)&x[t]);
        __m256i tmp = _mm256_madd_epi16(mFilter, input);
        tmp = _mm256_hadd_epi32(tmp, tmp);
        tmp = _mm256_hadd_epi32(tmp, tmp);
        y[t + 0] = _mm256_extract_epi32(tmp, 3);
        y[t + 1] = _mm256_extract_epi32(tmp, 7);
    }
    return y;
}

int16_t* smartAvx2(const int16_t* x, int16_t* y, size_t yLen, const int16_t* filter) {
    __m256i mFilter[data::filterLen];
    for (auto i = 0; i < data::filterLen; ++i) {
        mFilter[i] = _mm256_set1_epi16(filter[i]);
    }
    __m256i input[data::filterLen];
    for (auto t = 0; t < yLen; t += 16) {
        for (auto i = 0; i < data::filterLen; ++i) {
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
    int16_t* (*targetFunction)(const int16_t*, int16_t*, size_t, const int16_t*) = naive;

    bool shouldValidate = false;
    if (argc > 1)
    {
        std::string firstArg{argv[1]};

        if (argc > 2) {
            std::string secondArg{argv[2]};
            if (secondArg == "--validate") {
                shouldValidate = true;
            }
        }
        if (firstArg == "--naive") {
            targetFunction = naive;
        }
        else if (firstArg == "--dumbSse2") {
            targetFunction = dumbSse2;
        }
        else if (firstArg == "--sse2") {
            targetFunction = sse2;
        }
        else if (firstArg == "--avx2") {
            targetFunction = avx2;
        }
        else if (firstArg == "--smartAvx2") {
            targetFunction = smartAvx2;
        }
        else {
            std::cerr << "usage: " << argv[0] << " --naive|--dumbSse2|--sse2|--avx2|--smartAvx2 [--validate]" << std::endl;
            std::cerr << "unknown argument: " << firstArg << std::endl;
            return -1;
        }
    }
    uint64_t startTicks = __rdtsc();
    uint64_t durationTicks = 0;
    uint64_t runCount = 0;
    do
    {
        auto* result = targetFunction(&data::x[0], &data::y[0], data::yLen, &data::filter[0]);
        if (shouldValidate and not validate(result, data::yLen)) {
            return -1;
        }
        ++runCount;
        durationTicks = __rdtsc() - startTicks;
    } while (durationTicks < 10000000000);

    std::cout << "number of runs: " << runCount << std::endl;

    return 0;
}

