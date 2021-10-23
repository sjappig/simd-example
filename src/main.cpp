#include <iostream>
#include <immintrin.h>
#include <string.h>
#include "data.hpp"

int16_t y[data::yLen + 3]{};

// filter[0] * x[4 + t]
// filter[1] * x[3 + t]
// filter[2] * x[2 + t]
// filter[3] * x[1 + t]
void naive(int n) {
    for (int round = 0; round < n; ++round) {
        for (auto t = 0; t < data::yLen; ++t) {
            y[t] = 0;
            for (auto i = 0; i < data::filterLen; ++i) {
                y[t] += data::filter[i] * data::x[4 + t - i];
            }
        }
    }
}

void sse2(int n) {
    // reverse filter (not really needed since it symmetrical)
    const __m128i filter = _mm_set_epi16(0, 0, 0, data::filter[4], data::filter[3], data::filter[2], data::filter[1], data::filter[0]);
    const __m128i mask = _mm_set_epi16(0, 0, 0, -1, -1, -1, -1, -1);
    for (int round = 0; round < n; ++round) {
        for (auto t = 0; t < data::yLen; ++t) {
            //__m128i x = _mm_set_epi16(data::x[t], data::x[t + 1], data::x[t + 2], data::x[t + 3], data::x[t + 4], 0, 0, 0);
            __m128i x =_mm_loadu_si128((__m128i*)&data::x[t]);
            x = _mm_and_si128(x, mask);
            __m128i tmp = _mm_madd_epi16(filter, x);
            tmp = _mm_hadd_epi32(tmp, tmp);
            tmp = _mm_hadd_epi32(tmp, tmp);
            y[t] = _mm_extract_epi32(tmp, 3);
        }
    }
}

void avx2(int n) {
    const __m256i filter = _mm256_set_epi16(
            0, 0, 0, data::filter[4], data::filter[3], data::filter[2], data::filter[1], data::filter[0],
            0, 0, 0, data::filter[4], data::filter[3], data::filter[2], data::filter[1], data::filter[0]
    );
    const __m256i mask = _mm256_set_epi16(
            0, 0, 0, -1, -1, -1, -1, -1,
            0, 0, 0, -1, -1, -1, -1, -1
    );
    for (int round = 0; round < n; ++round) {
        for (auto t = 0; t < data::yLen; t += 2) {
            /*__m256i x = _mm256_set_epi16(
                    data::x[t + 1], data::x[t + 2], data::x[t + 3], data::x[t + 4], data::x[t + 5], 0, 0, 0,
                    data::x[t + 0], data::x[t + 1], data::x[t + 2], data::x[t + 3], data::x[t + 4], 0, 0, 0
            );*/
            __m256i x = _mm256_loadu2_m128i((__m128i*)&data::x[t + 1], (__m128i*)&data::x[t]);
            x = _mm256_and_si256(x, mask);
            __m256i tmp = _mm256_madd_epi16(filter, x);
            tmp = _mm256_hadd_epi32(tmp, tmp);
            tmp = _mm256_hadd_epi32(tmp, tmp);
            y[t + 0] = _mm256_extract_epi32(tmp, 3);
            y[t + 1] = _mm256_extract_epi32(tmp, 7);
        }
    }
}

void validateY() {
    for (auto t = 0; t < data::yLen; ++t) {
        if (data::yExpected[t] != y[t]) {
            std::cout << "Error at index " << t << ", got " << static_cast<int>(y[t]) << ", expected " << static_cast<int>(data::yExpected[t]) << std::endl;
        }
    }
}

int main(int argc, char** argv) {
    void (*targetFunction)(int) = naive;

    if (argc > 1)
    {
        std::string firstArg{argv[1]};

        if (firstArg == "--naive") {
            targetFunction = naive;
        }
        else if (firstArg == "--sse2") {
            targetFunction = sse2;
        }
        else if (firstArg == "--avx2") {
            targetFunction = avx2;
        }
        else {
            std::cerr << "usage: " << argv[0] << " [--naive|--sse2|--avx2]" << std::endl;
            std::cerr << "unknown argument: " << firstArg << std::endl;
            return -1;
        }
    }
    targetFunction(1);
    validateY();

    targetFunction(10000000);
    return 0;
}

