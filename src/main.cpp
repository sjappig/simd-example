#include <iostream>
#include <immintrin.h>
#include <string.h>
#include "data.hpp"

int16_t y[data::yLen + 3]{};

void naive() {
    for (auto t = 0; t < data::yLen; ++t) {
        y[t] = 0;
        for (auto i = 0; i < data::filterLen; ++i) {
            y[t] += data::filter[i] * data::x[4 + t - i];
        }
    }
}

void sse2() {
    // filter[0] * x[4 + t]
    // filter[1] * x[3 + t]
    // filter[2] * x[2 + t]
    // filter[3] * x[1 + t]
    // filter[4] * x[t]
    // reverse filter (not really needed since it symmetrical)
    __m128i tmpFilter = _mm_set_epi16(data::filter[4], data::filter[3], data::filter[2], data::filter[1], data::filter[0], 0, 0, 0);
    for (auto t = 0; t < data::yLen; ++t) {
        auto x = _mm_set_epi16(data::x[t], data::x[t + 1], data::x[t + 2], data::x[t + 3], data::x[t + 4], 0, 0, 0);
        auto tmp = _mm_madd_epi16(tmpFilter, x);
        tmp = _mm_hadd_epi32(tmp, tmp);
        tmp = _mm_hadd_epi32(tmp, tmp);
        y[t] = _mm_extract_epi32(tmp, 0);
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
    void (*targetFunction)() = naive;

    if (argc > 1)
    {
        std::string firstArg{argv[1]};

        if (firstArg == "--naive") {
            targetFunction = naive;
        }
        if (firstArg == "--sse2") {
            targetFunction = sse2;
        }
        else {
            std::cerr << "usage: " << argv[0] << " [--naive|--sse2]" << std::endl;
            std::cerr << "unknown argument: " << firstArg << std::endl;
            return -1;
        }
    }
    for (uint64_t i = 0; i < 1000000; ++i) {
        targetFunction();
    }
    validateY();
    return 0;
}

