#include "data.hpp"

namespace data {

const int16_t filter[filterLen] = {-1, 2, 10, 2, -1};

const int16_t x[xLen + 2 * (filterLen - 1)] = {0, 0, 0, 0, // zero-padding
                                               -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
                                               0, 0, 0, 0  // zero-padding
                                              };

const int16_t yExpected[yLen] = {9, -10, -99, -106, -84, -72, -60, -48, -36, -24, -12, 0, 12, 24, 36, 48, 60, 72, 84, 106, 99, 10, -9};
}
