#include <cstdint>
#include <cstddef>

namespace data {
constexpr size_t filterLen = 5;
extern const int16_t filter[];

constexpr size_t xLen = 19;
extern const int16_t x[];

constexpr size_t yLen = filterLen + xLen - 1;
extern const int16_t yExpected[];
}
