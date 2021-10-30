#include <cstdint>
#include <cstddef>

namespace data {
constexpr size_t hLen = 5;
extern const int16_t h[];

constexpr size_t xLen = 1999;
extern const int16_t x[];

constexpr size_t yLen = hLen + xLen - 1;
extern const int16_t yExpected[];
}
