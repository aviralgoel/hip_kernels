#include <concepts>

// C++20 ceiling division using concepts for type constraints
// Only accepts integral types (int, long, uint32_t, etc.)
// Example: ceilDiv(10, 3) returns 4
template<std::integral T1, std::integral T2>
inline T1 ceilDiv(T1 numerator, T2 denominator)
{
    return (numerator + denominator - 1) / denominator;
}