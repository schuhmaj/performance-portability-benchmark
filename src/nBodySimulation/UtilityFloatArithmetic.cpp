#include "UtilityFloatArithmetic.h"


namespace ppb::util {

    template<typename FloatType>
    __attribute__((device)) __attribute__((host)) bool almostEqualUlps(FloatType lhs, FloatType rhs, int ulpDistance) {
        static_assert(std::is_same<FloatType, float>::value || std::is_same<FloatType, double>::value,
                      "Template argument must be FloatType: Either float or double!");

        // In case the floats are equal in their representation, return true
        if (lhs == rhs) {
            return true;
        }

        // In case the signs mismatch, return false
        if (lhs < static_cast<FloatType>(0.0) && rhs > static_cast<FloatType>(0.0) ||
            lhs > static_cast<FloatType>(0.0) && rhs < static_cast<FloatType>(0.0)) {
            return false;
            }

        if constexpr (std::is_same<FloatType, float>::value) {
            // In case of float, compute ULP distance by interpreting float as 32-bit integer
            return reinterpret_cast<std::int32_t&>(rhs) - reinterpret_cast<std::int32_t&>(lhs) <= ulpDistance;
        }
        else if constexpr (std::is_same<FloatType, double>::value) {
            // In case of double, compute ULP distance by interpreting double as 64-bit integer
            return reinterpret_cast<std::int64_t&>(rhs) - reinterpret_cast<std::int64_t&>(lhs) <= ulpDistance;
        }

        // Due to the static_assert above, this should not happen
        return false;
    }

    // Template Instantations for float and double (required since definition is in .cpp file,
    //      also makes the static assert not strictly necessary)
    template bool __attribute__((device)) __attribute__((host)) almostEqualUlps<float>(float lhs, float rhs, int ulpDistance);
    template bool __attribute__((device)) __attribute__((host)) almostEqualUlps<double>(double lhs, double rhs, int ulpDistance);


    template<typename FloatType>
    __attribute__((device)) __attribute__((host)) bool almostEqualRelative(FloatType lhs, FloatType rhs, double epsilon) {
        const FloatType diff = std::abs(rhs - lhs);
        const FloatType largerValue = std::max(std::abs(rhs), std::abs(lhs));
        return diff <= largerValue * epsilon;
    }

    template bool __attribute__((device)) __attribute__((host)) almostEqualRelative<float>(float lhs, float rhs, double epsilon);
    template bool __attribute__((device)) __attribute__((host)) almostEqualRelative<double>(double lhs, double rhs, double epsilon);

}