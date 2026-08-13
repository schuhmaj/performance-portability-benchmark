#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>
#include <type_traits>

namespace ppb::util {

    /**
     * This relative EPSILON is utilized ONLY for testing purposes to compare floating points.
     * It is used in the {@link ppb::util::almostEqualRelative} function.
     */
    constexpr double EPSILON_ALMOST_EQUAL = 1e-4;

    /**
     * The maximal allowed ULP distance utilized for FloatingPoint comparisons using the
     * {@link ppb::util::almostEqualUlps} function.
     *
     * @see https://randomascii.wordpress.com/2012/02/25/comparing-floating-point-numbers-2012-edition/
     */
    constexpr int MAX_ULP_DISTANCE = 4;


    /**
     * Function for comparing closeness of two floating point numbers using ULP (Units in the Last Place) method.
     *
     * @tparam FloatType must be either double or float (ensured by static assertion)
     * @param lhs The left hand side floating point number to compare.
     * @param rhs The right hand side floating point number to compare.
     * @param ulpDistance The maximum acceptable ULP distance between the two floating points
     *      for which they would be considered near each other. This is optional and by default, it will be {@link MAX_ULP_DISTANCE}.
     *
     * @return true if the ULP distance between lhs and rhs is less than or equal to the provided ulpDistance value, otherwise, false.
     *  Returns true if both numbers are exactly the same. Returns false if the signs do not match.
     * @note The ULP distance between 3.0 and std::nextafter(3.0, INFINITY) would be 1,
     *      the ULP distance of 3.0 and std::nextafter(std::nextafter(3.0, INFINITY), INFINITY) would be 2, etc.
     * @see https://randomascii.wordpress.com/2012/02/25/comparing-floating-point-numbers-2012-edition/
     */
    template<typename FloatType>
    bool almostEqualUlps(FloatType lhs, FloatType rhs, int ulpDistance = MAX_ULP_DISTANCE);

    /**
     * Function to check if two floating point numbers are relatively equal to each other within a given error range or tolerance.
     *
     * @tparam FloatType must be either double or float (ensured by static assertion)
     * @param lhs The first floating-point number to be compared.
     * @param rhs The second floating-point number to be compared.
     * @param epsilon The tolerance for comparison. Two numbers that are less than epsilon apart are considered equal.
     *                The default value is {@link EPSILON_ALMOST_EQUAL}.
     *
     * @return boolean value - Returns `true` if the absolute difference between `lhs` and `rhs` is less than or equal to
     *                         the relative error factored by the larger of the magnitude of `lhs` and `rhs`. Otherwise, `false`.
     * @see https://randomascii.wordpress.com/2012/02/25/comparing-floating-point-numbers-2012-edition/
     */
    template<typename FloatType>
    bool almostEqualRelative(FloatType lhs, FloatType rhs, double epsilon = EPSILON_ALMOST_EQUAL);


    /**
     * Computes the ceiling of the division of two numbers.
     * This function ensures that division rounds up to the nearest integer if there is any remainder.
     *
     * @tparam T The type of the input operands, typically an integral type.
     * @param a The dividend.
     * @param b The divisor. It is expected that `b` is non-zero.
     * @return The smallest integer value not less than `a / b`.
     *
     * @note This function is decorated with both __host__ and __device__ attributes, making it
     *       callable on both the host (CPU) and device (GPU) in CUDA programming.
     */
    template<typename T>
    __attribute__((host)) __attribute__((device)) inline T ceilDiv(const T &a, const T &b) {
        return (a + b - 1) / b;
    }


    /**
     * Rounds up a value to the nearest multiple of a specified group size.
     * This function ensures that the computed result is a multiple of `b`,
     * either equal to or greater than `a`.
     *
     * @tparam T The type of the input operands, typically an integral type.
     * @param a The value to be rounded up.
     * @param b The size of the group to which `a` will be rounded up. Must be non-zero.
     * @return The smallest multiple of `b` that is greater than or equal to `a`.
     *
     * @note If `a` is already a multiple of `b`, it remains unchanged.
     */
    template<typename T>
    constexpr inline T roundUp(const T &a, const T &b) {
        int r = a % b;
        return r == 0 ? a : a + b - r;
    }

    /**
     * Converts a floating-point type to its string representation at compile time.
     *
     * This function provides a compile-time mechanism to obtain a human-readable string
     * representation of floating-point types. It is particularly useful for logging,
     * debugging, or generating type-specific identifiers in template code.
     *
     * @tparam T The floating-point type to convert to a string. Should be float or double.
     *           Other types will return "unknown".
     *
     * @return A pointer to a constant character string representing the type name:
     *         - "float" if T is float
     *         - "double" if T is double
     *         - "unknown" for any other type
     */
    template <typename T>
    constexpr inline const char* to_string() {
        if constexpr (std::is_same_v<T, float>) {
            return "32";
        } else if constexpr (std::is_same_v<T, double>) {
            return "64";
        } else {
            return "unknown";
        }
    }
}
