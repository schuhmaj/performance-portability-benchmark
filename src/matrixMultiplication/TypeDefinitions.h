#pragma once

#include <compare>
#include <concepts>
#include <cstddef>
#include <iterator>
#include <type_traits>

namespace ppb {
    /**
     * @brief Concept that defines requirements for a type T to be considered a Container.
     *
     * @tparam T The type to be checked.
     *
     * Requirements:
     * - std::begin(a) must return an input iterator.
     * - std::end(a) must return a sentinel that is compatible with the iterator from std::begin(a).
     * - std::size(a) must return the size of the container as std::size_t.
     */
    template <typename T>
    concept Container = requires(T a) {
        { std::begin(a) } -> std::input_iterator;
        { std::end(a) } -> std::sentinel_for<decltype(std::begin(a))>;
        { std::size(a) } -> std::same_as<std::size_t>;
    }
    && !std::convertible_to<T, std::string_view>
    && !std::is_array_v<std::remove_reference_t<T>>
    && !std::is_pointer_v<std::decay_t<T>>;

    /**
     * @brief Concept to ensure the type is comparable using the <=> operator.
     *
     * This concept checks that the type T supports the three-way comparison operator (<=>)
     * and that the result of this comparison can be converted to std::strong_ordering.
     * Types that satisfy this concept can be used in contexts where strong ordering is required,
     * such as sorting algorithms.
     *
     * @tparam T The type to check for comparability.
     */
    template <typename T>
    concept Comparable = requires(T a, T b) {
        { a <=> b } -> std::convertible_to<std::partial_ordering>;
    };

    /**
     * @brief Concept to determine if a type is a std::vector with any value_type and allocator_type.
     *
     * This concept checks whether a type `T` is a `std::vector` with arbitrary `value_type` and
     * `allocator_type`. It ensures that `T` has both `value_type` and `allocator_type` nested types
     * and that `T` is equivalent to `std::vector<typename T::value_type, typename T::allocator_type>`.
     *
     * @tparam T The type to check against the `std::vector` concept.
     *
     * @note This concept will fail for types that do not have `value_type` and `allocator_type`
     * members or for types that are not exactly `std::vector` with the specific type arguments.
     */
    template <typename T>
    concept Vector = requires {
        typename T::value_type;
        typename T::allocator_type;
    } && std::is_same_v<T, std::vector<typename T::value_type, typename T::allocator_type>>;

}