#pragma once

#include <cstdlib>
#include <limits>
#include <new>
#include <type_traits>
#include <vector>

namespace ppb {
    /**
     * Allocator to be used with the Containers of the standard library to be grounded on aligned memory.
     * @tparam T the type of the Allocator
     * @tparam Alignment the alignment of the reserved memory
     */
    template <typename T, std::size_t Alignment>
    class AlignedAllocator {

    public:
        /**Type Alias of the AlignedAllocator*/
        using value_type = T;
        using pointer = T *;
        using const_pointer = const T *;
        using reference = T &;
        using const_reference = const T &;
        using size_type = std::size_t;
        using difference_type = std::ptrdiff_t;

        template <typename U>
        struct rebind {
            using other = AlignedAllocator<U, Alignment>;
        };

        /** Default Constructor */
        AlignedAllocator() noexcept = default;

        /** Conversion Constructor */
        template <typename U>
        AlignedAllocator(const AlignedAllocator<U, Alignment> &) noexcept {}

        /**
         * Compares two AlignedAllocators and returns true if the types and alignments match.
         * @tparam OtherT the type of the other AlignedAllocator
         * @tparam OtherAlignment the alignment of the other AlignedAllocator
         * @param other the other Aligned Allocator
         * @return true if compile-time parameters match
         */
        template <typename OtherT, size_t OtherAlignment>
        constexpr bool operator==(const AlignedAllocator<OtherT, OtherAlignment> &other) const noexcept {
            return std::is_same_v<OtherT, T> && OtherAlignment == Alignment;
        }

        /**
         * Compares two AlignedAllocators and returns true if the types and alignments differ.
         * @tparam OtherT the type of the other AlignedAllocator
         * @tparam OtherAlignment the alignment of the other AlignedAllocator
         * @param other the other Aligned Allocator
         * @return true if compile-time parameters differ
         */
        template <typename OtherT, size_t OtherAlignment>
        constexpr bool operator!=(const AlignedAllocator<OtherT, OtherAlignment> &other) const noexcept {
            return !(*this == other);
        }

        /**
         * Allocates memory using the C++17's standard method `std::aligned_alloc`.
         * @param n the amount of elements T for which to allocate memory
         * @return pointer to the allocated memory
         * @throws std::bad_alloc if the allocation does not succeed.
         */
        T *allocate(const std::size_t n) {
            // Trivial Case, return nullptr
            if (n == 0) {
                return nullptr;
            }
            // If this is true, we would run into an overflow when we later require a multiplication n * sizeof(T)
            if (sizeof(T) * n > std::numeric_limits<size_t>::max() / sizeof(T)) {
                throw std::bad_alloc();
            }
            // The size of the aligned memory must be a multiple of the Alignment
            // This is a requirement by C++'s 17 standard std::aligned_alloc, even though the concrete implementation
            // might allow arbitrary size values, e.g., posix_memalign
            // see https://en.cppreference.com/w/cpp/memory/c/aligned_alloc
            size_t totalSize = n * sizeof(T);
            if (totalSize % Alignment != 0) {
                totalSize += Alignment - (totalSize % Alignment);
            }
            // Allocate and return the pointer
            if (auto p = std::aligned_alloc(Alignment, totalSize)) {
                return static_cast<T *>(p);
            }
            throw std::bad_alloc();
        }

        /**
         * Deallocates the memory pointed to by pointer p by calling `std::free`.
         * @param p pointer to the allocated memory
         */
        void deallocate(T *p, std::size_t) noexcept { std::free(p); }
    };

    /**
     * Vector with Aligned Memory. The memory is aligned according to the correspoding Alignement
     * @tparam T the data type of the vector
     * @tparam Alignment the memory alignment of the vector
     */
    template <typename T, std::size_t Alignment>
    using aligned_vector = std::vector<T, AlignedAllocator<T, Alignment>>;
}