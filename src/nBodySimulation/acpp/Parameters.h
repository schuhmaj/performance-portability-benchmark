/**
 * @file Parameters.h
 *
 * Manages the Parameters used to configure the device-compatible n-body simulation
 * Parameters include the supported Algorithms and their respective Parameters, as well as Parameters to fine-tune the performance of the simulation
 */

#pragma once

#include <concepts>
#include <cstddef>

namespace ppb {
    enum class AlgorithmKinds { Naive, CellList, Verlet };
    inline std::string to_string(const AlgorithmKinds kind) {
        switch (kind) {
            case AlgorithmKinds::Naive: return "Naive";
            case AlgorithmKinds::CellList: return "CellList";
            case AlgorithmKinds::Verlet: return "Verlet";
            default: return "Unknown";
        }
    }
    inline AlgorithmKinds algorithm_kinds_from_string(const std::string &kind) {
        if (kind == "Naive") {
            return AlgorithmKinds::Naive;
        }
        if (kind == "CellList") {
            return AlgorithmKinds::CellList;
        }
        if (kind == "Verlet") {
            return AlgorithmKinds::Verlet;
        }
        return AlgorithmKinds::Naive;
    }

    enum class SorterKinds { None, MergeProjection, MergeCellID/*, WarpProjection*/ };
    template<int n>
    struct select_sorter {
        static constexpr SorterKinds kind =
            n == 0 ? SorterKinds::None :
            n == 1 ? SorterKinds::MergeProjection :
            SorterKinds::MergeCellID;
            // SorterKinds::WarpProjection;
    };
    inline std::string to_string(const SorterKinds sorter) {
        switch (sorter) {
            case SorterKinds::None: return "None";
            case SorterKinds::MergeProjection: return "MergeProjection";
            case SorterKinds::MergeCellID: return "MergeCellID";
            // case SorterKinds::WarpProjection: return "WarpProjection";
            default: return "Unknown";
        }
    }
    inline SorterKinds sorter_kinds_from_string(const std::string &sorter) {
        if (sorter == "None") {
            return SorterKinds::None;
        }
        if (sorter == "MergeProjection") {
            return SorterKinds::MergeProjection;
        }
        if (sorter == "MergeCellID") {
            return SorterKinds::MergeCellID;
        }
        // if (sorter == "WarpProjection") {
        //     return SorterKinds::WarpProjection;
        // }
        return SorterKinds::None;
    }

    template <typename Algorithm>
    concept AlgorithmType = requires {
        { Algorithm::kind } -> std::convertible_to<AlgorithmKinds>;
    };


    struct NaiveConfig {
        static constexpr SorterKinds sorter = SorterKinds::MergeProjection;
        static constexpr size_t local_size_x_float = 1024;
        static constexpr size_t local_size_x_double = 1024;
    };

    template<
        SorterKinds sorter = NaiveConfig::sorter//,
        // std::size_t local_size_x_float = NaiveConfig::local_size_x_float,
        // std::size_t local_size_x_double = NaiveConfig::local_size_x_double
    >
    struct Naive {
        // local_size_x_float is a power of two
        // static_assert(local_size_x_float && ((local_size_x_float & (local_size_x_float - 1)) == 0));
        // local_size_x_double is a power of two
        // static_assert(local_size_x_double && ((local_size_x_double & (local_size_x_double - 1)) == 0));
        // local_size_x_float is less than the maximum work group size
        // static_assert(local_size_x_float <= 1024);
        // local_size_x_double is less than the maximum work group size
        // static_assert(local_size_x_double <= 1024);

        static constexpr AlgorithmKinds kind = AlgorithmKinds::Naive;
        static constexpr SorterKinds _sorter = sorter;
        static constexpr size_t _local_size_x_float = NaiveConfig::local_size_x_float;
        static constexpr size_t _local_size_x_double = NaiveConfig::local_size_x_double;
    };

    struct CellListConfig {
        static constexpr SorterKinds sorter = SorterKinds::MergeCellID;
        static constexpr size_t local_size_x_float = 1;
        static constexpr size_t local_size_x_double = 1;
        static constexpr size_t max_cell_density = 4;
        static constexpr size_t particles_per_cell = 32;
    };

    template<
        SorterKinds sorter = CellListConfig::sorter//,
        // std::size_t local_size_x_float = CellListConfig::local_size_x_float,
        // std::size_t local_size_x_double = CellListConfig::local_size_x_double,
        // std::size_t max_cell_density = CellListConfig::max_cell_density,
        // std::size_t particles_per_cell = CellListConfig::particles_per_cell
    >
    struct CellList {
        // local_size_x_float is a power of two
        // static_assert(local_size_x_float && ((local_size_x_float & (local_size_x_float - 1)) == 0));
        // local_size_x_double is a power of two
        // static_assert(local_size_x_double && ((local_size_x_double & (local_size_x_double - 1)) == 0));
        // local_size_x_float is less than the maximum work group size
        // static_assert(local_size_x_float <= 1024);
        // local_size_x_double is less than the maximum work group size
        // static_assert(local_size_x_double <= 1024);

        static constexpr AlgorithmKinds kind = AlgorithmKinds::CellList;
        static constexpr SorterKinds _sorter = sorter;
        static constexpr size_t _local_size_x_float = CellListConfig::local_size_x_float;
        static constexpr size_t _local_size_x_double = CellListConfig::local_size_x_double;
        static constexpr size_t _max_cell_density = CellListConfig::max_cell_density;
        static constexpr size_t _particles_per_cell = CellListConfig::particles_per_cell;
    };
}