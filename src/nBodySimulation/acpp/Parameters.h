/**
 * @file Parameters.h
 *
 * Manages the Parameters used to configure the device-compatible n-body simulation
 * Parameters include the supported Algorithms and their respective Parameters, as well as Parameters to fine-tune the performance of the simulation
 */

#pragma once

#include <concepts>

static constexpr size_t BLOCK_SIZE = 32; // block size in bytes : BLOCK_SIZE * sizeof(sycl::vec<FloatType, 4>) = 512 * sizeof(FloatType)
static size_t pad(const size_t value) { return ((value - 1) / BLOCK_SIZE + 1) * BLOCK_SIZE; }

static constexpr size_t sorter_frequency = 1000;

namespace ppb {
    enum class AlgorithmKinds { Naive, CellList };
    inline std::string to_string(const AlgorithmKinds kind) {
        switch (kind) {
            case AlgorithmKinds::Naive: return "Naive";
            case AlgorithmKinds::CellList: return "CellList";
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
        return AlgorithmKinds::Naive;
    }

    enum class SorterKinds { None, MergeProjection, MergeCellID };
    template<int n>
    struct select_sorter {
        static constexpr SorterKinds kind =
            n == 0 ? SorterKinds::None :
            n == 1 ? SorterKinds::MergeProjection :
            SorterKinds::MergeCellID;
    };
    inline std::string to_string(const SorterKinds sorter) {
        switch (sorter) {
            case SorterKinds::None: return "None";
            case SorterKinds::MergeProjection: return "MergeProjection";
            case SorterKinds::MergeCellID: return "MergeCellID";
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
        return SorterKinds::None;
    }

    template <typename Algorithm>
    concept AlgorithmType = requires {
        { Algorithm::kind } -> std::convertible_to<AlgorithmKinds>;
    };


    struct NaiveConfig {
        static constexpr SorterKinds sorter = SorterKinds::MergeProjection;
        static constexpr double cutoff = 3.0;
    };

    template<
        SorterKinds sorter = NaiveConfig::sorter,
        double cutoff = NaiveConfig::cutoff
    >
    struct Naive {
        static constexpr AlgorithmKinds kind = AlgorithmKinds::Naive;
        static constexpr SorterKinds _sorter = sorter;
        static constexpr double _cutoff = cutoff;
    };

    struct CellListConfig {
        static constexpr SorterKinds sorter = SorterKinds::MergeCellID;
        static constexpr double cutoff = 3.0;
        // more particles per cell means more interactions and therefore more accuracy at the cost of runtime
        static constexpr size_t particles_per_cell = 32;
    };

    template<
        SorterKinds sorter = CellListConfig::sorter,
        double cutoff = CellListConfig::cutoff,
        size_t particles_per_cell = CellListConfig::particles_per_cell
    >
    struct CellList {
        static constexpr AlgorithmKinds kind = AlgorithmKinds::CellList;
        static constexpr SorterKinds _sorter = sorter;
        static constexpr double _cutoff = cutoff;
        static constexpr size_t _particles_per_cell = particles_per_cell;
    };
}