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
    inline AlgorithmKinds from_string(const std::string &kind) {
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

    template <typename Algorithm>
    concept AlgorithmType = requires {
        { Algorithm::kind } -> std::convertible_to<AlgorithmKinds>;
    };

    static constexpr std::array<size_t, 2> NaiveDefaults = { 1024, 1024 };
    template<std::size_t local_size_x_float = NaiveDefaults[0], std::size_t local_size_x_double = NaiveDefaults[1]>
    struct Naive {
        static constexpr AlgorithmKinds kind = AlgorithmKinds::Naive;
        static constexpr size_t _local_size_x_float = local_size_x_float;
        static constexpr size_t _local_size_x_double = local_size_x_double;
    };

    static constexpr std::array<size_t, 4> CellListDefaults = { 1, 1, 4, 8 };
    template<std::size_t local_size_x_float = CellListDefaults[0], std::size_t local_size_x_double = CellListDefaults[1], std::size_t max_cell_density = CellListDefaults[2], std::size_t particles_per_cell = CellListDefaults[3]>
    struct CellList {
        static constexpr AlgorithmKinds kind = AlgorithmKinds::CellList;
        static constexpr size_t _local_size_x_float = local_size_x_float;
        static constexpr size_t _local_size_x_double = local_size_x_double;
        static constexpr size_t _max_cell_density = max_cell_density;
        static constexpr size_t _particles_per_cell = particles_per_cell;
    };
}