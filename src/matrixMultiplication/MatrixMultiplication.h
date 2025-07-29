#pragma once

#include "benchmark/benchmark.h"

#include <algorithm>
#include <chrono>
#include <iostream>
#include <random>
#include <vector>

#include "ContainerUtility.h"

namespace ppb {


    struct MatrixMultiplicationConfig {
        int m;
        int n;
        int l;
    };

    /**
     * @class MatrixMultiplication
     *
     * @brief Implements matrix multiplication functionality.
     *
     * Provides matrix multiplication for two input matrices, A and B, resulting
     * in an output matrix C. The sizes of the matrices are defined by the
     * constructor parameters m, n, and k.
     *
     * The data is stored in a column-major format.
     *
     */
    template <class Implementation>
    class MatrixMultiplication final {

    public:
        /**
         * The floating point type used by the simulation (extracted from the implementation).
         */
        using FloatType = typename Implementation::float_type;

    protected:
        /** Input matrix A in column-major format. */
        std::vector<FloatType> _inputA;

        /** Input matrix B in column-major format. */
        std::vector<FloatType> _inputB;

        /**
         * The simulation implementation instance.
         */
        Implementation _impl;

        MatrixMultiplicationConfig _config;

    public:
        /**
         * Constructs a square matrix multiplication object with given size and seed.
         * @param size The number of rows and columns (size) for matrices A, B, and C.
         * @param seed Seed for the random number generator. Defaults to 42.
         */
        explicit MatrixMultiplication(const int size, const unsigned int seed = 42u)
            : MatrixMultiplication{size, size, size, seed} {}

        /**
         * Constructor to initialize matrix dimensions and seed for random number generation.
         * @param m Number of rows in matrix A and matrix C.
         * @param n Number of columns in matrix B and matrix C.
         * @param l Number of columns in matrix A and number of rows in matrix B.
         * @param seed Seed for the random number generator.
         */
        MatrixMultiplication(const int m, const int n, const int l, const unsigned int seed = 42u)
            : _inputA{ppb::generateUniformVector<std::vector<FloatType>>(m * l, seed)},
              _inputB{ppb::generateUniformVector<std::vector<FloatType>>(l * n, seed + 1)},
              _impl{},
              _config{MatrixMultiplicationConfig{m, n, l}} {
            isFunctional();
        }

        /**
         * Default destructor.
         */
        ~MatrixMultiplication() = default;

        /**
         * Multiplies the matrices _inputA and _inputB, and stores the result in _outputC.
         * The computation follows the standard matrix multiplication algorithm.
         *
         * @return The resulting matrix C.
         */
        std::vector<FloatType> operator()() {
            return _impl(_inputA, _inputB, _config);
        }

        void isFunctional() {
            const std::vector<FloatType> matrixA = {1, 3, 2, 4};
            const std::vector<FloatType> matrixB = {5, 7, 6, 8};
            const std::vector<FloatType> expectedResult = {19, 43, 22, 50};
            const auto actualResult = _impl(matrixA, matrixB, {2, 2, 2});
            if (!std::equal(actualResult.begin(), actualResult.end(), expectedResult.begin())) {
                std::cerr << "Matrix multiplication failed!" << std::endl;
                std::cerr << "Expected: " << expectedResult << std::endl;
                std::cerr << "Actual: " << actualResult << std::endl;
                std::exit(1);
            }
        }

        /**
         * Method suitable for Google Benchmark framework to measure performance.
         * @param state Benchmark state.
         */
        static void inline benchmark(benchmark::State& state) {
            const size_t size = state.range(0);
            MatrixMultiplication matrixMultiplication{static_cast<int>(size)};
            for (auto _ : state) {
                const auto start = std::chrono::high_resolution_clock::now();

                auto result = matrixMultiplication();
                benchmark::DoNotOptimize(result);

                const auto end = std::chrono::high_resolution_clock::now();
                const auto elapsed_seconds = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
                state.SetIterationTime(elapsed_seconds.count());
            }
            state.SetComplexityN(static_cast<long long>(size));
        }

    };
}
