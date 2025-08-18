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
        int k;
    };

    /**
     * @class MatrixMultiplication
     *
     * @brief Implements matrix multiplication functionality.
     *
     * Provides matrix multiplication for two input matrices, A and B, resulting
     * in an output matrix C. The sizes of the matrices are defined by the
     * constructor parameters m, n, and l.
     * A is M x K
     * B is K x N
     * C is M x N
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
        using is_row_major = typename Implementation::row_major;

    protected:
        /** Input matrix A in column-major format. */
        std::vector<FloatType> _inputA;
        std::vector<FloatType> _inputA_row_major;

        /** Input matrix B in column-major format. */
        std::vector<FloatType> _inputB;
        std::vector<FloatType> _inputB_row_major;

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
         * @param k Number of columns in matrix A and number of rows in matrix B.
         * @param seed Seed for the random number generator.
         */
        MatrixMultiplication(const int m, const int n, const int k, const unsigned int seed = 42u)
            : _inputA{ppb::generateUniformVector<std::vector<FloatType>>(m * k, seed)},
               _inputA_row_major{changeOrdering(_inputA, m)},
              _inputB{ppb::generateUniformVector<std::vector<FloatType>>(k * n, seed + 1)},
                _inputB_row_major{changeOrdering(_inputB, k)},
              _impl{},
              _config{MatrixMultiplicationConfig{m, n, k}} {
            //isFunctional();
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
            if constexpr (std::is_same_v<is_row_major, std::true_type>) {
                return _impl(_inputA_row_major, _inputB_row_major, _config);
            } else {
                return _impl(_inputA, _inputB, _config);
            }
        }

        /**
         * Quick check if the selected implementation is functional and can calculate the product of
         * array([[1, 2], [3, 4], [5, 6]]) * array([[7, 8, 9], [10, 11, 12]]) resulting in
         * array([[ 27,  30,  33], [ 61,  68,  75], [ 95, 106, 117]])
         * Matrices are stored in column-major format as specified in the documentation.
         */
        void isFunctional() {
            const std::vector<FloatType> matrixA = {1, 3, 5, 2, 4, 6};
            const std::vector<FloatType> matrixA_row_major = {1, 2, 3, 4, 5, 6};
            const std::vector<FloatType> matrixB = {7, 10, 8, 11, 9, 12};
            const std::vector<FloatType> matrixB_row_major = {7, 8, 9, 10, 11, 12};
            const std::vector<FloatType> expectedResult = {27, 61, 95, 30, 68, 106, 33, 75, 117};
            const std::vector<FloatType> expectedResult_row_major = {27, 30, 33, 61, 68, 75, 95, 106, 117};
            std::vector<FloatType> actualResult;
            if constexpr (std::is_same_v<is_row_major, std::true_type>) {
                actualResult = _impl(matrixA_row_major, matrixB_row_major, {3, 3, 2});
                if (!std::equal(actualResult.begin(), actualResult.end(), expectedResult_row_major.begin())) {
                    std::cerr << "Matrix multiplication failed!" << std::endl;
                    std::cerr << "Expected: " << expectedResult_row_major << std::endl;
                    std::cerr << "Actual: " << actualResult << std::endl;
                    std::exit(1);
                }
            } else {
                actualResult = _impl(matrixA, matrixB, {3, 3, 2});
                if (!std::equal(actualResult.begin(), actualResult.end(), expectedResult.begin())) {
                    std::cerr << "Matrix multiplication failed!" << std::endl;
                    std::cerr << "Expected: " << expectedResult << std::endl;
                    std::cerr << "Actual: " << actualResult << std::endl;
                    std::exit(1);
                }
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
