#include "Impl_Cpp.h"

namespace ppb {
    template <typename FloatType>
    std::pair<std::vector<FloatType>, double> ImplCpp<FloatType>::operator()(const std::vector<FloatType> &a,
                                                               const std::vector<FloatType> &b, const MatrixMultiplicationConfig &config) {
        std::vector<FloatType> result(config.m * config.n, 0.0);
        constexpr int TILE_SIZE = 64;
        const auto start = std::chrono::high_resolution_clock::now();
        if constexpr (row_major::value) {
            for (int tile = 0; tile < config.k; tile += TILE_SIZE) {
                const int endK = std::min(tile + TILE_SIZE, config.k);
                for (int i = 0; i < config.m; ++i) {
                    for (int entry = tile; entry < endK; ++entry) {
                        for (int j = 0; j < config.n; ++j) {
                            result[i * config.n + j] += a[i * config.k + entry] * b[entry * config.n + j];
                        }
                    }
                }
            }
        } else {
            for (int tile = 0; tile < config.k; tile += TILE_SIZE) {
                const int endK = std::min(tile + TILE_SIZE, config.k);
                for (int j = 0; j < config.n; ++j) {
                    for (int entry = tile; entry < endK; ++entry) {
                        for (int i = 0; i < config.m; ++i) {
                            result[i + j * config.m] += a[i + entry * config.m] * b[entry + j * config.k];
                        }
                    }
                }
            }
        }
        const auto end = std::chrono::high_resolution_clock::now();
        const double elapsed_nanoseconds = static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count());
        return std::make_pair(result, elapsed_nanoseconds);
    }

    /* Explicit Instantiation for float and double */
    template class ImplCpp<float>;
    template class ImplCpp<double>;
}