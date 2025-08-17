#include "Impl_Cpp.h"
#include "omp.h"

namespace ppb {
    template <typename FloatType>
    std::vector<FloatType> ImplCpp<FloatType>::operator()(const std::vector<FloatType> &a,
                                                               const std::vector<FloatType> &b, const MatrixMultiplicationConfig &config) {
        std::vector<FloatType> result(config.m * config.n, 0.0);
        constexpr int TILE_SIZE = 64;
#pragma omp parallel for
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
        return result;
    }

    /* Explicit Instantiation for float and double */
    template class ImplCpp<float>;
    template class ImplCpp<double>;
}