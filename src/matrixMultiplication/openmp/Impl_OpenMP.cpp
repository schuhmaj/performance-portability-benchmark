#include "Impl_OpenMP.h"

namespace ppb {

    template <typename FloatType>
    std::vector<FloatType> ppb::ImplOpenMP<FloatType>::operator()(const std::vector<FloatType> &a,
                                                               const std::vector<FloatType> &b,  const MatrixMultiplicationConfig &config) {
        std::vector<FloatType> result(config.m * config.n, 0.0);
#pragma omp target parallel for collapse(2)
        for (int i = 0; i < config.m; ++i) {
            for (int j = 0; j < config.n; ++j) {
                for (int entry = 0; entry < config.k; ++entry) {
                    result[i + j * config.m] += a[i + entry * config.m] * b[entry + j * config.k];
                }
            }
        }
        return result;
    }

    /* Explicit Instantiation for float and double */
    template class ImplOpenMP<float>;
    template class ImplOpenMP<double>;
}