#include "Impl_OpenMP.h"

namespace ppb {

    template <typename FloatType>
    std::pair<std::vector<FloatType>, double> ppb::ImplOpenMP<FloatType>::operator()(const std::vector<FloatType> &a,
                                                               const std::vector<FloatType> &b,  const MatrixMultiplicationConfig &config) {
        std::vector<FloatType> result(config.m * config.n, 0.0);
        const auto start = std::chrono::high_resolution_clock::now();
#pragma omp parallel for
        for (int j = 0; j < config.n; ++j) {
            for (int entry = 0; entry < config.k; ++entry) {
                for (int i = 0; i < config.m; ++i) {
                    result[i + j * config.m] += a[i + entry * config.m] * b[entry + j * config.k];
                }
            }
        }
        const auto end = std::chrono::high_resolution_clock::now();
        const double elapsed_seconds = std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count();
        return std::make_pair(result, elapsed_seconds);
    }

    /* Explicit Instantiation for float and double */
    template class ImplOpenMP<float>;
    template class ImplOpenMP<double>;
}