#include "Impl_Cpp.h"

namespace ppb {
    template <typename FloatType>
    std::vector<FloatType> ImplCpp<FloatType>::operator()(const std::vector<FloatType> &a,
                                                               const std::vector<FloatType> &b, const MatrixMultiplicationConfig &config) {
        std::vector<FloatType> result(config.m * config.n, 0.0);
        for (int i = 0; i < config.m; ++i) {
            for (int entry = 0; entry < config.k; ++entry) {
                for (int j = 0; j < config.n; ++j) {
                    result[i * config.n + j] += a[i * config.k + entry] * b[entry * config.n + j];
                }
            }
        }
        return result;
    }

    /* Explicit Instantiation for float and double */
    template class ImplCpp<float>;
    template class ImplCpp<double>;
}