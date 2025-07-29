#include "Impl_Cpp.h"
#include "omp.h"

namespace ppb {
    template <typename FloatType>
    std::vector<FloatType> ImplCpp<FloatType>::operator()(const std::vector<FloatType> &a,
                                                               const std::vector<FloatType> &b, const MatrixMultiplicationConfig &config) {
        std::vector<FloatType> result(config.n * config.m, 0.0);
        for (int i = 0; i < config.m; ++i) {
            for (int j = 0; j < config.n; ++j) {
                for (int entry = 0; entry < config.l; ++entry) {
                    result[i + j * config.m] += a[i + entry * config.m] * b[entry + j * config.l];
                }
            }
        }
        return result;
    }

    /* Explicit Instantiation for float and double */
    template class ImplCpp<float>;
    template class ImplCpp<double>;
}