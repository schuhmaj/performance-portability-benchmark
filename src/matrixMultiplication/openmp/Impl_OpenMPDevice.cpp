#include "Impl_OpenMPDevice.h"

namespace ppb {

    template <typename FloatType>
    std::pair<std::vector<FloatType>, double> ppb::ImplOpenMPDevice<FloatType>::operator()(const std::vector<FloatType> &a,
                                                               const std::vector<FloatType> &b, const MatrixMultiplicationConfig &config) {
        const size_t sizeA = a.size();
        const size_t sizeB = b.size();
        const size_t sizeC = config.m * config.n;
        std::vector<FloatType> result(sizeC, 0.0);

        const FloatType *aPtr = a.data();
        const FloatType *bPtr = b.data();
        FloatType *resultPtr = result.data();

        const auto start = std::chrono::high_resolution_clock::now();
#pragma omp target teams distribute parallel for collapse(2) map(to : aPtr[0 : sizeA], bPtr[0 : sizeB]) map(tofrom : resultPtr[0 : sizeC])
        for (int j = 0; j < config.n; ++j) {
            for (int i = 0; i < config.m; ++i) {
                FloatType sum = 0.0;
                for (int entry = 0; entry < config.k; ++entry) {
                    sum += aPtr[i + entry * config.m] * bPtr[entry + j * config.k];
                }
                resultPtr[i + j * config.m] = sum;
            }
        }

        const auto end = std::chrono::high_resolution_clock::now();
        const double elapsed_seconds = std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count();
        return std::make_pair(result, elapsed_seconds);
    }

    /* Explicit Instantiation for float and double */
    template class ImplOpenMPDevice<float>;
    template class ImplOpenMPDevice<double>;
}