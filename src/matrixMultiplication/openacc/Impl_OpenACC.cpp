#include "Impl_OpenACC.h"

namespace ppb {

    template <typename FloatType>
    std::pair<std::vector<FloatType>, double> ppb::ImplOpenACC<FloatType>::operator()(const std::vector<FloatType> &a,
                                                               const std::vector<FloatType> &b, const MatrixMultiplicationConfig &config) {
        const size_t sizeA = a.size();
        const size_t sizeB = b.size();
        const size_t sizeC = config.m * config.n;
        std::vector<FloatType> result(sizeC, 0.0);

        const FloatType *aPtr = a.data();
        const FloatType *bPtr = b.data();
        FloatType *resultPtr = result.data();

        const auto start = std::chrono::high_resolution_clock::now();
#pragma acc data copyin(aPtr[0:sizeA], bPtr[0:sizeB]) copy(resultPtr[0:sizeC])
        {
#pragma acc parallel loop collapse(2) present(aPtr, bPtr, resultPtr)
            for (int j = 0; j < config.n; ++j) {
                for (int i = 0; i < config.m; ++i) {
                    FloatType sum = 0.0;
#pragma acc loop seq
                    for (int entry = 0; entry < config.k; ++entry) {
                        sum += aPtr[i + entry * config.m] * bPtr[entry + j * config.k];
                    }
                    resultPtr[i + j * config.m] = sum;
                }
            }
        }

        const auto end = std::chrono::high_resolution_clock::now();
        const double elapsed_nanoseconds = static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count());
        return std::make_pair(result, elapsed_nanoseconds);
    }

    template class ImplOpenACC<float>;
    template class ImplOpenACC<double>;
}