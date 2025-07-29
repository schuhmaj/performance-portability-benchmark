#include "Impl_OpenMPDevice.h"

namespace ppb {

    template <typename FloatType>
    std::vector<FloatType> ppb::ImplOpenMPDevice<FloatType>::operator()(const std::vector<FloatType> &a,
                                                               const std::vector<FloatType> &b) {
        const size_t sizeA = a.size();
        const size_t sizeB = b.size();
        const size_t sizeC = config.n * config.m;
        std::vector<FloatType> result(sizeC, 0.0);

        const FloatType *aPtr = a.data();
        const FloatType *bPtr = b.data();
        FloatType *resultPtr = result.data();

#pragma omp target parallel for collapse(2) map(to : aPtr[0 : sizeA], bPtr[0 : sizeB]) map(from : resultPtr[0 : sizeC])
        for (int i = 0; i < config.m; ++i) {
            for (int j = 0; j < config.n; ++j) {
                for (int entry = 0; entry < config.l; ++entry) {
                    resultPtr[i + j * config.m] += aPtr[i + entry * config.m] * bPtr[entry + j * config.l];
                }
            }
        }
        return result;
    }

    /* Explicit Instantiation for float and double */
    template class ImplOpenMPDevice<float>;
    template class ImplOpenMPDevice<double>;
}