#include "Impl_Stdpar.h"

#include <algorithm>
#include <execution>

namespace ppb {

    template <typename FloatType>
    std::pair<std::vector<FloatType>, double> ImplStdpar<FloatType>::operator()(const std::vector<FloatType> &a,
                                                               const std::vector<FloatType> &b, const MatrixMultiplicationConfig &config) {
        const int m = config.m;
        const int n = config.n;
        const int k = config.k;
        std::vector<FloatType> result(static_cast<size_t>(m) * n, 0.0);

        const FloatType *aPtr = a.data();
        const FloatType *bPtr = b.data();
        FloatType *resultPtr = result.data();

        // The parallel loop runs over the output buffer itself and recovers the flat index from the
        // pointer distance. A counting range (std::views::iota) would be the more obvious choice, but
        // its iterators are not usable with every stdpar backend: oneDPL rejects them (they are neither
        // USM pointers nor writable, and iota_view<size_t>::difference_type is __int128, which SPIR-V
        // targets cannot represent), so keep to plain pointers, which all backends map to device memory.
        const auto start = std::chrono::high_resolution_clock::now();
        std::for_each(std::execution::par_unseq, resultPtr, resultPtr + result.size(), [=](FloatType &element) {
            const size_t index = static_cast<size_t>(&element - resultPtr);
            const int i = static_cast<int>(index % m);
            const int j = static_cast<int>(index / m);
            FloatType sum = 0.0;
            for (int entry = 0; entry < k; ++entry) {
                sum += aPtr[i + entry * m] * bPtr[entry + j * k];
            }
            element = sum;
        });
        const auto end = std::chrono::high_resolution_clock::now();
        const double elapsed_nanoseconds = static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count());
        return std::make_pair(result, elapsed_nanoseconds);
    }

    /* Explicit Instantiation for float and double */
    template class ImplStdpar<float>;
    template class ImplStdpar<double>;
}
