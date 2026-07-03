#include "Impl_Stdpar.h"

#include <algorithm>
#include <execution>
#include <ranges>

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

        const auto indices = std::views::iota(size_t{0}, result.size());

        const auto start = std::chrono::high_resolution_clock::now();
        std::for_each(std::execution::par_unseq, indices.begin(), indices.end(), [=](const size_t index) {
            const int i = static_cast<int>(index % m);
            const int j = static_cast<int>(index / m);
            FloatType sum = 0.0;
            for (int entry = 0; entry < k; ++entry) {
                sum += aPtr[i + entry * m] * bPtr[entry + j * k];
            }
            resultPtr[index] = sum;
        });
        const auto end = std::chrono::high_resolution_clock::now();
        const double elapsed_nanoseconds = static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count());
        return std::make_pair(result, elapsed_nanoseconds);
    }

    /* Explicit Instantiation for float and double */
    template class ImplStdpar<float>;
    template class ImplStdpar<double>;
}
