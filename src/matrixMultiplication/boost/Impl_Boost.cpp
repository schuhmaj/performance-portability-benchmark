#include "Impl_Boost.h"
#include <chrono>
#include <utility>
#include "boost/compute.hpp"
#include "matrixMultiplication/MatrixMultiplication.h"
#include "common/UtilityFloatArithmetic.h"

namespace ppb {


    template <typename FloatType>
    ImplBoost<FloatType>::ImplBoost()
        : gpu{boost::compute::system::default_device()}
        , context{gpu}
        , queue{context, gpu, boost::compute::command_queue::enable_profiling}
        , program{boost::compute::program::build_with_source(std::string(KERNEL_SOURCE), context)}
        , kernel{program, std::string(kernel_name())}
    {}


    template <typename FloatType>
    std::pair<std::vector<FloatType>, double>
    ImplBoost<FloatType>::operator()(const std::vector<FloatType> &a, const std::vector<FloatType> &b,
                                     const MatrixMultiplicationConfig &config) {
        const size_t resultSize = config.m * config.n;
        std::vector<FloatType> result(resultSize, 0.0);
        boost::compute::vector<FloatType> deviceA{a.size(), context};
        boost::compute::vector<FloatType> deviceB{b.size(), context};
        boost::compute::vector<FloatType> resultBuffer(resultSize, context);
        boost::compute::copy(a.begin(), a.end(), deviceA.begin(), queue);
        boost::compute::copy(b.begin(), b.end(), deviceB.begin(), queue);

        kernel.set_arg(0, deviceA);
        kernel.set_arg(1, deviceB);
        kernel.set_arg(2, resultBuffer);
        kernel.set_arg(3, config.m);
        kernel.set_arg(4, config.n);
        kernel.set_arg(5, config.k);

        const size_t localSize[2] = {32, 32};
        const size_t globalSize[2] = {
            util::roundUp<size_t>(config.m, localSize[0]),
            util::roundUp<size_t>(config.n, localSize[1])
        };

        const auto event = queue.enqueue_nd_range_kernel(kernel, 2, nullptr, globalSize, localSize);
        event.wait();

        const double elapsed_nanoseconds = event.duration<boost::chrono::nanoseconds>().count();
        boost::compute::copy(resultBuffer.begin(), resultBuffer.end(), result.begin(), queue);
        queue.finish();
        return std::make_pair(std::move(result), elapsed_nanoseconds);
    }

    template class ImplBoost<float>;
    template class ImplBoost<double>;
} // namespace ppb