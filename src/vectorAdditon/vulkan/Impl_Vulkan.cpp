#include <algorithm>
#include <benchmark/benchmark.h>
#include <chrono>
#include <iostream>
#include <memory>
#include "VectorAdditionShader.h"
#include "kompute/Kompute.hpp"
#include "vectorAdditon/VectorAddition.h"

namespace ppb {
    template <typename FloatType>
    struct ImplVulkan {
        using float_type = FloatType;

        std::pair<std::vector<FloatType>, double> operator()(const std::vector<FloatType> &a, const std::vector<FloatType> &b) {
            const unsigned int size = a.size();
            std::vector<FloatType> result(size, 0.0);

            kp::Manager mgr{};

            auto tensorA = mgr.tensor(a);
            auto tensorB = mgr.tensor(b);
            std::shared_ptr<kp::TensorT<float>> tensorC = mgr.tensor(result);

            std::vector<std::shared_ptr<kp::Tensor>> params = {tensorA, tensorB, tensorC};
            std::vector<uint32_t> shader(VECTORADDITIONSHADER_COMP_SPV.begin(), VECTORADDITIONSHADER_COMP_SPV.end());
            kp::Workgroup workgroup{{size, 1, 1}};

            auto algorithm = mgr.algorithm(
                params,
                shader,
                workgroup
            );
            auto sequence = mgr.sequence();
            sequence->template record<kp::OpTensorSyncDevice>(params)->eval();

            const auto start = std::chrono::high_resolution_clock::now();

            sequence->template record<kp::OpAlgoDispatch>(algorithm)->eval();

            const auto end = std::chrono::high_resolution_clock::now();
            const double elapsed_seconds = std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count();

            sequence->template record<kp::OpTensorSyncLocal>(params)->eval();
            result = tensorC->vector();
            return std::make_pair(result, elapsed_seconds);
        }
    };

    template class ImplVulkan<float>;
};

BENCHMARK(ppb::VectorAddition<ppb::ImplVulkan<float>>::benchmark)
    ->Name("VecAdd-CStd-Float")
    ->RangeMultiplier(10)
    ->Range(1e3, 1e8)
#ifdef PPB_MEASURE_ONLY_KERNEL
    ->UseManualTime()
#endif
    ->Complexity();

int main(int argc, char **argv) {
    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
}
