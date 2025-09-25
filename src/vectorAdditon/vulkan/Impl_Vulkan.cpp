#include <algorithm>
#include <benchmark/benchmark.h>
#include <chrono>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <vector>
#include <vulkan/vulkan.hpp>
#include "VectorAdditionShader.h"
#include "kompute/Kompute.hpp"
#include "vectorAdditon/VectorAddition.h"
#include "common/vulkan/VulkanUtility.h"

namespace ppb {
    template <typename FloatType>
    struct ImplVulkan {
        using float_type = FloatType;

        std::pair<std::vector<FloatType>, double> operator()(const std::vector<FloatType> &a, const std::vector<FloatType> &b) {
            const unsigned int size = a.size();
            std::vector<FloatType> result(size, 0.0);
            try {
                vulkan_utility::VulkanManager manager{};

                auto tensorA = manager.tensor(a);
                auto tensorB = manager.tensor(b);
                auto tensorC = manager.tensor(result);

                std::vector<std::shared_ptr<kp::Tensor>> params = {tensorA, tensorB, tensorC};
                std::vector<uint32_t> shader(VECTORADDITIONSHADER_COMP_SPV.begin(), VECTORADDITIONSHADER_COMP_SPV.end());
                kp::Workgroup workgroup{{size, 1, 1}};

                auto algorithm = manager.algorithm(params, shader, workgroup);
                auto sequence = manager.sequence();

                sequence->template record<kp::OpTensorSyncDevice>(params)->eval();

                const auto start = std::chrono::high_resolution_clock::now();

                sequence->template record<kp::OpAlgoDispatch>(algorithm)->eval();

                const auto end = std::chrono::high_resolution_clock::now();
                const double elapsed_seconds = std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count();

                sequence->template record<kp::OpTensorSyncLocal>(params)->eval();
                result = tensorC->vector();
                return std::make_pair(result, elapsed_seconds);
            } catch (const std::exception &ex) {
                std::cerr << "Vulkan/Kompute initialization or execution failed: " << ex.what() << std::endl;
                return std::make_pair(result, 0.0);
            }
        }
    };

    template class ImplVulkan<float>;
};

BENCHMARK(ppb::VectorAddition<ppb::ImplVulkan<float>>::benchmark)
    ->Name("VecAdd-Vulkan-Float")
    ->RangeMultiplier(10)
    ->Range(1e3, 1e8)
#ifdef PPB_MEASURE_ONLY_KERNEL
    ->UseManualTime()
#endif
    ->Complexity();

int main(int argc, char **argv) {
    benchmark::MaybeReenterWithoutASLR(argc, argv);
    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
}
