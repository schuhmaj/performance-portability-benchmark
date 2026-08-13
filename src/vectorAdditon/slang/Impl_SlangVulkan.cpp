#include <chrono>
#include <utility>
#include "common/vulkan/VulkanUtility.h"
#include "matrixMultiplication/MatrixMultiplication.h"
#include "VectorAdditionShader.h"
#include "vectorAdditon/VectorAddition.h"


namespace ppb {
    template <typename FloatType>
    struct ImplSlangVulkan {
        using float_type = FloatType;

        vulkan_utility::VulkanManager manager;
        std::vector<uint32_t> shader;
        std::shared_ptr<kp::Sequence> sequence;

        ImplSlangVulkan() : manager{}, shader{std::begin(vecadd_kernel), std::end(vecadd_kernel)}, sequence{manager.sequence()}{}

        std::pair<std::vector<FloatType>, double> operator()(const std::vector<FloatType> &a, const std::vector<FloatType> &b) {
            const unsigned int size = a.size();
            std::vector<FloatType> result(size, 0.0);
            try {
                auto tensorA = manager.tensor(a);
                auto tensorB = manager.tensor(b);
                auto tensorC = manager.tensor(result);

                std::vector<std::shared_ptr<kp::Tensor>> params = {tensorA, tensorB, tensorC};
                constexpr uint32_t local_size_x = 256;
                constexpr uint32_t elements_per_vec4 = 4;
                uint32_t vec4_count = (size + elements_per_vec4 - 1) / elements_per_vec4;
                uint32_t num_workgroups = (vec4_count + local_size_x - 1) / local_size_x;
                kp::Workgroup workgroup{{num_workgroups, 1, 1}};

                auto algorithm = manager.algorithm(params, shader, workgroup);

                sequence->template record<kp::OpTensorSyncDevice>(params)->eval();

                const auto start = std::chrono::high_resolution_clock::now();

                sequence->template record<kp::OpAlgoDispatch>(algorithm)->eval();

                const auto end = std::chrono::high_resolution_clock::now();
                const double elapsed_nanoseconds =
                    static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count());

                sequence->template record<kp::OpTensorSyncLocal>(params)->eval();
                result = tensorC->vector();
                return std::make_pair(result, elapsed_nanoseconds);
            } catch (const std::exception &ex) {
                std::cerr << "Vulkan/Kompute initialization or execution failed: " << ex.what() << std::endl;
                return std::make_pair(result, 0.0);
            }
        }
    };

    template class ImplSlangVulkan<float>;
    template class ImplSlangVulkan<double>;
};

BENCHMARK(ppb::VectorAddition<ppb::ImplSlangVulkan<ppb::VectorAdditionBenchmarkConf::float_type>>::benchmark)
    ->Name("VecAdd")
    ->RangeMultiplier(10)
    ->Range(ppb::VectorAdditionBenchmarkConf::MIN_SIZE, ppb::VectorAdditionBenchmarkConf::MAX_SIZE)
#ifdef PPB_MEASURE_ONLY_KERNEL
    ->UseManualTime()
#endif
    ->Complexity();

int main(int argc, char **argv) {
    ppb::VectorAdditionBenchmarkConf::addContext("Slang-Vulkan");
    benchmark::MaybeReenterWithoutASLR(argc, argv);
    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
}
