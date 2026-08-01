#include <chrono>
#include <utility>
#include "common/cuda/Common_Structs.cuh"
#include "common/UtilityFloatArithmetic.h"
#include "matrixMultiplication/MatrixMultiplication.h"
#include "vectorAdditon/VectorAddition.h"



namespace ppb {
    template <typename FloatType>
    struct ImplSlangCuda {
        using float_type = FloatType;

        CudaContext context;
        DeviceModule module;

        ImplSlangCuda() {
            CHECK(cuInit(0));
            context.create();
            CHECK(cuCtxSetCurrent(context.ctx));
            CHECK(cuModuleLoad(&module.mod, SLANG_PTX_DIR "/VectorAdditionShader.ptx"));
            CHECK(cuModuleGetFunction(&module.kernel, module.mod, "vecadd_kernel"));
        }

        std::pair<std::vector<FloatType>, double> operator()(const std::vector<FloatType> &a, const std::vector<FloatType> &b) {
            CHECK(cuCtxSetCurrent(context.ctx));
            const size_t size = a.size();

            DeviceMemory devA{sizeof(FloatType) * size};
            DeviceMemory devB{sizeof(FloatType) * size};
            DeviceMemory devC{sizeof(FloatType) * size};

            CHECK(cuMemcpyHtoD(devA.ptr, a.data(), sizeof(FloatType) * size));
            CHECK(cuMemcpyHtoD(devB.ptr, b.data(), sizeof(FloatType) * size));
            CHECK(cuMemsetD8(devC.ptr, 0, sizeof(FloatType) * size));

            struct PushParams {
                ResourceSlot A;
                ResourceSlot B;
                ResourceSlot C;
            };
            PushParams params{
                    {devA.ptr, size},
                    {devB.ptr, size},
                    {devC.ptr, size}
            };

            constexpr uint32_t local_size_x = 256;
            const uint32_t global_size_x = size / local_size_x + 1;
            constexpr uint32_t elements_per_vec4 = 4;
            uint32_t vec4_count = (size + elements_per_vec4 - 1) / elements_per_vec4;
            uint32_t num_workgroups = (vec4_count + local_size_x - 1) / local_size_x;

            CUdeviceptr memory;
            size_t s;
            CHECK(cuModuleGetGlobal(&memory, &s, module.mod, "SLANG_globalParams"));
            CHECK(cuMemcpyHtoD(memory, &params, sizeof(PushParams)));

            float elapsedTime;
            CUevent start, stop;
            CUstream stream;
            CHECK(cuEventCreate(&start, CU_EVENT_DEFAULT));
            CHECK(cuEventCreate(&stop, CU_EVENT_DEFAULT));
            CHECK(cuStreamCreate(&stream, 0));


            CHECK(cuEventRecord(start, stream));
            CHECK(cuLaunchKernel(module.kernel, num_workgroups, 1, 1, local_size_x, 1, 1, 0, stream, nullptr, nullptr));
            CHECK(cuEventRecord(stop, stream));

            CHECK(cuEventSynchronize(stop));
            CHECK(cuEventElapsedTime(&elapsedTime, start, stop));
            double elapsed_nanoseconds = elapsedTime * 1e6;

            std::vector<FloatType> result(size);
            CHECK(cuMemcpyDtoH(result.data(), devC.ptr, sizeof(FloatType) * size));
            return {result, 0};
        }
    };

    template class ImplSlangCuda<float>;
    template class ImplSlangCuda<double>;
};

BENCHMARK(ppb::VectorAddition<ppb::ImplSlangCuda<ppb::VectorAdditionBenchmarkConf::float_type>>::benchmark)
    ->Name("VecAdd")
    ->RangeMultiplier(10)
    ->Range(ppb::VectorAdditionBenchmarkConf::MIN_SIZE, ppb::VectorAdditionBenchmarkConf::MAX_SIZE)
#ifdef PPB_MEASURE_ONLY_KERNEL
    ->UseManualTime()
#endif
    ->Complexity();

int main(int argc, char **argv) {
    ppb::VectorAdditionBenchmarkConf::addContext("Slang-Cuda");
    benchmark::MaybeReenterWithoutASLR(argc, argv);
    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
}