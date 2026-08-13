#include <chrono>
#include <utility>
#include "Impl_SlangCuda.cuh"
#include "common/UtilityFloatArithmetic.h"
#include "matrixMultiplication/MatrixMultiplication.h"

namespace ppb {

    template<typename FloatType>
    ImplSlangCuda<FloatType>::ImplSlangCuda() {
        CHECK(cuInit(0));
        context.create();
        CHECK(cuCtxSetCurrent(context.ctx));
        CHECK(cuModuleLoad(&module.mod, SLANG_PTX_DIR "/MatrixMultiplicationShader.ptx"));
        CHECK(cuModuleGetFunction(&module.kernel, module.mod, "matmul_kernel"));
    }

    template <typename FloatType>
    std::pair<std::vector<FloatType>, double>
    ImplSlangCuda<FloatType>::operator()(const std::vector<FloatType> &a, const std::vector<FloatType> &b,
                                     const MatrixMultiplicationConfig &config) {
        CHECK(cuCtxSetCurrent(context.ctx));

        const size_t resultSize = config.m * config.n;
        std::vector<FloatType> result(resultSize, 0.0);

        DeviceMemory devA(sizeof(FloatType) * a.size());
        DeviceMemory devB(sizeof(FloatType) * b.size());
        DeviceMemory devC(sizeof(FloatType) * resultSize);

        CHECK(cuMemcpyHtoD(devA.ptr, a.data(), sizeof(FloatType) * a.size()));
        CHECK(cuMemcpyHtoD(devB.ptr, b.data(), sizeof(FloatType) * b.size()));
        CHECK(cuMemsetD8(devC.ptr, 0, sizeof(FloatType) * resultSize));

        struct PushConsts {
            uint32_t M;
            uint32_t N;
            uint32_t K;
        };

        PushConsts pc_host{
            static_cast<uint32_t>(config.m),
            static_cast<uint32_t>(config.n),
            static_cast<uint32_t>(config.k)
        };
        DeviceMemory devPc(sizeof(PushConsts));
        CHECK(cuMemcpyHtoD(devPc.ptr, &pc_host, sizeof(PushConsts)));

        struct PushParams {
            CUdeviceptr pc;
            ResourceSlot A;
            ResourceSlot B;
            ResourceSlot C;
        };

        PushParams params{};
        params.pc = devPc.ptr;
        params.A = {devA.ptr, 0};
        params.B = {devB.ptr, 0};
        params.C = {devC.ptr, 0};

        CUdeviceptr memory;
        size_t size;
        CHECK(cuModuleGetGlobal(&memory, &size, module.mod, "SLANG_globalParams"));
        CHECK(cuMemcpyHtoD(memory, &params, sizeof(PushParams)));

        constexpr unsigned int TILE_SIZE = 32;
        const unsigned int groups_x = util::ceilDiv<unsigned int>(config.m, TILE_SIZE);
        const unsigned int groups_y = util::ceilDiv<unsigned int>(config.n, TILE_SIZE);

        float elapsedTime;
        CUevent start, stop;
        CUstream stream;
        CHECK(cuEventCreate(&start, CU_EVENT_DEFAULT));
        CHECK(cuEventCreate(&stop, CU_EVENT_DEFAULT));
        CHECK(cuStreamCreate(&stream, 0));

        CHECK(cuEventRecord(start, stream));
        CHECK(cuLaunchKernel(module.kernel, groups_x, groups_y, 1, TILE_SIZE, TILE_SIZE, 1, 0, stream, nullptr, nullptr));
        CHECK(cuEventRecord(stop, stream));

        CHECK(cuEventSynchronize(stop));
        CHECK(cuEventElapsedTime(&elapsedTime, start, stop));
        double elapsed_nanoseconds = elapsedTime * 1e6;

        CHECK(cuMemcpyDtoH(result.data(), devC.ptr, sizeof(FloatType) * resultSize));

        CHECK(cuEventDestroy(start));
        CHECK(cuEventDestroy(stop));
        CHECK(cuStreamDestroy(stream));

        return std::make_pair(result, elapsed_nanoseconds);
    }

    template class ImplSlangCuda<float>;
    template class ImplSlangCuda<double>;
} // namespace ppb