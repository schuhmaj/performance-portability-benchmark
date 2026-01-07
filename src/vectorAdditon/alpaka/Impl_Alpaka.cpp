#include <benchmark/benchmark.h>
#include <iostream>
#include <utility>
#include "alpaka/alpaka.hpp"
#include "alpaka/example/ExecuteForEachAccTag.hpp"
#include "alpaka/example/ExampleDefaultAcc.hpp"
#include "vectorAdditon/VectorAddition.h"

namespace ppb {


    struct AlpakaImpl {
        /** Float Type of Implementation, Required by the Benchmark **/
        using float_type = float;

        /** Dimensionality of the Problem, for Vector Addition it's 1D */
        using Dim = alpaka::DimInt<1u>;
        /** The Integer Type used for indexing and sizes **/
        using Idx = std::size_t;
        /** The Host Backend, Serial CPU **/
        using Host = alpaka::DevCpu;
        /** Defines the Compute Backend/ Device to use; We chose the first one which is enabled; CUDA/ HIP devices have precedence in this "ExampleDefault" List **/
        using Acc = alpaka::ExampleDefaultAcc<Dim, Idx>;
        /** Defines the Runtime of the chosen Accelerator, i.e. CUDA, the software layer **/
        using Platform = alpaka::Platform<Acc>;
        /** Defines the actual physical device, i.e. RTX 2080 **/
        using Device = alpaka::Dev<Platform>;
        /** The Compute Pipline for the Accelerator Device **/
        using Queue = alpaka::Queue<Device, alpaka::Blocking>;

        /** Alias for Buffer on the CPU/ Host **/
        using BufHost = alpaka::Buf<Host, float_type, Dim, Idx>;
        /** Alias for the Buffer on the Device **/
        using BufAcc = alpaka::Buf<Device, float_type, Dim, Idx>;

        /** The CPU/ Host **/
        Host host;

        /** The actual device chosen by ID, using the runtime, i.e. CUDA_DEVICE 0, 1, ... **/
        Device device;
        /** The compute pipeline for the chosen device **/
        Queue queue;


        AlpakaImpl()
            : host(alpaka::getDevByIdx(alpaka::PlatformCpu{}, 0))
            , device(alpaka::getDevByIdx(alpaka::Platform<Acc>{}, 0))
            , queue(device) {}

        ~AlpakaImpl() = default;

        std::pair<std::vector<float_type>, double> operator()(const std::vector<float_type> &a, const std::vector<float_type> &b) {
            const Idx size = a.size();
            const alpaka::Vec<Dim, Idx> extent(size);

            // Create host views to data (or alternativley one could allocate host buffers)
            const auto bufHostA = alpaka::createView(host, const_cast<float_type*>(a.data()), extent);
            const auto bufHostB = alpaka::createView(host, const_cast<float_type*>(b.data()), extent);

            // Allocate device buffers
            auto bufDevA = alpaka::allocBuf<float_type, Idx>(device, extent);
            auto bufDevB = alpaka::allocBuf<float_type, Idx>(device, extent);
            auto bufDevC = alpaka::allocBuf<float_type, Idx>(device, extent);

            // Copy data from host to device
            alpaka::memcpy(queue, bufDevA, bufHostA, extent);
            alpaka::memcpy(queue, bufDevB, bufHostB, extent);

            // Create Kernel
            const auto vectorAddKernel = [=] ALPAKA_FN_ACC (const Acc& acc, const float_type* const A, const float_type* const B, float_type* const
                                                            C, const Idx size) {
                const auto globalThreadIdx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0u];
                if (globalThreadIdx < size) {
                    C[globalThreadIdx] = A[globalThreadIdx] + B[globalThreadIdx];
                }
            };

            // Calculate optimal block and grid size for the given problem extent
            // Also incoperate how many elements of the input shall be processed per Thread
            const auto workDiv = alpaka::getValidWorkDiv(
                alpaka::KernelCfg<Acc>{extent, 1},
                device,
                vectorAddKernel,
                alpaka::getPtrNative(bufDevA),
                alpaka::getPtrNative(bufDevB),
                alpaka::getPtrNative(bufDevC),
                size
            );
            auto const taskKernel = alpaka::createTaskKernel<Acc>(
                workDiv,
                vectorAddKernel,
                alpaka::getPtrNative(bufDevA),
                alpaka::getPtrNative(bufDevB),
                alpaka::getPtrNative(bufDevC),
                size
            );

            // Kernel Execution
            const auto start = std::chrono::high_resolution_clock::now();
            alpaka::enqueue(queue, taskKernel);
            alpaka::wait(queue);
            const auto end = std::chrono::high_resolution_clock::now();
            const double elapsed_nanoseconds = static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count());

            // Create result vector
            std::vector<float_type> result(size);
            auto resultView = alpaka::createView(host, result.data(), extent);
            alpaka::memcpy(queue, resultView, bufDevC, extent);
            alpaka::wait(queue);

            return std::make_pair(std::move(result), elapsed_nanoseconds);
        }
    };
};

BENCHMARK(ppb::VectorAddition<ppb::AlpakaImpl>::benchmark)
    ->Name("VecAdd-Float-Alpaka")
    ->RangeMultiplier(10)
    ->Range(ppb::VectorAdditionBenchmarkConf::MIN_SIZE, ppb::VectorAdditionBenchmarkConf::MAX_SIZE)
#ifdef PPB_MEASURE_ONLY_KERNEL
    ->UseManualTime()
#endif
    ->Complexity();



int main(int argc, char **argv) {
    benchmark::MaybeReenterWithoutASLR(argc, argv);

    std::cout << "Alpaka Enabled Accelerator Tags:" << std::endl;
    alpaka::printTagNames<alpaka::EnabledAccTags>();

    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
}
