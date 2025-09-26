#include <Metal/Metal.hpp>
#include <benchmark/benchmark.h>
#include <chrono>
#include <iostream>
#include <utility>
#include <vector>
#include "vectorAdditon/VectorAddition.h"

namespace ppb {

    /**
     * Metal backend implementation for the VectorAddition benchmark.
     * while using Apple's Metal API for GPU execution.
     */
    struct ImplMetal {
        /** Float Type of Implementation, Required by the Benchmark **/
        using float_type = float;

        /** The actual physical Metal device (i.e., discrete or integrated GPU) **/
        MTL::Device *device = nullptr;
        /** Compiled library holding the whole compute kernel function **/
        MTL::Library *library = nullptr;
        /** Handle to the vector-addition compute function (kernel) inside the library **/
        MTL::Function *function = nullptr;
        /**
         * Compute pipeline state object used to encode and execute the kernel
         * This can be referred to as the prepared statement/ pipline on the GPU
         * It only requires to be started/ and its inputs and outputs need to be set
         */
        MTL::ComputePipelineState *pipelineState = nullptr;
        /** Command queue representing the execution queue for command buffers **/
        MTL::CommandQueue *commandQueue = nullptr;

        /**
         * Constructor: Initialize Metal device, compile the compute shader, create pipeline and command queue.
         */
        ImplMetal() {
            NS::Error *error = nullptr;

            // Select default Metal device; throws if Metal is unavailable on this system
            device = MTL::CreateSystemDefaultDevice();
            if (!device) {
                throw std::runtime_error("Metal is not supported on this device!");
            }

            // Simple compute shader performing C[i] = A[i] + B[i]
            const char *shader = R"(
            #include <metal_stdlib>
            using namespace metal;

            // The kernel receives two input buffers (inA, inB) and writes to result.
            // Each thread handles one element identified by thread_position_in_grid.
            kernel void vector_add(const device float* inA [[buffer(0)]],
                                   const device float* inB [[buffer(1)]],
                                   device float* result [[buffer(2)]],
                                   uint id [[thread_position_in_grid]]) {
                result[id] = inA[id] + inB[id];
            }
            )";

            // Create a Metal library from the shader source
            library = device->newLibrary(NS::String::string(shader, NS::UTF8StringEncoding), nullptr, &error);
            if (!library) {
                throw std::runtime_error("Failed to load shader: " +
                                         std::string(error->localizedDescription()->utf8String()));
            }

            // Extract the kernel function and create the compute pipeline
            function = library->newFunction(NS::String::string("vector_add", NS::UTF8StringEncoding));
            pipelineState = device->newComputePipelineState(function, &error);
            if (!pipelineState) {
                throw std::runtime_error("Failed to create pipeline state: " +
                                         std::string(error->localizedDescription()->utf8String()));
            }

            // Create a command queue to submit work to the GPU
            commandQueue = device->newCommandQueue();
        }

        /**
         * Destructor: Release Metal resources.
         */
        ~ImplMetal() {
            device->release();
            library->release();
            function->release();
            pipelineState->release();
            commandQueue->release();
        }

        /**
         * Executes vector addition on the GPU using Metal and returns the result vector
         * together with the GPU time measured via Metal timestamps.
         *
         * a - First input vector
         * b - Second input vector (must be same size as 'a')
         */
        std::pair<std::vector<float_type>, double> operator()(const std::vector<float_type> &a,
                                                              const std::vector<float_type> &b) {
            // Allocate result vector on the host
            std::vector<float_type> result(a.size(), 0.0);

            // Create shared-memory buffers so CPU and GPU can access the same memory
            auto bufferA = device->newBuffer(a.data(), sizeof(float_type) * a.size(), MTL::ResourceStorageModeShared);
            auto bufferB = device->newBuffer(b.data(), sizeof(float_type) * b.size(), MTL::ResourceStorageModeShared);
            auto bufferResult =
                device->newBuffer(result.data(), sizeof(float_type) * result.size(), MTL::ResourceStorageModeShared);

            // Create command buffer and compute encoder to record GPU work
            // The Buffer acts like a wrapper to start the Kernel
            auto commandBuffer = commandQueue->commandBuffer();
            // The Encoder acts like the specification of the Kernel (Input/ Output Arguments, Thread Size, etc.)
            auto computeEncoder = commandBuffer->computeCommandEncoder();

            // Bind pipeline and buffers
            computeEncoder->setComputePipelineState(pipelineState);
            computeEncoder->setBuffer(bufferA, 0, 0);
            computeEncoder->setBuffer(bufferB, 0, 1);
            computeEncoder->setBuffer(bufferResult, 0, 2);

            // Decide launch configuration and add it to the pipeline
            // Grid size equals number of elements; choose a feasible threadgroup size
            size_t dataSize = a.size();
            auto gridSize = MTL::Size(dataSize, 1, 1);
            auto maxThreadGroupSize = pipelineState->maxTotalThreadsPerThreadgroup();
            auto threadGroupSize = MTL::Size(std::min(maxThreadGroupSize, dataSize), 1, 1);
            computeEncoder->dispatchThreads(gridSize, threadGroupSize);
            computeEncoder->endEncoding();

            // Submit and wait for completion
            commandBuffer->commit();
            commandBuffer->waitUntilCompleted();

            // Measure GPU time using Metal timestamps (seconds)
            const double start = commandBuffer->GPUStartTime();
            const double end = commandBuffer->GPUEndTime();
            const double gpuTimeSeconds = end - start;

            // Copy results back to host memory
            memcpy(result.data(), bufferResult->contents(), sizeof(float_type) * result.size());

            // Release temporary buffers (Metal objects are ref-counted)
            bufferA->release();
            bufferB->release();
            bufferResult->release();
            return std::make_pair(result, gpuTimeSeconds);
        }
    };
} // namespace ppb

BENCHMARK(ppb::VectorAddition<ppb::ImplMetal>::benchmark)
    ->Name("VecAdd-Metal-Float")
#ifdef PPB_MEASURE_ONLY_KERNEL
    ->UseManualTime()
#endif
    ->RangeMultiplier(10)
    ->Range(1e3, 1e8)
    ->Complexity();

int main(int argc, char **argv) {
    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
}
