#include <Metal/Metal.hpp>
#include <benchmark/benchmark.h>
#include <iostream>
#include <vector>
#include "VectorAddition.h"

MTL::Device* device = nullptr;
MTL::Library* library = nullptr;
MTL::Function* function = nullptr;
MTL::ComputePipelineState* pipelineState = nullptr;

template<typename FloatType>
std::vector<FloatType> ppb::VectorAddition<FloatType>::operator()() {
        NS::Error* error = nullptr;

        // Prepare buffers
        auto commandQueue = device->newCommandQueue();
        auto bufferA = device->newBuffer(_inA.data(), sizeof(FloatType) * _inA.size(), MTL::ResourceStorageModeShared);
        auto bufferB = device->newBuffer(_inB.data(), sizeof(FloatType) * _inB.size(), MTL::ResourceStorageModeShared);
        auto bufferResult = device->newBuffer(_outC.data(), sizeof(FloatType) * _outC.size(), MTL::ResourceStorageModeShared);

        // Execute the compute function
        auto commandBuffer = commandQueue->commandBuffer();
        auto computeEncoder = commandBuffer->computeCommandEncoder();

        computeEncoder->setComputePipelineState(pipelineState);
        computeEncoder->setBuffer(bufferA, 0, 0);
        computeEncoder->setBuffer(bufferB, 0, 1);
        computeEncoder->setBuffer(bufferResult, 0, 2);

        // Dispatch threads
        size_t dataSize = _inA.size();
        auto gridSize = MTL::Size(dataSize, 1, 1);
        auto maxThreadGroupSize = pipelineState->maxTotalThreadsPerThreadgroup();
        auto threadGroupSize = MTL::Size(std::min(maxThreadGroupSize, dataSize), 1, 1);

        computeEncoder->dispatchThreads(gridSize, threadGroupSize);
        computeEncoder->endEncoding();

        commandBuffer->commit();
        commandBuffer->waitUntilCompleted();

        // Retrieve results
        memcpy(_outC.data(), bufferResult->contents(), sizeof(FloatType) * _outC.size());

        // Clean up
        bufferA->release();
        bufferB->release();
        bufferResult->release();
        commandQueue->release();

        return _outC;
}


// Instantiate a benchmark using single precision
template std::vector<float> ppb::VectorAddition<float>::operator()();
BENCHMARK(ppb::VectorAddition<float>::vectorAdditionBenchmark)->Name("VecAdd-MetalCpp-Float")->RangeMultiplier(10)->Range(1e3, 1e8)->Complexity();

int main(int argc, char** argv) {
    benchmark::Initialize(&argc, argv);
    NS::Error* error = nullptr;

    // Initialize Metal device
    device = MTL::CreateSystemDefaultDevice();
    if (!device) {
        throw std::runtime_error("Metal is not supported on this device!");
    }

    const char* VectorAddShaderSource = R"(
            #include <metal_stdlib>
            using namespace metal;

            kernel void vector_add(const device float* inA [[buffer(0)]],
                                   const device float* inB [[buffer(1)]],
                                   device float* result [[buffer(2)]],
                                   uint id [[thread_position_in_grid]]) {
                result[id] = inA[id] + inB[id];
            }
        )";

    library = device->newLibrary(NS::String::string(VectorAddShaderSource, NS::UTF8StringEncoding), nullptr, &error);
    if (!library) {
        throw std::runtime_error("Failed to load shader: " + std::string(error->localizedDescription()->utf8String()));
    }
    function = library->newFunction(NS::String::string("vector_add", NS::UTF8StringEncoding));
    pipelineState = device->newComputePipelineState(function, &error);
    if (!pipelineState) {
        throw std::runtime_error("Failed to create pipeline state: " + std::string(error->localizedDescription()->utf8String()));
    }
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
    device->release();
    library->release();
    function->release();
    pipelineState->release();
}