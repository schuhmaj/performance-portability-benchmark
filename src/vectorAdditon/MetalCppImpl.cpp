#include <iostream>
#include <vector>
#include <metal/metal.hpp>


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

int main() {
    // Initialize Metal device
    auto device = MTL::CreateSystemDefaultDevice();

    if (!device) {
        std::cerr << "Metal is not supported on this device!" << std::endl;
        return -1;
    }

    // Load the shader
    NS::Error* error = nullptr;
    auto library = device->newLibrary(NS::String::string(VectorAddShaderSource, NS::UTF8StringEncoding), nullptr, &error);

    if (!library) {
        std::cerr << "Failed to load shader: " << error->localizedDescription()->utf8String() << std::endl;
        return -1;
    }

    auto function = library->newFunction(NS::String::string("vector_add", NS::UTF8StringEncoding));
    auto pipelineState = device->newComputePipelineState(function, &error);

    if (!pipelineState) {
        std::cerr << "Failed to create pipeline state: " << error->localizedDescription()->utf8String() << std::endl;
        return -1;
    }

    // Sample data
    const size_t vectorSize = 1024;
    std::vector<float> inA(vectorSize, 1.0f);
    std::vector<float> inB(vectorSize, 2.0f);
    std::vector<float> result(vectorSize, 0.0f);

    // Buffers
    auto commandQueue = device->newCommandQueue();
    auto bufferA = device->newBuffer(inA.data(), sizeof(float) * vectorSize, MTL::ResourceStorageModeShared);
    auto bufferB = device->newBuffer(inB.data(), sizeof(float) * vectorSize, MTL::ResourceStorageModeShared);
    auto bufferResult = device->newBuffer(result.data(), sizeof(float) * vectorSize, MTL::ResourceStorageModeShared);

    // Command buffer and encoder
    auto commandBuffer = commandQueue->commandBuffer();
    auto computeEncoder = commandBuffer->computeCommandEncoder();

    computeEncoder->setComputePipelineState(pipelineState);
    computeEncoder->setBuffer(bufferA, 0, 0);
    computeEncoder->setBuffer(bufferB, 0, 1);
    computeEncoder->setBuffer(bufferResult, 0, 2);

    // Dispatch threads
    auto gridSize = MTL::Size(vectorSize, 1, 1);
    auto maxThreadGroupSize = pipelineState->maxTotalThreadsPerThreadgroup();
    auto threadGroupSize = MTL::Size(std::min(maxThreadGroupSize, vectorSize), 1, 1);

    computeEncoder->dispatchThreads(gridSize, threadGroupSize);
    computeEncoder->endEncoding();

    commandBuffer->commit();
    commandBuffer->waitUntilCompleted();

    // Retrieve the result and print it
    memcpy(result.data(), bufferResult->contents(), sizeof(float) * vectorSize);

    std::cout << "Result: ";
    for (size_t i = 0; i < 10; ++i) { // Print first 10 elements for brevity
        std::cout << result[i] << " ";
    }
    std::cout << "..." << std::endl;

    // Clean up
    bufferA->release();
    bufferB->release();
    bufferResult->release();
    commandQueue->release();
    pipelineState->release();
    function->release();
    library->release();
    device->release();

    return 0;
}