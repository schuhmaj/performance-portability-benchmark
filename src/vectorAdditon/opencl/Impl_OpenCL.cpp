#include <benchmark/benchmark.h>
#include <iostream>
#include "VectorAddition.h"
#include "opencl/util/OpenCLUtility.h""

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif


template <typename FloatType>
struct ppb::VectorAddition<FloatType>::impl {
    cl_context context = nullptr;
    cl_command_queue queue = nullptr;
    cl_device_id device = nullptr;
    cl_program program = nullptr;
    cl_kernel kernel = nullptr;

    cl_mem deviceA = nullptr;
    cl_mem deviceB = nullptr;

    impl(size_t size, const std::vector<FloatType>& a, const std::vector<FloatType>& b) {
        // 0. Get device
        device = util::getFirstGPU();

        // 1. Context & queue
        cl_int err;
        context = clCreateContext(0, 1, &device, nullptr, nullptr, &err);
        queue = clCreateCommandQueue(context, device, 0, &err);

        // 2. Buffers (allocate once, refill on demand)
        deviceA = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,  size * sizeof(FloatType), const_cast<FloatType*>(a.data()), &err);
        deviceB = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,  size * sizeof(FloatType), const_cast<FloatType*>(b.data()), &err);

        // 3. OpenCL program & kernel
        std::string kernelSource;
        if constexpr (std::is_same_v<FloatType, float>) {
            kernelSource = "__kernel void add_vector(__global const float* a, __global const float* b, __global float* c) {"
            " int gid = get_global_id(0);"
            " c[gid] = a[gid] + b[gid];"
            " }";
        } else if constexpr (std::is_same_v<FloatType, double>) {
            kernelSource = "__kernel void add_vector(__global const double* a, __global const double* b, __global double* c) {"
            " int gid = get_global_id(0);"
            " c[gid] = a[gid] + b[gid];"
            " }";
        } else {
            static_assert(std::is_same_v<FloatType, float> || std::is_same_v<FloatType, double>, "Unsupported type");
        }
        const char* kernelProg = kernelSource.c_str();
        program = clCreateProgramWithSource(context, 1, &kernelProg, nullptr, &err);

        err = clBuildProgram(program, 0, nullptr, nullptr, nullptr, nullptr);
        kernel = clCreateKernel(program, "add_vector", &err);
    }

    ~impl() {
        clReleaseMemObject(deviceA);
        clReleaseMemObject(deviceB);
        clReleaseProgram(program);
        clReleaseKernel(kernel);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
    }
};


template<typename FloatType>
void ppb::VectorAddition<FloatType>::init() {
    _impl = std::make_unique<impl>(_size, _inA, _inB);
}

static size_t roundUp(int group_size, int global_size) {
    int r = global_size % group_size;
    return r == 0 ? global_size : global_size + group_size - r;
}

template <typename FloatType>
std::vector<FloatType> ppb::VectorAddition<FloatType>::operator()() {
    cl_int err = 0;
    std::vector<FloatType> result(_size);
    cl_mem resultBuffer = clCreateBuffer(_impl->context, CL_MEM_WRITE_ONLY, _size * sizeof(FloatType), nullptr, nullptr);;

    err = clSetKernelArg(_impl->kernel, 0, sizeof(cl_mem), &(_impl->deviceA));
    if (err != CL_SUCCESS) throw std::runtime_error("SetKernelArg 0 failed");
    err = clSetKernelArg(_impl->kernel, 1, sizeof(cl_mem), &(_impl->deviceB));
    if (err != CL_SUCCESS) throw std::runtime_error("SetKernelArg 1 failed");
    err = clSetKernelArg(_impl->kernel, 2, sizeof(cl_mem), &resultBuffer);
    if (err != CL_SUCCESS) throw std::runtime_error("SetKernelArg 2 failed");

    // 3. Launch kernel
    const size_t localWorkSize = 1024;
    const size_t globalWorkSize = roundUp(localWorkSize, _size);
    err = clEnqueueNDRangeKernel(_impl->queue, _impl->kernel, 1, nullptr, &globalWorkSize, &localWorkSize, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) throw std::runtime_error("EnqueueNDRangeKernel failed");

    // 4. Copy result C back
    err = clEnqueueReadBuffer(_impl->queue, resultBuffer, CL_TRUE, 0, _size * sizeof(FloatType), const_cast<FloatType*>(result.data()), 0, nullptr, nullptr);
    if (err != CL_SUCCESS) throw std::runtime_error("ReadBuffer result failed: ");
    clFinish(_impl->queue);
    clReleaseMemObject(resultBuffer);

    return result;
}



template std::vector<float> ppb::VectorAddition<float>::operator()();
BENCHMARK(ppb::VectorAddition<float>::benchmark)
    ->Name("VecAdd-OpenCL-Float")
    ->RangeMultiplier(10)
    ->Range(1e3, 1e8)
    ->Complexity();

template std::vector<double> ppb::VectorAddition<double>::operator()();
BENCHMARK(ppb::VectorAddition<double>::benchmark)
    ->Name("VecAdd-OpenCL-Double")
    ->RangeMultiplier(10)
    ->Range(1e3, 1e8)
    ->Complexity();

int main(int argc, char **argv) {
    auto gpu = util::getFirstGPU();
    std::cout << "GPU Name: " << util::getDeviceName(gpu) << '\n';

    benchmark::MaybeReenterWithoutASLR(argc, argv);
    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
}
