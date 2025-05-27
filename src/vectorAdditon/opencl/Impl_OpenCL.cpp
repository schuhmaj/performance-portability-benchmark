#include <benchmark/benchmark.h>
#include <iostream>
#include "VectorAddition.h"
#include "util/OpenCLUtlity.h"

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

const char *kernelSrc =
    "__kernel void add_vector( __global const float *a, __global const float *b, __global float *c) {    \
    int gid = get_global_id(0);                                                                      \
    c[gid] = a[gid] + b[gid];                                                                        \
}                                                                                                    \
";

size_t roundUp(int group_size, int global_size) {
    int r = global_size % group_size;
    if (r == 0) {
        return global_size;
    }
    else {
        return global_size + group_size - r;
    }
}


template <typename FloatType>
std::vector<FloatType> VectorAddition<FloatType>::operator()() {
    cl_int err;

    // Step 0: Get the device
    cl_device_id device = util::getFirstGPU();

    // Step 1: Create the compute context and the queue
    cl_context context = clCreateContext(0, 1, &device, nullptr, nullptr, &err);
    cl_command_queue queue = clCreateCommandQueue(context, device, 0, &err);

    // Step 2: Create Buffers between Host and Device Memory
    cl_mem a_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, _inA.size() * sizeof(FloatType),
                                     _inA.data(), nullptr);
    cl_mem b_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, _inB.size() * sizeof(FloatType),
                                     _inB.data(), nullptr);
    cl_mem c_buffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, _outC.size() * sizeof(FloatType), nullptr, nullptr);


    // Step 3a: Create the Kernel
    cl_program program = clCreateProgramWithSource(context, 1, (const char **)&kernelSrc, nullptr, &err);
    err = clBuildProgram(program, 0, nullptr, nullptr, nullptr, nullptr);
    cl_kernel kernel = clCreateKernel(program, "add_vector", &err);

    // Step 3b: Set-Up the Kernels Arguments
    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &a_buffer);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &b_buffer);
    err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &c_buffer);

    // Step 4: Execute the kernel
    const size_t localWorkSize = 1024;
    const size_t globalWorkSize = roundUp(localWorkSize, _inA.size());
    err = clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &globalWorkSize, &localWorkSize, 0, nullptr, nullptr);

    // Step 5: Copy result back to host
    err = clEnqueueReadBuffer(queue, c_buffer, CL_TRUE, 0, _outC.size() * sizeof(FloatType), _outC.data(), 0, nullptr,
                              nullptr);
    clFinish(queue);

    // Step 6: Clean up
    clReleaseMemObject(a_buffer);
    clReleaseMemObject(b_buffer);
    clReleaseMemObject(c_buffer);
    clReleaseProgram(program);
    clReleaseKernel(kernel);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    checkValidity();
    return _outC;
}


template std::vector<float> VectorAddition<float>::operator()();
BENCHMARK(VectorAddition<float>::vectorAdditionBenchmark)
    ->Name("VecAdd-OpenCL-Float")
    ->RangeMultiplier(10)
    ->Range(1e3, 1e8)
    ->Complexity();

// One does not have a dedicated double example here, as Boost.Compute only supports OpenCL types
// Ergo, one only have int and float as potential template specializations

int main(int argc, char **argv) {
    auto gpu = util::getFirstGPU();
    std::cout << "GPU Name: " << util::getDeviceName(gpu) << '\n';

    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
}
