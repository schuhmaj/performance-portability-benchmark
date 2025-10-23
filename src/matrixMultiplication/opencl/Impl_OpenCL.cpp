#include "Impl_OpenCL.h"
#include "common/UtilityFloatArithmetic.h"
#include <benchmark/benchmark.h>
#include <iostream>
#include <utility>

template <typename FloatType>
ppb::ImplOpenCL<FloatType>::ImplOpenCL() {
    // 0. Get device
    device = opencl_utility::getFirstGPU();

    // 1. Context & queue
    cl_int err;
    context = clCreateContext(0, 1, &device, nullptr, nullptr, &err);
    queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err);

    // 3. OpenCL program & kernel
    std::string kernelSource;
    const char *kernelProg = KERNEL_SOURCE;
    program = clCreateProgramWithSource(context, 1, &kernelProg, nullptr, &err);
    err = clBuildProgram(program, 0, nullptr, nullptr, nullptr, nullptr);
    if constexpr (std::is_same_v<FloatType, float>) {
        kernel = clCreateKernel(program, "matrix_multiplication_float", &err);
    } else if constexpr (std::is_same_v<FloatType, double>) {
#ifdef __APPLE__
    static_assert(true, "Not possible on the Apple Platform - Metal doesn't have a float64 type!");
#endif
        kernel = clCreateKernel(program, "matrix_multiplication_double", &err);
    }
    else {
        static_assert(std::is_same_v<FloatType, float> || std::is_same_v<FloatType, double>, "Unsupported type");
    }
}

template <typename FloatType>
ppb::ImplOpenCL<FloatType>::~ImplOpenCL() {
    clReleaseProgram(program);
    clReleaseKernel(kernel);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    clReleaseDevice(device);
}

template <typename FloatType>
std::pair<std::vector<FloatType>, double>
ppb::ImplOpenCL<FloatType>::operator()(const std::vector<FloatType> &a, const std::vector<FloatType> &b,
                                       const MatrixMultiplicationConfig &config) {
    const size_t size = a.size();
    cl_int err = 0;
    cl_mem deviceA = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, size * sizeof(FloatType),
                                    const_cast<FloatType *>(a.data()), &err);
    cl_mem deviceB = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, size * sizeof(FloatType),
                                    const_cast<FloatType *>(b.data()), &err);

    std::vector<FloatType> result(size);
    cl_mem resultBuffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, size * sizeof(FloatType), nullptr, nullptr);

    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &(deviceA));
    if (err != CL_SUCCESS)
        throw std::runtime_error("SetKernelArg Buffer A failed");
    err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &(deviceB));
    if (err != CL_SUCCESS)
        throw std::runtime_error("SetKernelArg Buffer B failed");
    err = clSetKernelArg(kernel, 2, sizeof(cl_mem), &resultBuffer);
    if (err != CL_SUCCESS)
        throw std::runtime_error("SetKernelArg Buffer C failed");
    err = clSetKernelArg(kernel, 3, sizeof(int), &config.m);
    if (err != CL_SUCCESS)
        throw std::runtime_error("SetKernelArg Integer M failed");
    err = clSetKernelArg(kernel, 4, sizeof(int), &config.n);
    if (err != CL_SUCCESS)
        throw std::runtime_error("SetKernelArg Integer N failed");
    err = clSetKernelArg(kernel, 5, sizeof(int), &config.k);
    if (err != CL_SUCCESS)
        throw std::runtime_error("SetKernelArg Integer K failed");

    // 2D launch configuration
    const size_t localSize[2] = {32, 32};
    const size_t globalSize[2] = {
        util::roundUp<size_t>(config.m, localSize[0]),
        util::roundUp<size_t>(config.n, localSize[1])
    };

    // Launch and time
    cl_event event;
    cl_ulong start, end;
    err = clEnqueueNDRangeKernel(queue, kernel, 2, nullptr, globalSize, localSize, 0, nullptr, &event);
    if (err != CL_SUCCESS) throw std::runtime_error("EnqueueNDRangeKernel failed");

    // Read back C (size M*N)
    err = clEnqueueReadBuffer(queue, resultBuffer, CL_TRUE, 0,
                              static_cast<size_t>(config.m) * config.n * sizeof(FloatType),
                              result.data(), 0, nullptr, nullptr);
    if (err != CL_SUCCESS) throw std::runtime_error("ReadBuffer result failed");
    clFinish(queue);

    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(start), &start, nullptr);
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(end), &end, nullptr);
    double elapsed_nanoseconds = end - start;

    clReleaseMemObject(deviceA);
    clReleaseMemObject(deviceB);
    clReleaseMemObject(resultBuffer);
    clReleaseEvent(event);
    return std::make_pair(std::move(result), elapsed_nanoseconds);
}

template class ppb::ImplOpenCL<float>;