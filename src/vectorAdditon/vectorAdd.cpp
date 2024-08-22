// OpenCL header
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#include <iostream>
#include <vector>

#include "OpenCLUtlity.h"

const char* kernelSrc = \
"__kernel void add_vector( __global const int *a, __global const int *b, __global int *c) {          \
    int gid = get_global_id(0);                                                                      \
    c[gid] = a[gid] + b[gid];                                                                        \
}                                                                                                    \
";

size_t roundUp(int group_size, int global_size) {
    int r = global_size % group_size;
    if(r == 0) {
        return global_size;
    } else {
        return global_size + group_size - r;
    }
}


int main() {
    cl_int err;
    // Create the Problem
    constexpr int SIZE = 10000;
    std::vector<int> inputA(SIZE), inputB(SIZE), outputC(SIZE);
    for(int i=0; i < SIZE; i++) {
        inputA[i] = i;
        inputB[i] = i;
    }

    // Get platform and device
    const auto platform1 = util::getOpenCLPlattforms()[0];
    std::vector<cl_device_id> gpus{};
    try {
        gpus = util::getOpenCLDevices(platform1, CL_DEVICE_TYPE_GPU);
    } catch (std::runtime_error &e) {
        std::cout << "No GPU devices found!";
    }
    const auto devices = util::getOpenCLDevices(platform1, CL_DEVICE_TYPE_ALL);
    cl_device_id device;
    if (gpus.empty()) {
        std::cout << "No GPU devices found! Chosing first device from all available devices" << '\n';
        device = devices[0];
    } else {
        device = gpus[0];
    }
    std::cout << "Using the Device: " << util::getDeviceName(device) << '\n';

    // Create Context
    cl_context context = clCreateContext(0, 1, &device, nullptr, nullptr, &err);

    // Create Command Queue
    cl_command_queue queue = clCreateCommandQueue(context, device, 0, &err);

    // Create Program
    cl_program program = clCreateProgramWithSource(context, 1, (const char **)&kernelSrc, nullptr, &err);

    // Build Program
    err = clBuildProgram(program, 0, nullptr, nullptr, nullptr, nullptr);

    // Create Kernel
    cl_kernel kernel = clCreateKernel(program, "add_vector", &err);

    // Create Buffers
    cl_mem a_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, inputA.size() * sizeof(int), inputA.data(), nullptr);
    cl_mem b_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, inputB.size() * sizeof(int), inputB.data(), nullptr);
    cl_mem c_buffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, outputC.size() * sizeof(int), nullptr, nullptr);

    // Set Kernel Arguments
    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &a_buffer);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &b_buffer);
    err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &c_buffer);

    // Execute Kernel
    size_t localWorkSize = 64, globalWorkSize = roundUp(localWorkSize, SIZE);
    err = clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &globalWorkSize, &localWorkSize, 0, nullptr, nullptr);

    // Copy results back from the output buffer
    err = clEnqueueReadBuffer(queue, c_buffer, CL_TRUE, 0, outputC.size() * sizeof(int), outputC.data(), 0, nullptr, nullptr);

    // Clean up
    clReleaseMemObject(a_buffer);
    clReleaseMemObject(b_buffer);
    clReleaseMemObject(c_buffer);
    clReleaseProgram(program);
    clReleaseKernel(kernel);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    // Print results
    for (const auto& el : outputC) {
        std::cout << el << ' ';
    }

    return 0;
}