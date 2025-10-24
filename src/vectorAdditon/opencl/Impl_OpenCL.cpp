#include <benchmark/benchmark.h>
#include <iostream>
#include <utility>
#include "vectorAdditon/VectorAddition.h"
#include "common/opencl/OpenCLUtility.h"
#include "common/UtilityFloatArithmetic.h"
#include "VectorAdditionKernel.h"

#ifdef __APPLE__
#include <OpenCL/opencl.h>
constexpr size_t WORKGROUP_SIZE = 32;
#else
#include <CL/cl.h>
constexpr size_t WORKGROUP_SIZE = 1024;
#endif

namespace ppb {
    template <typename FloatType>
    struct ImplOpenCL {
        using float_type = FloatType;

        cl_context context = nullptr;
        cl_command_queue queue = nullptr;
        cl_device_id device = nullptr;
        cl_program program = nullptr;
        cl_kernel kernel = nullptr;

        ImplOpenCL() {
            // 0. Get device
            device = opencl_utility::getFirstGPU();

            // 1. Context & queue
            cl_int err;
            context = clCreateContext(0, 1, &device, nullptr, nullptr, &err);
            queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err);

            // 3. opencl_utility program & kernel
            const char* kernelProg = KERNEL_SOURCE;
            program = clCreateProgramWithSource(context, 1, &kernelProg, nullptr, &err);
            err = clBuildProgram(program, 0, nullptr, nullptr, nullptr, nullptr);

            if constexpr (std::is_same_v<FloatType, float>) {
                kernel = clCreateKernel(program, "add_vector_float", &err);
            } else if constexpr (std::is_same_v<FloatType, double>) {
                kernel = clCreateKernel(program, "add_vector_double", &err);
            } else {
                static_assert(std::is_same_v<FloatType, float> || std::is_same_v<FloatType, double>, "Unsupported type");
            }
        }

        ~ImplOpenCL() {
            clReleaseProgram(program);
            clReleaseKernel(kernel);
            clReleaseCommandQueue(queue);
            clReleaseContext(context);
        }

        std::pair<std::vector<FloatType>, double> operator()(const std::vector<FloatType> &a, const std::vector<FloatType> &b) {
            const size_t size = a.size();
            cl_int err = 0;
            cl_mem deviceA = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,  size * sizeof(FloatType), const_cast<FloatType*>(a.data()), &err);
            cl_mem deviceB = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,  size * sizeof(FloatType), const_cast<FloatType*>(b.data()), &err);

            std::vector<FloatType> result(size);
            cl_mem resultBuffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, size * sizeof(FloatType), nullptr, nullptr);;

            err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &(deviceA));
            if (err != CL_SUCCESS) throw std::runtime_error("SetKernelArg 0 failed");
            err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &(deviceB));
            if (err != CL_SUCCESS) throw std::runtime_error("SetKernelArg 1 failed");
            err = clSetKernelArg(kernel, 2, sizeof(cl_mem), &resultBuffer);
            if (err != CL_SUCCESS) throw std::runtime_error("SetKernelArg 2 failed");

            // 3. Launch kernel and Measure Time
            cl_event event;
            cl_ulong start, end;
            const size_t localWorkSize = WORKGROUP_SIZE;
            const size_t globalWorkSize = util::roundUp(size, localWorkSize);
            err = clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &globalWorkSize, &localWorkSize, 0, nullptr, &event);
            if (err != CL_SUCCESS) throw std::runtime_error("EnqueueNDRangeKernel failed");

            // 4. Copy result C back
            err = clEnqueueReadBuffer(queue, resultBuffer, CL_TRUE, 0, size * sizeof(FloatType), const_cast<FloatType*>(result.data()), 0, nullptr, nullptr);
            if (err != CL_SUCCESS) throw std::runtime_error("ReadBuffer result failed: ");
            clFinish(queue);

            clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(start), &start, nullptr);
            clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(end), &end, nullptr);
            double elapsed_nanoseconds = static_cast<double>(end - start);

            clReleaseMemObject(deviceA);
            clReleaseMemObject(deviceB);
            clReleaseMemObject(resultBuffer);
            clReleaseEvent(event);
            return std::make_pair(result, elapsed_nanoseconds);
        }
    };

    template class ImplOpenCL<float>;
    template class ImplOpenCL<double>;
}

BENCHMARK(ppb::VectorAddition<ppb::ImplOpenCL<float>>::benchmark)
    ->Name("VecAdd-Float-OpenCL")
    ->RangeMultiplier(10)
    ->Range(1e3, 1e8)
#ifdef PPB_MEASURE_ONLY_KERNEL
    ->UseManualTime()
#endif
    ->Complexity();

// Apple Metal Devices don't support Double Precision (as of 29.09.2025 - tested with M1 Pro)
// Hence, building the Kernel will fail on an Apple Chip
// #ifndef __APPLE__
// BENCHMARK(ppb::VectorAddition<ppb::ImplOpenCL<double>>::benchmark)
//     ->Name("VecAdd-Double-OpenCL")
//     ->RangeMultiplier(10)
//     ->Range(1e3, 1e8)
// #ifdef PPB_MEASURE_ONLY_KERNEL
//     ->UseManualTime()
// #endif
//     ->Complexity();
// #endif

int main(int argc, char **argv) {
    benchmark::MaybeReenterWithoutASLR(argc, argv);
    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
}
