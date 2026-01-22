#pragma once
#include <vector>
#include <array>
#include <utility>
#include "matrixMultiplication/MatrixMultiplication.h"
#include "MatrixMultiplicationKernel.h"
#include "common/opencl/OpenCLUtility.h"

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif


namespace ppb {

    template<typename FloatType>
    class ImplOpenCL {

    public:

        using float_type = FloatType;
        using row_major = std::false_type;

        cl_context context = nullptr;
        cl_command_queue queue = nullptr;
        cl_device_id device = nullptr;
        cl_program program = nullptr;
        cl_kernel kernel = nullptr;

        ImplOpenCL();
        ~ImplOpenCL();

        std::pair<std::vector<FloatType>, double> operator()(const std::vector<FloatType> &a, const std::vector<FloatType> &b, const MatrixMultiplicationConfig &config);

    };

    };
