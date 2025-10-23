#pragma once
#include <vector>
#include <array>
#include <utility>
#include "alpaka/alpaka.hpp"
#include "alpaka/example/ExecuteForEachAccTag.hpp"
#include "alpaka/example/ExampleDefaultAcc.hpp"
#include "matrixMultiplication/MatrixMultiplication.h"


namespace ppb {

    template<typename FloatType>
    class ImplAlpaka {

    public:

        using float_type = FloatType;
        using row_major = std::false_type;

        /** Dimensionality of the Problem, for Vector Addition it's 1D */
        using Dim = alpaka::DimInt<2u>;
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

        ImplAlpaka();

        std::pair<std::vector<FloatType>, double> operator()(const std::vector<FloatType> &a, const std::vector<FloatType> &b, const MatrixMultiplicationConfig &config);

    };

    };
