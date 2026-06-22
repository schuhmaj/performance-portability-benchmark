#include "polyhedralGravity/PolyhedralGravityDefinitions.h"

#include <CL/cl.h>
#include <CL/cl_ext.h>
#undef cl_khr_command_buffer
#undef cl_khr_command_buffer_multi_device
#undef cl_khr_command_buffer_mutable_dispatch

// The legacy OpenCL 1.x C++ binding <CL/cl.hpp> is no longer shipped by the
// Khronos headers; use the current C++ binding (formerly cl2.hpp) instead.
#define CL_HPP_TARGET_OPENCL_VERSION 200
#define CL_HPP_MINIMUM_OPENCL_VERSION 110
#include <CL/opencl.hpp>

#include "common/opencl/OpenCLUtility.h"

#include "opencl_eval.h"
#include "opencl_init.h"
#include "opencl_sum.h"

#if FLOAT_BITS == 32
using VectorTypeCl = cl_float3;
using VectorTypeCl4 = cl_float4;
using VectorTypeCl16 = cl_float16;
const char *COMPILE_ARGS = "-cl-std=CL2.0 -D FloatType=float -D FloatType3=float3 -D FloatType4=float4 -D FloatType16=float16";
#elif FLOAT_BITS == 64
using VectorTypeCl = cl_double3;
using VectorTypeCl4 = cl_double4;
using VectorTypeCl16 = cl_double16;
const char *COMPILE_ARGS = "-cl-std=CL2.0 -D FloatType=double -D FloatType3=double3 -D FloatType4=double4 -D FloatType16=double16";
#else
#error "Invliad float bits size"
#endif

GlobalResources::GlobalResources(int &argc, char *argv[]) {
}
GlobalResources::~GlobalResources() = default;

void compile_and_check(cl::Device &device, cl::Program &program) {
    program.build({device}, COMPILE_ARGS);

    cl_build_status status;
    program.getBuildInfo(device, CL_PROGRAM_BUILD_STATUS, &status);
    if (status == CL_BUILD_SUCCESS) { return; }

    std::string build_log;
    program.getBuildInfo(device, CL_PROGRAM_BUILD_LOG, &build_log);
    throw std::runtime_error(build_log);
}

class GravityEvaluable : public GravityEvaluableBase {
public:
    GravityEvaluable(
            const std::vector<Array3> &Vertices,
            const std::vector<IndexArray3> &Faces,
            const double density)
        : GravityEvaluableBase(Vertices, Faces, density) {
        // Scan every platform for a GPU rather than assuming platform[0]/device[0]
        // exist: with the oneAPI runtime the GPU is often not the first platform,
        // and indexing an empty list here segfaults instead of erroring cleanly.
        cl::Device device{opencl_utility::getFirstGPU()};

        device.getInfo(CL_DEVICE_MAX_WORK_GROUP_SIZE, &maxWorkGroupSize);

        context = cl::Context(device);
        queue = cl::CommandQueue(context, device);

        program_init = cl::Program(context, ppb::KERNEL_INIT);
        program_eval = cl::Program(context, ppb::KERNEL_EVAL);
        program_sum = cl::Program(context, ppb::KERNEL_SUM);

        cl_build_status status;

        compile_and_check(device, program_init);
        compile_and_check(device, program_eval);
        compile_and_check(device, program_sum);

        kernel_init = cl::Kernel(program_init, "vecadd");
        kernel_eval = cl::Kernel(program_eval, "vecadd");
        kernel_sum = cl::Kernel(program_sum, "sum");

        std::vector<cl_int3> faces(_faces.size());
        for (int i = 0; i < faces.size(); i++) {
            faces[i].x = _faces[i][0];
            faces[i].y = _faces[i][1];
            faces[i].z = _faces[i][2];
        }

        std::vector<VectorTypeCl> vertices(_vertices.size());
        for (int i = 0; i < vertices.size(); i++) {
            vertices[i].x = _vertices[i][0];
            vertices[i].y = _vertices[i][1];
            vertices[i].z = _vertices[i][2];
        }

        buffer_vertices = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, vertices.size() * sizeof(vertices[0]), vertices.data());
        buffer_faces = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, faces.size() * sizeof(faces[0]), faces.data());

        buffer_normals = cl::Buffer(context, CL_MEM_READ_WRITE, _faces.size() * sizeof(VectorTypeCl));
        buffer_segmentVectors = cl::Buffer(context, CL_MEM_READ_WRITE, _faces.size() * sizeof(VectorTypeCl) * 3);
        buffer_segmentNormals = cl::Buffer(context, CL_MEM_READ_WRITE, _faces.size() * sizeof(VectorTypeCl) * 3);

        nWorkGroups = (_faces.size() + local_n - 1) / local_n;

        if (nWorkGroups > 128) {
            const int size = nWorkGroups;
            const int local_size = local_n2;
            nWorkGroups2 = (size + local_size - 1) / local_size;
            results.resize(nWorkGroups2);
        } else {
            nWorkGroups2 = 0;
            results.resize(nWorkGroups);
        }

        buffer_results = cl::Buffer(context, CL_MEM_READ_WRITE, nWorkGroups * sizeof(VectorTypeCl16));
        reduction_buffer = cl::Buffer(context, CL_MEM_READ_WRITE, nWorkGroups2 * sizeof(VectorTypeCl16));

        kernel_init.setArg(0, buffer_vertices);
        kernel_init.setArg(1, buffer_faces);
        kernel_init.setArg(2, buffer_normals);
        kernel_init.setArg(3, buffer_segmentVectors);
        kernel_init.setArg(4, buffer_segmentNormals);
        kernel_init.setArg(5, (int32_t) _faces.size());

        kernel_eval.setArg(0, buffer_vertices);
        kernel_eval.setArg(1, buffer_faces);
        kernel_eval.setArg(2, buffer_normals);
        kernel_eval.setArg(3, buffer_segmentVectors);
        kernel_eval.setArg(4, buffer_segmentNormals);
        kernel_eval.setArg(5, buffer_results);
        kernel_eval.setArg(6, (int32_t) _faces.size());

        kernel_sum.setArg(0, buffer_results);
        kernel_sum.setArg(1, reduction_buffer);
        kernel_sum.setArg(2, (int32_t) nWorkGroups);
    }

    GravityModelResult evaluate(const Array3 &Point) override {
        if (!_initialized) init();

        GravityModelResult result{};

        kernel_eval.setArg(7, (FloatType) Point[0]);
        kernel_eval.setArg(8, (FloatType) Point[1]);
        kernel_eval.setArg(9, (FloatType) Point[2]);

        int global_n = _faces.size();
        global_n = ((global_n + local_n - 1) / local_n) * local_n;

        cl::NDRange global(global_n);
        cl::NDRange local(local_n);
        queue.enqueueNDRangeKernel(kernel_eval, cl::NullRange, global, local);


        if (nWorkGroups2) {
            cl::NDRange global(nWorkGroups2 * local_n2);
            cl::NDRange local(local_n2);

            queue.enqueueNDRangeKernel(kernel_sum, cl::NullRange, global, local);
            queue.enqueueReadBuffer(reduction_buffer, CL_TRUE, 0, sizeof(results[0]) * results.size(), results.data());
        } else {
            queue.enqueueReadBuffer(buffer_results, CL_TRUE, 0, sizeof(results[0]) * results.size(), results.data());
        }

        queue.finish();

        for (int i = 0; i < results.size(); i++) {
            result.potential += results[i].w;
            result.acceleration[0] += results[i].x;
            result.acceleration[1] += results[i].y;
            result.acceleration[2] += results[i].z;

            for (int j = 0; j < 6; j++) {
                result.gradiometricTensor.data[j] += results[i].s[j + 4];
            }
        }

        const double prefix = GRAVITATIONAL_CONSTANT * _density;
        result.potential = (result.potential * prefix) / 2.0;
        result.acceleration = result.acceleration * (-1.0 * prefix);
        result.gradiometricTensor = result.gradiometricTensor * prefix;

        return result;
    }

private:
    void init() {
        cl::NDRange global(_faces.size());
        queue.enqueueNDRangeKernel(kernel_init, cl::NullRange, global);
        queue.finish();


        _initialized = true;
    }

    cl::Context context;
    cl::CommandQueue queue;

    size_t maxWorkGroupSize;
    int nWorkGroups;
    int nWorkGroups2;

#if FLOAT_BITS == 32
    int local_n = 16;
    int local_n2 = 16;
#else
    int local_n = 32;
    int local_n2 = 32;
#endif

    cl::Program program_init;
    cl::Program program_eval;
    cl::Program program_sum;

    cl::Kernel kernel_init;
    cl::Kernel kernel_eval;
    cl::Kernel kernel_sum;

    cl::Buffer buffer_vertices;
    cl::Buffer buffer_faces;
    cl::Buffer buffer_normals;
    cl::Buffer buffer_segmentVectors;
    cl::Buffer buffer_segmentNormals;

    cl::Buffer reduction_buffer;
    cl::Buffer buffer_results;

    std::vector<VectorTypeCl16> results;
};

std::unique_ptr<GravityEvaluableBase> create_gravity_evaluable(
        const std::vector<Array3> &Vertices,
        const std::vector<IndexArray3> &Faces,
        double density) {
    return std::make_unique<GravityEvaluable>(Vertices, Faces, density);
}
