#include "polyhedralGravity/PolyhedralGravityDefinitions.h"

#include "boost/compute.hpp"

#include "opencl_eval.h"
#include "opencl_init.h"
#include "opencl_sum.h"

namespace bc = boost::compute;

// The OpenCL kernels operate on 3-component vectors (float3 / int3 / ...).
// Boost.Compute only ships vector types for the sizes 2, 4, 8 and 16, but an
// OpenCL `float3` occupies the same 16 bytes as a `float4`, so the 4-component
// Boost.Compute types are layout-compatible host mirrors for the kernel data.
#if FLOAT_BITS == 32
using VecType = bc::float4_;
using Vec16Type = bc::float16_;
using ScalarType = cl_float;
const char *COMPILE_ARGS = "-cl-std=CL2.0 -D FloatType=float -D FloatType3=float3 -D FloatType4=float4 -D FloatType16=float16";
#elif FLOAT_BITS == 64
using VecType = bc::double4_;
using Vec16Type = bc::double16_;
using ScalarType = cl_double;
const char *COMPILE_ARGS = "-cl-std=CL2.0 -D FloatType=double -D FloatType3=double3 -D FloatType4=double4 -D FloatType16=double16";
#else
#error "Invalid float bits size"
#endif

using IntVecType = bc::int4_;

GlobalResources::GlobalResources(int &argc, char *argv[]) {}
GlobalResources::~GlobalResources() = default;

class GravityEvaluable : public GravityEvaluableBase {
public:
    GravityEvaluable(
            const std::vector<Array3> &Vertices,
            const std::vector<IndexArray3> &Faces,
            const double density)
        : GravityEvaluableBase(Vertices, Faces, density)
        , device{bc::system::default_device()}
        , context{device}
        , queue{context, device, bc::command_queue::enable_profiling} {

        maxWorkGroupSize = device.get_info<size_t>(CL_DEVICE_MAX_WORK_GROUP_SIZE);

        program_init = bc::program::build_with_source(std::string(ppb::KERNEL_INIT), context, COMPILE_ARGS);
        program_eval = bc::program::build_with_source(std::string(ppb::KERNEL_EVAL), context, COMPILE_ARGS);
        program_sum = bc::program::build_with_source(std::string(ppb::KERNEL_SUM), context, COMPILE_ARGS);

        kernel_init = bc::kernel(program_init, "vecadd");
        kernel_eval = bc::kernel(program_eval, "vecadd");
        kernel_sum = bc::kernel(program_sum, "sum");

        // Pack the host-side geometry into the layout-compatible vector types.
        std::vector<IntVecType> faces(_faces.size());
        for (size_t i = 0; i < faces.size(); ++i) {
            faces[i] = IntVecType(static_cast<cl_int>(_faces[i][0]),
                                  static_cast<cl_int>(_faces[i][1]),
                                  static_cast<cl_int>(_faces[i][2]),
                                  0);
        }

        std::vector<VecType> vertices(_vertices.size());
        for (size_t i = 0; i < vertices.size(); ++i) {
            vertices[i] = VecType(static_cast<ScalarType>(_vertices[i][0]),
                                  static_cast<ScalarType>(_vertices[i][1]),
                                  static_cast<ScalarType>(_vertices[i][2]),
                                  0);
        }

        buffer_vertices = bc::vector<VecType>(vertices.size(), context);
        buffer_faces = bc::vector<IntVecType>(faces.size(), context);
        bc::copy(vertices.begin(), vertices.end(), buffer_vertices.begin(), queue);
        bc::copy(faces.begin(), faces.end(), buffer_faces.begin(), queue);

        buffer_normals = bc::vector<VecType>(_faces.size(), context);
        buffer_segmentVectors = bc::vector<VecType>(_faces.size() * 3, context);
        buffer_segmentNormals = bc::vector<VecType>(_faces.size() * 3, context);

        nWorkGroups = (static_cast<int>(_faces.size()) + local_n - 1) / local_n;

        if (nWorkGroups > 128) {
            nWorkGroups2 = (nWorkGroups + local_n2 - 1) / local_n2;
            results.resize(nWorkGroups2);
        } else {
            nWorkGroups2 = 0;
            results.resize(nWorkGroups);
        }

        buffer_results = bc::vector<Vec16Type>(nWorkGroups, context);
        // Boost.Compute (like OpenCL) rejects zero-sized buffers; keep at least one
        // element even when the second reduction stage is unused.
        reduction_buffer = bc::vector<Vec16Type>(std::max(nWorkGroups2, 1), context);

        kernel_init.set_arg(0, buffer_vertices);
        kernel_init.set_arg(1, buffer_faces);
        kernel_init.set_arg(2, buffer_normals);
        kernel_init.set_arg(3, buffer_segmentVectors);
        kernel_init.set_arg(4, buffer_segmentNormals);
        kernel_init.set_arg(5, static_cast<cl_int>(_faces.size()));

        kernel_eval.set_arg(0, buffer_vertices);
        kernel_eval.set_arg(1, buffer_faces);
        kernel_eval.set_arg(2, buffer_normals);
        kernel_eval.set_arg(3, buffer_segmentVectors);
        kernel_eval.set_arg(4, buffer_segmentNormals);
        kernel_eval.set_arg(5, buffer_results);
        kernel_eval.set_arg(6, static_cast<cl_int>(_faces.size()));

        kernel_sum.set_arg(0, buffer_results);
        kernel_sum.set_arg(1, reduction_buffer);
        kernel_sum.set_arg(2, static_cast<cl_int>(nWorkGroups));
    }

    GravityModelResult evaluate(const Array3 &Point) override {
        if (!_initialized) init();

        GravityModelResult result{};

        kernel_eval.set_arg(7, static_cast<ScalarType>(Point[0]));
        kernel_eval.set_arg(8, static_cast<ScalarType>(Point[1]));
        kernel_eval.set_arg(9, static_cast<ScalarType>(Point[2]));

        const size_t local_size = static_cast<size_t>(local_n);
        const size_t global_size = ((_faces.size() + local_size - 1) / local_size) * local_size;
        queue.enqueue_nd_range_kernel(kernel_eval, 1, nullptr, &global_size, &local_size);

        if (nWorkGroups2) {
            const size_t local_size2 = static_cast<size_t>(local_n2);
            const size_t global_size2 = static_cast<size_t>(nWorkGroups2) * local_size2;
            queue.enqueue_nd_range_kernel(kernel_sum, 1, nullptr, &global_size2, &local_size2);
            bc::copy(reduction_buffer.begin(), reduction_buffer.begin() + results.size(), results.begin(), queue);
        } else {
            bc::copy(buffer_results.begin(), buffer_results.begin() + results.size(), results.begin(), queue);
        }

        queue.finish();

        for (size_t i = 0; i < results.size(); ++i) {
            // Layout of the packed result (see opencl_eval.cl):
            //   [0..2] acceleration, [3] potential, [4..9] gradiometric tensor
            result.potential += results[i][3];
            result.acceleration[0] += results[i][0];
            result.acceleration[1] += results[i][1];
            result.acceleration[2] += results[i][2];

            for (int j = 0; j < 6; ++j) {
                result.gradiometricTensor.data[j] += results[i][j + 4];
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
        const size_t global = _faces.size();
        queue.enqueue_nd_range_kernel(kernel_init, 1, nullptr, &global, nullptr);
        queue.finish();

        _initialized = true;
    }

    bc::device device;
    bc::context context;
    bc::command_queue queue;

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

    bc::program program_init;
    bc::program program_eval;
    bc::program program_sum;

    bc::kernel kernel_init;
    bc::kernel kernel_eval;
    bc::kernel kernel_sum;

    bc::vector<VecType> buffer_vertices;
    bc::vector<IntVecType> buffer_faces;
    bc::vector<VecType> buffer_normals;
    bc::vector<VecType> buffer_segmentVectors;
    bc::vector<VecType> buffer_segmentNormals;

    bc::vector<Vec16Type> buffer_results;
    bc::vector<Vec16Type> reduction_buffer;

    std::vector<Vec16Type> results;
};

std::unique_ptr<GravityEvaluableBase> create_gravity_evaluable(
        const std::vector<Array3> &Vertices,
        const std::vector<IndexArray3> &Faces,
        double density) {
    return std::make_unique<GravityEvaluable>(Vertices, Faces, density);
}
