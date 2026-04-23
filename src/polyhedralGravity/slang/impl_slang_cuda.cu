#include "common.h"

#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>

#if FLOAT_BITS == 32
using VectorType = float3;
using VectorType4 = float4;
#include <helper_math.h>
#elif FLOAT_BITS == 64
using VectorType = double3;
using VectorType4 = double4;
#include <helper_math_double.h>
#else
#error "Invliad float bits size"
#endif

struct Result {
    VectorType4 res;
    VectorType4 first;
    VectorType4 second;
    VectorType4 _padding;

    __host__ __device__ Result operator+(const Result &other) const {
        Result result{};
        result.res = res + other.res;
        result.first = other.first + other.first;
        result.second = other.second + other.second;
        return result;
    }
};

GlobalResources::GlobalResources(int &argc, char *argv[]) {
}
GlobalResources::~GlobalResources() = default;

void wrapper_eval(
        void *vertices,
        void *faces,
        void *normals,
        void *segmentVectors,
        void *segmentNormals,
        void *results,
        void *settings,
        unsigned int num_faces,
        FloatType p1,
        FloatType p2,
        FloatType p3,
        bool init);

template<typename T>
struct CudaMemory {
    void *_data{};

    explicit CudaMemory(uint32_t n_elem) {
        cudaMalloc(&_data, n_elem * sizeof(T));
    }

    ~CudaMemory() {
        // Calling this will freeze the program at the end
        // cudaFree(_data);
    }

    void fillWith(const std::vector<T> &data) {
        cudaMemcpy(_data, data.data(), data.size() * sizeof(T), cudaMemcpyHostToDevice);
    }

    void copyToHost(std::vector<T> &data) const {
        cudaMemcpy(data.data(), _data, data.size() * sizeof(T), cudaMemcpyDeviceToHost);
    }
};

struct Params {
    VectorType point_0;
    uint num_faces_0;
    uint inti_done_0;
};

class GravityEvaluable : public GravityEvaluableBase {
public:
    GravityEvaluable(
            const std::vector<Array3> &Vertices,
            const std::vector<IndexArray3> &Faces,
            const double density)
        : GravityEvaluableBase(Vertices, Faces, density), mem_vertices(Vertices.size()), mem_faces(Faces.size()), mem_normals(Faces.size()), mem_segmentVectors(Faces.size() * 3), mem_segmentNormals(Faces.size() * 3), mem_results(Faces.size()), mem_settings(1) {
    }

    GravityModelResult evaluate(const Array3 &Point) override {
        if (!_initialized) init();

        wrapper_eval(
                mem_vertices._data,
                mem_faces._data,
                mem_normals._data,
                mem_segmentVectors._data,
                mem_segmentNormals._data,
                mem_results._data,
                mem_settings._data,
                _faces.size(),
                Point[0],
                Point[1],
                Point[2],
                _initialized);

        _initialized = true;

        GravityModelResult result{};

        thrust::device_ptr<Result> cptr = thrust::device_pointer_cast(mem_results._data);
        Result init{};
        Result r = thrust::reduce(cptr, cptr + _faces.size(), init);

        result.potential = r.res.w;
        result.acceleration[0] = r.res.x;
        result.acceleration[1] = r.res.y;
        result.acceleration[2] = r.res.z;

        result.gradiometricTensor.data[0] = r.first.x;
        result.gradiometricTensor.data[1] = r.first.y;
        result.gradiometricTensor.data[2] = r.first.z;

        result.gradiometricTensor.data[3] = r.second.x;
        result.gradiometricTensor.data[4] = r.second.y;
        result.gradiometricTensor.data[5] = r.second.z;

        auto &[potential, acceleration, gradiometricTensor] = result;

        // 9. Step: Compute prefix consisting of GRAVITATIONAL_CONSTANT * density
        const double prefix = GRAVITATIONAL_CONSTANT * _density;

        // 10. Step: Final expressions after application of the prefix (and a division by 2 for the potential)
        potential = (potential * prefix) / 2.0;
        acceleration = acceleration * (-1.0 * prefix);
        gradiometricTensor = gradiometricTensor * prefix;
        return result;
    }

private:
    void init() {
        std::vector<VectorType> temp_vertices(_vertices.size());
        for (uint i = 0; i < temp_vertices.size(); i++) {
            temp_vertices[i].x = _vertices[i][0];
            temp_vertices[i].y = _vertices[i][1];
            temp_vertices[i].z = _vertices[i][2];
        }
        mem_vertices.fillWith(temp_vertices);

        std::vector<uint3> temp_faces(_faces.size());
        for (uint i = 0; i < temp_faces.size(); i++) {
            temp_faces[i].x = _faces[i][0];
            temp_faces[i].y = _faces[i][1];
            temp_faces[i].z = _faces[i][2];
        }
        mem_faces.fillWith(temp_faces);
    }

    CudaMemory<VectorType> mem_vertices;
    CudaMemory<uint3> mem_faces;

    CudaMemory<VectorType> mem_normals;
    CudaMemory<VectorType> mem_segmentVectors;
    CudaMemory<VectorType> mem_segmentNormals;

    CudaMemory<Result> mem_results;

    CudaMemory<Params> mem_settings;
};

std::unique_ptr<GravityEvaluableBase> create_gravity_evaluable(
        const std::vector<Array3> &Vertices,
        const std::vector<IndexArray3> &Faces,
        double density) {
    return std::make_unique<GravityEvaluable>(Vertices, Faces, density);
}
