#include "common/Marker.h"
#include "polyhedralGravity/PolyhedralGravityDefinitions.h"
#include <cuda_runtime.h>
#include <vector_types.h>

#include <thrust/device_vector.h>
#include <thrust/reduce.h>

#if FLOAT_BITS == 32
using VectorType = float3;
using VectorType4 = float4;
#include "common/cuda/helper_math.h"
inline __device__ VectorType4 make4(FloatType x, FloatType y, FloatType z, FloatType w) {
    return make_float4(x, y, z, w);
}
inline __device__ VectorType4 make4(VectorType xyz, FloatType w) {
    return make_float4(xyz.x, xyz.y, xyz.z, w);
}
#elif FLOAT_BITS == 64
using VectorType = double3;
using VectorType4 = double4;
inline __device__ VectorType4 make4(FloatType x, FloatType y, FloatType z, FloatType w) {
    return make_double4(x, y, z, w);
}
inline __device__ VectorType4 make4(VectorType xyz, FloatType w) {
    return make_double4(xyz.x, xyz.y, xyz.z, w);
}
#include "common/cuda/helper_math_double.h"
#else
#error "Invliad float bits size"
#endif

void checkCudaError(cudaError_t error, const char *msg) {
    if (error != cudaSuccess) {
        throw std::runtime_error(std::string(msg) + ": " + cudaGetErrorString(error));
    }
}

template<typename T>
class CudaMemory {
public:
    explicit CudaMemory(const size_t numElements) {
        size_ = numElements;
        checkCudaError(cudaMalloc(&ptr_, numElements * sizeof(T)), "cudaMalloc");
    }

    ~CudaMemory() {
        if (ptr_) {
            cudaFree(ptr_);
        }
    }

    T *get() {
        return ptr_;
    }

    void copyFromHost(const std::vector<T> &data) {
        checkCudaError(cudaMemcpy(ptr_, data.data(), data.size() * sizeof(T), cudaMemcpyHostToDevice), "cudaMemcpy Host to Device");
    }

    void copyToHost(std::vector<T> &data) const {
        if (data.size() < size_) { data.resize(size_); }
        checkCudaError(cudaMemcpy(data.data(), ptr_, data.size() * sizeof(T), cudaMemcpyDeviceToHost), "cudaMemcpy Device to Host");
    }

    void copyToHost(std::vector<T> &data, size_t N) const {
        if (data.size() < N) { data.resize(N); }
        checkCudaError(cudaMemcpy(data.data(), ptr_, N * sizeof(T), cudaMemcpyDeviceToHost), "cudaMemcpy Device to Host");
    }

private:
    T *ptr_ = nullptr;
    size_t size_;
};

__global__ void run_init(
        const VectorType *vertices,
        const int3 *faces,
        VectorType *normals,
        int num_faces) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= num_faces) {
        return;
    }

    VectorType face[3] = {
            vertices[faces[index].x],
            vertices[faces[index].y],
            vertices[faces[index].z],
    };

    // 1-02 Step: The plane unit normals N_p are the only per-face property that is cached
    normals[index] = normalize(cross(face[1] - face[0], face[2] - face[1]));
}

inline __device__ FloatType dot_cuda(VectorType a, VectorType b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

inline __device__ FloatType sgn_cuda(FloatType val) {
    if (val < -EPSILON_ZERO) return -1;
    if (val > EPSILON_ZERO) return 1;
    return 0;
}

inline __device__ FloatType euclideanNormCuda(VectorType v) {
    return sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
}

struct GravityModelResultCuda {
    VectorType4 acc_pot;
    VectorType first;
    VectorType second;

    __device__ GravityModelResultCuda operator+(const GravityModelResultCuda &other) const {
        return {
                acc_pot + other.acc_pot,
                first + other.first,
                second + other.second};
    }
};

// At most 256 threads per block, and a body small enough for three such blocks per SM, so that the register
// allocator keeps the kernel's occupancy up
__global__ __launch_bounds__(256, 3) void run_eval(
        const VectorType *vertices,
        const int3 *faces,
        const VectorType *normals,
        GravityModelResultCuda *result,
        int num_faces,
        FloatType p1,
        FloatType p2,
        FloatType p3) {
    VectorType point = {p1, p2, p3};
    const int face_index = blockIdx.x * blockDim.x + threadIdx.x;

    if (face_index >= num_faces) {
        return;
    }

    const VectorType face[3] = {
            vertices[faces[face_index].x] - point,
            vertices[faces[face_index].y] - point,
            vertices[faces[face_index].z] - point,
    };
    const VectorType planeUnitNormal = normals[face_index];

    // 1-01 Step: Compute the segment vectors G_pq
    // Recomputed rather than cached: three subtractions are cheaper than reading 36 more bytes per face
    const VectorType segmentVectors[3] = {face[1] - face[0], face[2] - face[1], face[0] - face[2]};

    // 1-04 to 1-07 Step: Compute sigma_p, h_p and P' from the projection of the face onto N_p
    // N_p is a unit vector, so N_p * v_0 is the signed distance of P to the plane and P' is N_p scaled by it
    const FloatType planeProjection = dot_cuda(planeUnitNormal, face[0]);
    const FloatType planeNormalOrientation = sgn_cuda(planeProjection);
    const FloatType planeDistance = std::abs(planeProjection);
    const VectorType orthogonalProjectionPointOnPlane = planeUnitNormal * planeProjection;

    // 1-08 to 1-12 Step: Compute sigma_pq, h_pq, the distances l1, l2, s1, s2 and the norms of P' - v_q
    // All of them follow from projecting P' - v_q onto n_pq and onto G_pq, so P'' is never formed
    const FloatType vertexNorms[3] = {euclideanNormCuda(face[0]), euclideanNormCuda(face[1]), euclideanNormCuda(face[2])};
    VectorType segmentUnitNormals[3];
    FloatType segmentNormalOrientations[3];
    FloatType segmentDistances[3];
    FloatType projectionPointVertexNorms[3];
    Distance distances[3];
    for (unsigned int index = 0; index < 3; ++index) {
        const VectorType relativeProjectionPoint = orthogonalProjectionPointOnPlane - face[index];
        projectionPointVertexNorms[index] = euclideanNormCuda(relativeProjectionPoint);

        // n_pq is normalized by |G_pq|, which is |G_pq x N_p| since N_p is a unit vector perpendicular to G_pq
        const FloatType squaredSegmentNorm = dot_cuda(segmentVectors[index], segmentVectors[index]);
        const FloatType inverseSegmentNorm = rsqrt(squaredSegmentNorm);
        const FloatType segmentNorm = squaredSegmentNorm * inverseSegmentNorm;
        segmentUnitNormals[index] = cross(segmentVectors[index], planeUnitNormal) * inverseSegmentNorm;

        // The projection onto n_pq has the sign -sigma_pq and the magnitude h_pq
        const FloatType normalProjection = dot_cuda(segmentUnitNormals[index], relativeProjectionPoint);
        segmentNormalOrientations[index] = -sgn_cuda(normalProjection);
        segmentDistances[index] = std::abs(normalProjection);

        // The projection onto G_pq is the signed position u of P'' along the segment, from its first endpoint
        const FloatType alongSegment = dot_cuda(relativeProjectionPoint, segmentVectors[index]) * inverseSegmentNorm;
        Distance &distance = distances[index];
        distance.l1 = vertexNorms[index];
        distance.l2 = vertexNorms[(index + 1) % 3];
        distance.s1 = std::abs(alongSegment);
        distance.s2 = std::abs(alongSegment - segmentNorm);

        // The 1., 2. and 3. Option of Tsoulis (2021) all amount to s1 = -u and s2 = |G_pq| - u
        if (std::abs(distance.s1 - distance.l1) >= EPSILON_ZERO || std::abs(distance.s2 - distance.l2) >= EPSILON_ZERO) {
            distance.s1 = -alongSegment;
            distance.s2 = segmentNorm - alongSegment;
        } else if (distance.s2 < distance.s1) {
            distance.s1 = -distance.s1;
            distance.s2 = -distance.s2;
            distance.l1 = -distance.l1;
            distance.l2 = -distance.l2;
        } else if (std::abs(distance.s2 - distance.s1) < EPSILON_ZERO) {
            distance.s1 = -distance.s1;
            distance.l1 = -distance.l1;
        }
    }

    // 1-13 Step: Compute the transcendental expressions LN_pq and AN_pq
    // Both are evaluated unconditionally and then selected: their guards only hold for degenerate positions of P'
    TranscendentalExpression transcendentalExpressions[3];
    for (unsigned int index = 0; index < 3; ++index) {
        const Distance &distance = distances[index];
        const FloatType r1Norm = projectionPointVertexNorms[(index + 1) % 3];
        const FloatType r2Norm = projectionPointVertexNorms[index];

        const bool logarithmVanishes =
                (segmentNormalOrientations[index] == 0 && (r1Norm < EPSILON_ZERO || r2Norm < EPSILON_ZERO)) ||
                (std::abs(distance.s1 + distance.s2) < EPSILON_ZERO && std::abs(distance.l1 + distance.l2) < EPSILON_ZERO);
        const FloatType logarithm = log((distance.s2 + distance.l2) / (distance.s1 + distance.l1));
        transcendentalExpressions[index].ln = logarithmVanishes ? 0 : logarithm;

        // atan(x) - atan(y) = atan((x - y) / (1 + xy)), which misses a whole PI (with the sign of x) if 1 + xy < 0
        const bool arcTangentVanishes = planeDistance < EPSILON_ZERO || segmentDistances[index] < EPSILON_ZERO;
        const FloatType upper = (planeDistance * distance.s2) / (segmentDistances[index] * distance.l2);
        const FloatType lower = (planeDistance * distance.s1) / (segmentDistances[index] * distance.l1);
        const FloatType denominator = 1 + upper * lower;
        const FloatType branchOffset = denominator < 0 ? (upper < 0 ? -PI : PI) : 0;
        const FloatType arcTangent = std::atan((upper - lower) / denominator) + branchOffset;
        transcendentalExpressions[index].an = arcTangentVanishes ? 0 : arcTangent;
    }

    // 1-14 Step: Compute the singularities sing A = factor * h_p and sing B = factor * sigma_p * N_p
    const bool allInside = segmentNormalOrientations[0] == 1 && segmentNormalOrientations[1] == 1 && segmentNormalOrientations[2] == 1;
    bool anyOnLine = false;
    bool anyAtVertex = false;
    unsigned int vertexSegment = 0;
    bool vertexIsSegmentEnd = false;
    for (unsigned int index = 0; index < 3; ++index) {
        if (segmentNormalOrientations[index] != 0) {
            continue;
        }
        const FloatType r1Norm = projectionPointVertexNorms[(index + 1) % 3];
        const FloatType r2Norm = projectionPointVertexNorms[index];
        const FloatType squaredSegmentNorm = dot_cuda(segmentVectors[index], segmentVectors[index]);
        const bool atVertex = r1Norm < EPSILON_ZERO || r2Norm < EPSILON_ZERO;
        anyOnLine |= !atVertex && r1Norm * r1Norm < squaredSegmentNorm && r2Norm * r2Norm < squaredSegmentNorm;
        vertexSegment = anyAtVertex ? vertexSegment : index;
        vertexIsSegmentEnd = anyAtVertex ? vertexIsSegmentEnd : r1Norm < EPSILON_ZERO;
        anyAtVertex |= atVertex;
    }
    FloatType vertexAngle = 0;
    if (anyAtVertex) {
        const VectorType g1 = vertexIsSegmentEnd ? segmentVectors[vertexSegment] : segmentVectors[(vertexSegment + 2) % 3];
        const VectorType g2 = vertexIsSegmentEnd ? segmentVectors[(vertexSegment + 1) % 3] : segmentVectors[vertexSegment];
        const FloatType gdot = -dot_cuda(g1, g2);
        vertexAngle = gdot == 0 ? PI_2 : std::acos(gdot / (euclideanNormCuda(g1) * euclideanNormCuda(g2)));
    }
    const FloatType singularityFactor = allInside ? -PI2 : anyOnLine ? -PI : -vertexAngle;
    const FloatType sing_alpha = singularityFactor * planeDistance;
    const VectorType sing_beta = planeUnitNormal * (singularityFactor * planeNormalOrientation);

    FloatType sum1PotentialAcceleration = 0;
    for (unsigned int index = 0; index < 3; ++index)
        sum1PotentialAcceleration += segmentNormalOrientations[index] * segmentDistances[index] * transcendentalExpressions[index].ln;

    VectorType sum1Tensor = {0, 0, 0};
    for (unsigned int index = 0; index < 3; ++index)
        sum1Tensor = sum1Tensor + segmentUnitNormals[index] * transcendentalExpressions[index].ln;

    FloatType sum2 = 0;
    for (unsigned int index = 0; index < 3; ++index)
        sum2 += segmentNormalOrientations[index] * transcendentalExpressions[index].an;

    FloatType planeSumPotentialAcceleration = sum1PotentialAcceleration + planeDistance * sum2 + sing_alpha;
    VectorType subSum = (sum1Tensor + (planeUnitNormal * (planeNormalOrientation * sum2))) + sing_beta;
    VectorType first = planeUnitNormal * subSum;

    VectorType reorderedNp = {planeUnitNormal.x, planeUnitNormal.x, planeUnitNormal.y};
    VectorType reorderedSubSum = {subSum.y, subSum.z, subSum.z};
    VectorType second = reorderedNp * reorderedSubSum;

    auto potential = planeNormalOrientation * planeDistance * planeSumPotentialAcceleration;
    auto accel = planeUnitNormal * planeSumPotentialAcceleration;
    result[face_index].acc_pot = make4(accel, potential);
    result[face_index].first = first;
    result[face_index].second = second;
}

GlobalResources::GlobalResources(int &argc, char *argv[]) {
}
GlobalResources::~GlobalResources() = default;

// https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf

class GravityEvaluable : public GravityEvaluableBase {
public:
    GravityEvaluable(
            const std::vector<Array3> &Vertices,
            const std::vector<IndexArray3> &Faces,
            const double density)
        : GravityEvaluableBase(Vertices, Faces, density), d_vertices(Vertices.size()), d_faces(Faces.size()), d_normals(Faces.size()), d_results(Faces.size()) {
    }

    GravityModelResult evaluate(const Array3 &Point) override {
        if (!_initialized) init();
        PPB_MARKER_GPU_SCOPE("evaluate");

        int num_faces = _faces.size();
        int blockSize = 256;
        int numBlocks = (num_faces + blockSize - 1) / blockSize;
        numBlocks = numBlocks > 0 ? numBlocks : 1;

        run_eval<<<numBlocks, blockSize>>>(
                d_vertices.get(), d_faces.get(), d_normals.get(), d_results.get(), num_faces, Point[0], Point[1], Point[2]);

        checkCudaError(cudaGetLastError(), "Kernel eval failed");

        thrust::device_ptr<GravityModelResultCuda> cptr = thrust::device_pointer_cast(d_results.get());
        GravityModelResultCuda init{};
        GravityModelResultCuda r = thrust::reduce(cptr, cptr + _faces.size(), init);

        GravityModelResult result{};
        result.potential = r.acc_pot.w;
        result.acceleration[0] = r.acc_pot.x;
        result.acceleration[1] = r.acc_pot.y;
        result.acceleration[2] = r.acc_pot.z;
        result.gradiometricTensor.data[0] = r.first.x;
        result.gradiometricTensor.data[1] = r.first.y;
        result.gradiometricTensor.data[2] = r.first.z;
        result.gradiometricTensor.data[3] = r.second.x;
        result.gradiometricTensor.data[4] = r.second.y;
        result.gradiometricTensor.data[5] = r.second.z;

        const double prefix = GRAVITATIONAL_CONSTANT * _density;
        result.potential = (result.potential * prefix) / 2.0;
        result.acceleration = result.acceleration * (-1.0 * prefix);
        result.gradiometricTensor = result.gradiometricTensor * prefix;

        return result;
    }

private:
    void init() {
        PPB_MARKER_GPU_SCOPE("init");
        std::vector<VectorType> tmp_vertices(_vertices.size());
        for (size_t i = 0; i < _vertices.size(); ++i) {
            auto v = _vertices[i];
            tmp_vertices[i].x = v[0];
            tmp_vertices[i].y = v[1];
            tmp_vertices[i].z = v[2];
        }
        d_vertices.copyFromHost(tmp_vertices);

        std::vector<int3> tmp_faces(_faces.size());
        for (size_t i = 0; i < _faces.size(); ++i) {
            auto f = _faces[i];
            tmp_faces[i].x = f[0];
            tmp_faces[i].y = f[1];
            tmp_faces[i].z = f[2];
        }
        d_faces.copyFromHost(tmp_faces);

        int num_faces = _faces.size();

        int blockSize = 256;
        int numBlocks = (num_faces + blockSize - 1) / blockSize;
        numBlocks = numBlocks > 0 ? numBlocks : 1;

        run_init<<<numBlocks, blockSize>>>(d_vertices.get(), d_faces.get(), d_normals.get(), num_faces);
        checkCudaError(cudaGetLastError(), "Kernel init failed");
    }

    CudaMemory<VectorType> d_vertices;
    CudaMemory<int3> d_faces;
    CudaMemory<VectorType> d_normals;

    CudaMemory<GravityModelResultCuda> d_results;
};

std::unique_ptr<GravityEvaluableBase> create_gravity_evaluable(
        const std::vector<Array3> &Vertices,
        const std::vector<IndexArray3> &Faces,
        double density) {
    return std::make_unique<GravityEvaluable>(Vertices, Faces, density);
}
