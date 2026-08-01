#include "polyhedralGravity/PolyhedralGravityDefinitions.h"
#include <hip/hip_runtime.h>

#include <thrust/device_vector.h>
#include <thrust/reduce.h>

#if FLOAT_BITS == 32
using VectorType = float3;
using VectorType4 = float4;
inline __host__ __device__ VectorType4 make4(FloatType x, FloatType y, FloatType z, FloatType w) {
    return make_float4(x, y, z, w);
}
inline __host__ __device__ VectorType4 make4(VectorType xyz, FloatType w) {
    return make_float4(xyz.x, xyz.y, xyz.z, w);
}
#elif FLOAT_BITS == 64
using VectorType = double3;
using VectorType4 = double4;
inline __host__ __device__ VectorType4 make4(FloatType x, FloatType y, FloatType z, FloatType w) {
    return make_double4(x, y, z, w);
}
inline __host__ __device__ VectorType4 make4(VectorType xyz, FloatType w) {
    return make_double4(xyz.x, xyz.y, xyz.z, w);
}
#else
#error "Invliad float bits size"
#endif

// HIP supplies the arithmetic operators for its vector types (float3/double3),
// but not the cross product or normalization helpers that the CUDA helper_math.h
// provided. Define the two we need here so the kernels stay backend-agnostic.
inline __host__ __device__ VectorType cross(const VectorType &a, const VectorType &b) {
    return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}

inline __host__ __device__ VectorType normalize(const VectorType &v) {
    const FloatType invLen = static_cast<FloatType>(1) / sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
    return {v.x * invLen, v.y * invLen, v.z * invLen};
}

void checkHipError(hipError_t error, const char *msg) {
    if (error != hipSuccess) {
        throw std::runtime_error(std::string(msg) + ": " + hipGetErrorString(error));
    }
}

template<typename T>
class HipMemory {
public:
    explicit HipMemory(const size_t numElements) {
        size_ = numElements;
        checkHipError(hipMalloc(&ptr_, numElements * sizeof(T)), "hipMalloc");
    }

    ~HipMemory() {
        if (ptr_) {
            hipFree(ptr_);
        }
    }

    T *get() {
        return ptr_;
    }

    void copyFromHost(const std::vector<T> &data) {
        checkHipError(hipMemcpy(ptr_, data.data(), data.size() * sizeof(T), hipMemcpyHostToDevice), "hipMemcpy Host to Device");
    }

    void copyToHost(std::vector<T> &data) const {
        if (data.size() < size_) { data.resize(size_); }
        checkHipError(hipMemcpy(data.data(), ptr_, data.size() * sizeof(T), hipMemcpyDeviceToHost), "hipMemcpy Device to Host");
    }

    void copyToHost(std::vector<T> &data, size_t N) const {
        if (data.size() < N) { data.resize(N); }
        checkHipError(hipMemcpy(data.data(), ptr_, N * sizeof(T), hipMemcpyDeviceToHost), "hipMemcpy Device to Host");
    }

private:
    T *ptr_ = nullptr;
    size_t size_;
};

__global__ void run_init(
        const VectorType *vertices,
        const int3 *faces,
        VectorType *normals,
        VectorType *segmentVectors,
        VectorType *segmentNormals,
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

    VectorType sv[3] = {
            face[1] - face[0], face[2] - face[1], face[0] - face[2]};

    VectorType n = normalize(cross(sv[0], sv[1]));
    normals[index] = n;

    segmentVectors[index * 3 + 0] = sv[0];
    segmentVectors[index * 3 + 1] = sv[1];
    segmentVectors[index * 3 + 2] = sv[2];

    segmentNormals[index * 3 + 0] = normalize(cross(sv[0], n));
    segmentNormals[index * 3 + 1] = normalize(cross(sv[1], n));
    segmentNormals[index * 3 + 2] = normalize(cross(sv[2], n));
}

inline __device__ FloatType dot_cuda(VectorType a, VectorType b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

inline __device__ VectorType to_vec(FloatType a[3]) {
    return {a[0], a[1], a[2]};
}

inline __device__ int sgn_cuda(FloatType val) {
    if (val < -EPSILON_ZERO_OFFSET) return -1;
    if (val > EPSILON_ZERO_OFFSET) return 1;
    return 0;
}

inline __device__ FloatType euclideanNormCuda(VectorType v) {
    return sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
}

__device__ void transpose_cuda(VectorType matrix[3]) {
    VectorType copy[3] = {matrix[0], matrix[1], matrix[2]};

    matrix[0].y = copy[1].x;
    matrix[0].z = copy[2].x;

    matrix[1].x = copy[0].y;
    matrix[1].z = copy[2].y;

    matrix[2].x = copy[0].z;
    matrix[2].y = copy[1].z;
}

__device__ FloatType det_cuda(VectorType matrix[3]) {
    return matrix[0].x * matrix[1].y * matrix[2].z +
           matrix[0].y * matrix[1].z * matrix[2].x +
           matrix[0].z * matrix[1].x * matrix[2].y -
           matrix[0].z * matrix[1].y * matrix[2].x -
           matrix[0].x * matrix[1].z * matrix[2].y -
           matrix[0].y * matrix[1].x * matrix[2].z;
}

__device__ FloatType det_v(VectorType a, VectorType b, VectorType c) {
    VectorType matrix[3] = {a, b, c};
    return det_cuda(matrix);
}

__device__ FloatType compute_singularities(
        int face_index,
        int segmentNormalOrientations[3],
        FloatType projectionPointVertexNorms[3],
        VectorType *segmentVectors) {
    bool allInside = true;
    for (uint index = 0; index < 3; ++index) {
        allInside &= segmentNormalOrientations[index] == 1;
    }
    if (allInside) return PI2;

    bool anyOnLine = false;
    for (uint index = 0; index < 3; ++index) {
        if (segmentNormalOrientations[index] != 0) {
            continue;
        }
        FloatType segmentVectorNorm = euclideanNormCuda(segmentVectors[face_index * 3 + index]);
        anyOnLine |= projectionPointVertexNorms[(index + 1) % 3] < segmentVectorNorm && projectionPointVertexNorms[index] < segmentVectorNorm;
    }

    if (anyOnLine) {
        return PI;
    }

    for (uint index = 0; index < 3; ++index) {
        if (segmentNormalOrientations[index] != 0) {
            continue;
        }

        FloatType r1Norm = projectionPointVertexNorms[(index + 1) % 3];
        FloatType r2Norm = projectionPointVertexNorms[index];

        if (!(r1Norm < EPSILON_ZERO_OFFSET || r2Norm < EPSILON_ZERO_OFFSET)) {
            continue;
        }

        VectorType g1 = r1Norm == 0.0 ? segmentVectors[face_index * 3 + index] : segmentVectors[face_index * 3 + (index - 1 + 3) % 3];
        VectorType g2 = r1Norm == 0.0 ? segmentVectors[face_index * 3 + (index + 1) % 3] : segmentVectors[face_index * 3 + index];

        FloatType gdot = -dot_cuda(g1, g2);
        FloatType theta = gdot == 0.0 ? PI_2 : std::acos(gdot / (euclideanNormCuda(g1) * euclideanNormCuda(g2)));
        return theta;
    }

    return 0.0;
}

struct GravityModelResultHip {
    VectorType4 acc_pot;
    VectorType first;
    VectorType second;

    __host__ __device__ GravityModelResultHip()
        : acc_pot(0), first{}, second{} {
    }

    __host__ __device__ GravityModelResultHip(const VectorType4 &_acc_pot, const VectorType &_first, const VectorType &_second)
        : acc_pot(_acc_pot), first(_first), second(_second) {
    }

    __host__ __device__ GravityModelResultHip operator+(const GravityModelResultHip &other) const {
        return {
                acc_pot + other.acc_pot,
                first + other.first,
                second + other.second};
    }
};

__global__ void run_eval(
        const VectorType *vertices,
        const int3 *faces,
        VectorType *normals,
        VectorType *segmentVectors,
        VectorType *segmentNormals,
        GravityModelResultHip *result,
        int num_faces,
        FloatType p1,
        FloatType p2,
        FloatType p3) {
    VectorType point = {p1, p2, p3};
    const int face_index = blockIdx.x * blockDim.x + threadIdx.x;

    if (face_index >= num_faces) {
        return;
    }

    VectorType face[3] = {
            vertices[faces[face_index].x] - point,
            vertices[faces[face_index].y] - point,
            vertices[faces[face_index].z] - point,
    };

    int planeNormalOrientation = sgn_cuda(dot_cuda(normals[face_index], face[0]));

    VectorType4 hessianPlane;
    {
        VectorType origin = {0.0, 0.0, 0.0};
        VectorType crossProduct = cross(face[0] - face[1], face[0] - face[2]);
        VectorType res = (origin - face[0]) * crossProduct;
        FloatType d = res.x + res.y + res.z;

        hessianPlane.x = crossProduct.x;
        hessianPlane.y = crossProduct.y;
        hessianPlane.z = crossProduct.z;
        hessianPlane.w = d;
    }

    FloatType planeDistance = std::abs(hessianPlane.w / sqrt(hessianPlane.x * hessianPlane.x + hessianPlane.y * hessianPlane.y + hessianPlane.z * hessianPlane.z));

    VectorType tmp1 = normals[face_index] * planeDistance;
    FloatType orthogonalProjectionPointOnPlane[3] = {tmp1.x, tmp1.y, tmp1.z};
    {
        FloatType intersections[3] = {
                hessianPlane.x == static_cast<FloatType>(0.0) ? static_cast<FloatType>(0.0) : hessianPlane.w / hessianPlane.x,
                hessianPlane.y == static_cast<FloatType>(0.0) ? static_cast<FloatType>(0.0) : hessianPlane.w / hessianPlane.y,
                hessianPlane.z == static_cast<FloatType>(0.0) ? static_cast<FloatType>(0.0) : hessianPlane.w / hessianPlane.z};

        for (unsigned int index = 0; index < 3; ++index) {
            if (intersections[index] < 0) {
                orthogonalProjectionPointOnPlane[index] = std::abs(orthogonalProjectionPointOnPlane[index]);
            } else {
                if (orthogonalProjectionPointOnPlane[index] > 0) {
                    orthogonalProjectionPointOnPlane[index] = -orthogonalProjectionPointOnPlane[index];
                } else {
                    orthogonalProjectionPointOnPlane[index] = orthogonalProjectionPointOnPlane[index];
                }
            }
        }
    }


    int segmentNormalOrientations[3];
    for (unsigned int index = 0; index < 3; ++index) {
        FloatType inner = dot_cuda(segmentNormals[face_index * 3 + index], to_vec(orthogonalProjectionPointOnPlane) - face[index]);
        segmentNormalOrientations[index] = -sgn_cuda(inner);
    }

    VectorType orthogonalProjectionPointsOnSegmentsForPlane[3];
    for (unsigned int index = 0; index < 3; ++index) {
        if (segmentNormalOrientations[index] == 0) {
            orthogonalProjectionPointsOnSegmentsForPlane[index] = to_vec(orthogonalProjectionPointOnPlane);
        } else {
            VectorType vertex1 = face[index];
            VectorType vertex2 = face[(index + 1) % 3];

            VectorType matrixRow1 = vertex2 - vertex1;
            VectorType matrixRow2 = cross(vertex1 - to_vec(orthogonalProjectionPointOnPlane), matrixRow1);
            VectorType matrixRow3 = cross(matrixRow2, matrixRow1);

            VectorType d = {
                    dot_cuda(matrixRow1, to_vec(orthogonalProjectionPointOnPlane)),
                    dot_cuda(matrixRow2, to_vec(orthogonalProjectionPointOnPlane)),
                    dot_cuda(matrixRow3, vertex1)};

            VectorType columnMatrix[3] = {
                    matrixRow1,
                    matrixRow2,
                    matrixRow3};
            transpose_cuda(columnMatrix);

            FloatType determinant = det_cuda(columnMatrix);

            if (determinant != 0.0) {
                VectorType r = {
                        det_v(d, columnMatrix[1], columnMatrix[2]),
                        det_v(columnMatrix[0], d, columnMatrix[2]),
                        det_v(columnMatrix[0], columnMatrix[1], d),
                };
                orthogonalProjectionPointsOnSegmentsForPlane[index] = r / determinant;
            }
        }
    }

    FloatType segmentDistances[3];
    for (unsigned int index = 0; index < 3; ++index) {
        segmentDistances[index] = euclideanNormCuda(orthogonalProjectionPointsOnSegmentsForPlane[index] - to_vec(orthogonalProjectionPointOnPlane));
    }


    Distance distances[3];
    for (unsigned int index = 0; index < 3; ++index) {
        distances[index].l1 = euclideanNormCuda(face[index]);
        distances[index].l2 = euclideanNormCuda(face[(index + 1) % 3]);

        distances[index].s1 = euclideanNormCuda(orthogonalProjectionPointsOnSegmentsForPlane[index] - face[index]);
        distances[index].s2 = euclideanNormCuda(orthogonalProjectionPointsOnSegmentsForPlane[index] - face[(index + 1) % 3]);

        if (abs(distances[index].s1 - distances[index].l1) < EPSILON_ZERO_OFFSET && abs(distances[index].s2 - distances[index].l2) < EPSILON_ZERO_OFFSET) {
            if (distances[index].s2 < distances[index].s1) {
                distances[index].s1 *= -1.0;
                distances[index].s2 *= -1.0;
                distances[index].l1 *= -1.0;
                distances[index].l2 *= -1.0;
            } else if (std::abs(distances[index].s2 - distances[index].s1) < EPSILON_ZERO_OFFSET) {
                distances[index].s1 *= -1.0;
                distances[index].l1 *= -1.0;
            }
        } else {
            FloatType norm = euclideanNormCuda(segmentVectors[face_index * 3 + index]);
            if (distances[index].s1 < norm && distances[index].s2 < norm) {
                distances[index].s1 *= -1.0;
            } else if (distances[index].s2 < distances[index].s1) {
                distances[index].s1 *= -1.0;
                distances[index].s2 *= -1.0;
            }
        }
    }


    FloatType projectionPointVertexNorms[3] = {
            euclideanNormCuda(to_vec(orthogonalProjectionPointOnPlane) - face[0]),
            euclideanNormCuda(to_vec(orthogonalProjectionPointOnPlane) - face[1]),
            euclideanNormCuda(to_vec(orthogonalProjectionPointOnPlane) - face[2]),
    };

    TranscendentalExpression transcendentalExpressions[3];
    for (unsigned int index = 0; index < 3; ++index) {
        FloatType r1Norm = projectionPointVertexNorms[(index + 1) % 3];
        FloatType r2Norm = projectionPointVertexNorms[index];

        if ((segmentNormalOrientations[index] == 0 && (r1Norm < EPSILON_ZERO_OFFSET || r2Norm < EPSILON_ZERO_OFFSET)) ||
            (std::abs(distances[index].s1 + distances[index].s2) < EPSILON_ZERO_OFFSET &&
             std::abs(distances[index].l1 + distances[index].l2) < EPSILON_ZERO_OFFSET)) {
            transcendentalExpressions[index].ln = 0.0;
        } else {
            FloatType inner_num = distances[index].s2 + distances[index].l2;
            FloatType inner_denom = distances[index].s1 + distances[index].l1;

            if (inner_num <= 0.0 || inner_denom <= 0.0) {
                transcendentalExpressions[index].ln = 0.0;
            } else {
                transcendentalExpressions[index].ln = log(inner_num / inner_denom);
            }
        }
        if (planeDistance < EPSILON_ZERO_OFFSET || segmentDistances[index] < EPSILON_ZERO_OFFSET) {
            transcendentalExpressions[index].an = 0.0;
        } else {
            FloatType frac1 = (planeDistance * distances[index].s2) / (segmentDistances[index] * distances[index].l2);
            FloatType frac2 = (planeDistance * distances[index].s1) / (segmentDistances[index] * distances[index].l1);

            transcendentalExpressions[index].an = std::atan(frac1) - std::atan(frac2);
        }
    }

    FloatType sing_theta = compute_singularities(face_index, segmentNormalOrientations, projectionPointVertexNorms, segmentVectors);
    FloatType sing_alpha = -planeDistance * sing_theta;
    VectorType sing_beta = normals[face_index] * (-sing_theta * planeNormalOrientation);

    FloatType sum1PotentialAcceleration = 0.0;
    for (unsigned int index = 0; index < 3; ++index)
        sum1PotentialAcceleration += segmentNormalOrientations[index] * segmentDistances[index] * transcendentalExpressions[index].ln;

    VectorType sum1Tensor = {0.0, 0.0, 0.0};
    for (unsigned int index = 0; index < 3; ++index)
        sum1Tensor = sum1Tensor + segmentNormals[face_index * 3 + index] * transcendentalExpressions[index].ln;

    FloatType sum2 = 0.0;
    for (unsigned int index = 0; index < 3; ++index)
        sum2 += segmentNormalOrientations[index] * transcendentalExpressions[index].an;

    FloatType planeSumPotentialAcceleration = sum1PotentialAcceleration + planeDistance * sum2 + sing_alpha;
    VectorType subSum = (sum1Tensor + (normals[face_index] * (planeNormalOrientation * sum2))) + sing_beta;
    VectorType first = normals[face_index] * subSum;

    VectorType reorderedNp = {normals[face_index].x, normals[face_index].x, normals[face_index].y};
    VectorType reorderedSubSum = {subSum.y, subSum.z, subSum.z};
    VectorType second = reorderedNp * reorderedSubSum;

    auto potential = planeNormalOrientation * planeDistance * planeSumPotentialAcceleration;
    auto accel = normals[face_index] * planeSumPotentialAcceleration;
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
        : GravityEvaluableBase(Vertices, Faces, density), d_vertices(Vertices.size()), d_faces(Faces.size()), d_normals(Faces.size()), d_segmentVectors(Faces.size() * 3), d_segmentNormals(Faces.size() * 3), d_results(Faces.size()) {
    }

    GravityModelResult evaluate(const Array3 &Point) override {
        if (!_initialized) init();

        int num_faces = _faces.size();
        int blockSize = 64;
        int numBlocks = (num_faces + blockSize - 1) / blockSize;
        numBlocks = numBlocks > 0 ? numBlocks : 1;

        std::vector<FloatType> resultPotential(num_faces);
        std::vector<VectorType> resultAcceleration(num_faces);

        run_eval<<<numBlocks, blockSize>>>(
                d_vertices.get(), d_faces.get(), d_normals.get(), d_segmentVectors.get(), d_segmentNormals.get(), d_results.get(), num_faces, Point[0], Point[1], Point[2]);

        checkHipError(hipGetLastError(), "Kernel eval failed");

        thrust::device_ptr<GravityModelResultHip> cptr = thrust::device_pointer_cast(d_results.get());
        GravityModelResultHip init{};
        GravityModelResultHip r = thrust::reduce(cptr, cptr + _faces.size(), init);

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

        run_init<<<numBlocks, blockSize>>>(d_vertices.get(), d_faces.get(), d_normals.get(), d_segmentVectors.get(), d_segmentNormals.get(), num_faces);
        checkHipError(hipGetLastError(), "Kernel init failed");
    }

    HipMemory<VectorType> d_vertices;
    HipMemory<int3> d_faces;
    HipMemory<VectorType> d_normals;
    HipMemory<VectorType> d_segmentVectors;
    HipMemory<VectorType> d_segmentNormals;

    HipMemory<GravityModelResultHip> d_results;
};

std::unique_ptr<GravityEvaluableBase> create_gravity_evaluable(
        const std::vector<Array3> &Vertices,
        const std::vector<IndexArray3> &Faces,
        double density) {
    return std::make_unique<GravityEvaluable>(Vertices, Faces, density);
}
