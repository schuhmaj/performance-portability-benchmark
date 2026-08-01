#include "polyhedralGravity/PolyhedralGravityDefinitions.h"

#include <alpaka/alpaka.hpp>
#include "alpaka/example/ExampleDefaultAcc.hpp"

#include <vector>

namespace {

/** Dimensionality of the problem - the faces are processed in a 1D grid. */
using Dim = alpaka::DimInt<1u>;
/** The integer type used for indexing and sizes. */
using Idx = std::size_t;
/** Compute backend - the first enabled one (CUDA/HIP have precedence over CPU). */
using Acc = alpaka::ExampleDefaultAcc<Dim, Idx>;
/** Serial CPU host backend. */
using Host = alpaka::DevCpu;
using PlatformAcc = alpaka::Platform<Acc>;
using PlatformHost = alpaka::PlatformCpu;
using DeviceAcc = alpaka::Dev<PlatformAcc>;
/** Blocking compute pipeline for the accelerator device. */
using Queue = alpaka::Queue<DeviceAcc, alpaka::Blocking>;

template<typename T>
using BufAcc = alpaka::Buf<DeviceAcc, T, Dim, Idx>;

inline alpaka::Vec<Dim, Idx> extentOf(const Idx n) {
    return alpaka::Vec<Dim, Idx>{n};
}

/**
 * Precomputes the per-face geometry (normals, segment vectors and segment normals)
 * that is independent of the evaluation point. Mirrors the OpenCL/Kokkos init kernel.
 */
struct InitKernel {
    template<typename TAcc>
    ALPAKA_FN_ACC void operator()(
            const TAcc &acc,
            const Array3 *vertices,
            const IndexArray3 *faces,
            Array3 *normals,
            Array3Triplet *segmentVectors,
            Array3Triplet *segmentNormals,
            const Idx num_faces) const {
        const Idx i = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0];
        if (i >= num_faces) {
            return;
        }

        const Array3Triplet face = {
                vertices[faces[i][0]],
                vertices[faces[i][1]],
                vertices[faces[i][2]]};

        const Array3Triplet sv = {
                face[1] - face[0], face[2] - face[1], face[0] - face[2]};

        const Array3 n = normal(sv[0], sv[1]);

        segmentVectors[i] = sv;
        normals[i] = n;
        segmentNormals[i] = {
                normal(sv[0], n),
                normal(sv[1], n),
                normal(sv[2], n)};
    }
};

/**
 * Evaluates the polyhedral gravity model for a single face with respect to the
 * computation point and stores the (not yet reduced) per-face contribution.
 * The math is a verbatim port of the Kokkos kernel using raw device pointers.
 */
struct EvalKernel {
    template<typename TAcc>
    ALPAKA_FN_ACC void operator()(
            const TAcc &acc,
            const Array3 *vertices,
            const IndexArray3 *faces,
            const Array3 *normals,
            const Array3Triplet *segmentVectors,
            const Array3Triplet *segmentNormals,
            GravityModelResult *results,
            const Idx num_faces,
            const FloatType p1,
            const FloatType p2,
            const FloatType p3) const {
        const Idx i = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0];
        if (i >= num_faces) {
            return;
        }

        const Array3 point{p1, p2, p3};

        const Array3Triplet face = {
                vertices[faces[i][0]] - point,
                vertices[faces[i][1]] - point,
                vertices[faces[i][2]] - point};

        //region 1-04 Step: Compute Plane Normal Orientation sigma_p
        const int planeNormalOrientation = sgn(dot(normals[i], face[0]));
        //endregion

        //region 1-05 Step: Compute Hessian Normal Plane Representation
        HessianPlane hessianPlane{};
        {
            constexpr Array3 origin{0.0, 0.0, 0.0};
            const auto crossProduct = cross(face[0] - face[1], face[0] - face[2]);
            const auto res = crossProduct * (origin - face[0]);
            const auto d = res[0] + res[1] + res[2];

            hessianPlane = {crossProduct[0], crossProduct[1], crossProduct[2], d};
        }
        //endregion

        //region 1-06 Step: Compute distance h_p between P and P'
        const auto planeDistance = std::abs(hessianPlane.d / std::sqrt(
                hessianPlane.a * hessianPlane.a + hessianPlane.b * hessianPlane.b + hessianPlane.c * hessianPlane.c));
        //endregion

        //region 1-07 Step: Compute the actual position of P' (projection of P on the plane)
        Array3 orthogonalProjectionPointOnPlane = normals[i] * planeDistance;
        {
            const Array3 intersections = {
                    hessianPlane.a == 0.0 ? static_cast<FloatType>(0.0) : hessianPlane.d / hessianPlane.a,
                    hessianPlane.b == 0.0 ? static_cast<FloatType>(0.0) : hessianPlane.d / hessianPlane.b,
                    hessianPlane.c == 0.0 ? static_cast<FloatType>(0.0) : hessianPlane.d / hessianPlane.c};

            for (unsigned int index = 0; index < 3; ++index) {
                if (intersections[index] < 0) {
                    orthogonalProjectionPointOnPlane[index] = std::abs(orthogonalProjectionPointOnPlane[index]);
                } else {
                    if (orthogonalProjectionPointOnPlane[index] > 0) {
                        orthogonalProjectionPointOnPlane[index] = -1.0 * orthogonalProjectionPointOnPlane[index];
                    } else {
                        orthogonalProjectionPointOnPlane[index] = orthogonalProjectionPointOnPlane[index];
                    }
                }
            }
        }
        //endregion

        //region 1-08 Step: Compute the segment normal orientation sigma_pq
        int segmentNormalOrientations[3];
        for (unsigned int index = 0; index < 3; ++index) {
            segmentNormalOrientations[index] = -sgn(dot(segmentNormals[i][index], orthogonalProjectionPointOnPlane - face[index]));
        }
        //endregion

        //region 1-09 Step: Compute the orthogonal projection point P'' of P' on each segment
        Array3 orthogonalProjectionPointsOnSegmentsForPlane[3];
        for (unsigned int index = 0; index < 3; ++index) {
            if (segmentNormalOrientations[index] == 0) {
                orthogonalProjectionPointsOnSegmentsForPlane[index] = orthogonalProjectionPointOnPlane;
            } else {
                const auto &vertex1 = face[index];
                const auto &vertex2 = face[(index + 1) % 3];

                const Array3 matrixRow1 = vertex2 - vertex1;
                const Array3 matrixRow2 = cross(vertex1 - orthogonalProjectionPointOnPlane, matrixRow1);
                const Array3 matrixRow3 = cross(matrixRow2, matrixRow1);
                const Array3 d = {dot(matrixRow1, orthogonalProjectionPointOnPlane),
                                  dot(matrixRow2, orthogonalProjectionPointOnPlane), dot(matrixRow3, vertex1)};
                const Matrix columnMatrix = transpose({matrixRow1, matrixRow2, matrixRow3});

                const auto determinant = det(columnMatrix);
                if (determinant != 0.0) {
                    orthogonalProjectionPointsOnSegmentsForPlane[index] =
                            Array3{det(Matrix{d, columnMatrix[1], columnMatrix[2]}),
                                   det(Matrix{columnMatrix[0], d, columnMatrix[2]}),
                                   det(Matrix{columnMatrix[0], columnMatrix[1], d})} /
                            determinant;
                }
            }
        }
        //endregion

        //region 1-10 Step: Compute the segment distances h_pq between P'' and P'
        Array3 segmentDistances{};
        for (unsigned int index = 0; index < 3; ++index) {
            segmentDistances[index] = euclideanNorm(orthogonalProjectionPointsOnSegmentsForPlane[index] - orthogonalProjectionPointOnPlane);
        }
        //endregion

        //region 1-11 Step: Compute the 3D distances l1, l2 and 1D distances s1, s2
        Distance distances[3];
        for (unsigned int index = 0; index < 3; ++index) {
            distances[index].l1 = euclideanNorm(face[index]);
            distances[index].l2 = euclideanNorm(face[(index + 1) % 3]);

            distances[index].s1 = euclideanNorm(orthogonalProjectionPointsOnSegmentsForPlane[index] - face[index]);
            distances[index].s2 = euclideanNorm(orthogonalProjectionPointsOnSegmentsForPlane[index] - face[(index + 1) % 3]);

            if (std::abs(distances[index].s1 - distances[index].l1) < EPSILON_ZERO_OFFSET &&
                std::abs(distances[index].s2 - distances[index].l2) < EPSILON_ZERO_OFFSET) {
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
                const auto norm = euclideanNorm(segmentVectors[i][index]);
                if (distances[index].s1 < norm && distances[index].s2 < norm) {
                    distances[index].s1 *= -1.0;
                } else if (distances[index].s2 < distances[index].s1) {
                    distances[index].s1 *= -1.0;
                    distances[index].s2 *= -1.0;
                }
            }
        }
        //endregion

        //region 1-12 Step: Compute the euclidian norms of the vectors of P' and the vertices
        const Array3 projectionPointVertexNorms{
                euclideanNorm(orthogonalProjectionPointOnPlane - face[0]),
                euclideanNorm(orthogonalProjectionPointOnPlane - face[1]),
                euclideanNorm(orthogonalProjectionPointOnPlane - face[2])};
        //endregion

        //region 1-13 Step: Compute the transcendental expressions LN_pq and AN_pq
        TranscendentalExpression transcendentalExpressions[3];
        for (unsigned int index = 0; index < 3; ++index) {
            const auto r1Norm = projectionPointVertexNorms[(index + 1) % 3];
            const auto r2Norm = projectionPointVertexNorms[index];

            if ((segmentNormalOrientations[index] == 0 && (r1Norm < EPSILON_ZERO_OFFSET || r2Norm < EPSILON_ZERO_OFFSET)) ||
                (std::abs(distances[index].s1 + distances[index].s2) < EPSILON_ZERO_OFFSET &&
                 std::abs(distances[index].l1 + distances[index].l2) < EPSILON_ZERO_OFFSET)) {
                transcendentalExpressions[index].ln = 0.0;
            } else {
                const FloatType inner_num = distances[index].s2 + distances[index].l2;
                const FloatType inner_denom = distances[index].s1 + distances[index].l1;

                if (inner_num <= 0.0 || inner_denom <= 0.0) {
                    transcendentalExpressions[index].ln = 0.0;
                } else {
                    transcendentalExpressions[index].ln = std::log(inner_num / inner_denom);
                }
            }

            if (planeDistance < EPSILON_ZERO_OFFSET || segmentDistances[index] < EPSILON_ZERO_OFFSET) {
                transcendentalExpressions[index].an = 0.0;
            } else {
                const auto frac1 = (planeDistance * distances[index].s2) / (segmentDistances[index] * distances[index].l2);
                const auto frac2 = (planeDistance * distances[index].s1) / (segmentDistances[index] * distances[index].l1);

                transcendentalExpressions[index].an = std::atan(frac1) - std::atan(frac2);
            }
        }
        //endregion

        //region 1-14 Step: Compute the singularities sing A and sing B
        Singularity singularities{};
        do {
            // 1. Case: P' lies inside the plane S_p
            bool allInside = true;
            for (unsigned int index = 0; index < 3; ++index) {
                allInside &= segmentNormalOrientations[index] == 1;
            }
            if (allInside) {
                singularities.a = -1.0 * PI2 * planeDistance;
                singularities.b = normals[i] * (-1.0 * PI2 * planeNormalOrientation);
                break;
            }

            // 2. Case: P' is located on one line segment G_p (but not on a vertex)
            bool anyOnLine = false;
            for (unsigned int index = 0; index < 3; ++index) {
                if (segmentNormalOrientations[index] != 0) {
                    continue;
                }
                const auto segmentVectorNorm = euclideanNorm(segmentVectors[i][index]);
                anyOnLine |= projectionPointVertexNorms[(index + 1) % 3] < segmentVectorNorm && projectionPointVertexNorms[index] < segmentVectorNorm;
            }

            if (anyOnLine) {
                singularities.a = -1.0 * PI * planeDistance;
                singularities.b = normals[i] * (-1.0 * PI * planeNormalOrientation);
                break;
            }

            // 3. Case: P' is located at one of G_p's vertices
            bool anyAtVertex = false;
            for (unsigned int index = 0; index < 3; ++index) {
                if (segmentNormalOrientations[index] != 0) {
                    continue;
                }

                const auto r1Norm = projectionPointVertexNorms[(index + 1) % 3];
                const auto r2Norm = projectionPointVertexNorms[index];

                if (!(r1Norm < EPSILON_ZERO_OFFSET || r2Norm < EPSILON_ZERO_OFFSET)) {
                    continue;
                }

                const Array3 &g1 = r1Norm == 0.0 ? segmentVectors[i][index] : segmentVectors[i][(index - 1 + 3) % 3];
                const Array3 &g2 = r1Norm == 0.0 ? segmentVectors[i][(index + 1) % 3] : segmentVectors[i][index];
                const FloatType gdot = dot(g1 * -1.0, g2);
                const FloatType theta = gdot == 0.0 ? PI_2 : std::acos(gdot / (euclideanNorm(g1) * euclideanNorm(g2)));

                singularities.a = -1.0 * theta * planeDistance;
                singularities.b = normals[i] * (-1.0 * theta * planeNormalOrientation);
                anyAtVertex = true;
                break;
            }

            // 4. Case: P' is located outside the plane S_p
            if (!anyAtVertex) {
                singularities.a = 0.0;
                singularities.b = {0.0, 0.0, 0.0};
            }
        } while (false);
        //endregion

        //region 2. Step: Sum 1 for potential and acceleration
        FloatType sum1PotentialAcceleration = 0.0;
        for (unsigned int index = 0; index < 3; ++index)
            sum1PotentialAcceleration += segmentNormalOrientations[index] * segmentDistances[index] * transcendentalExpressions[index].ln;
        //endregion

        //region 3. Step: Sum 1 for the gradiometric tensor
        Array3 sum1Tensor{0.0, 0.0, 0.0};
        for (unsigned int index = 0; index < 3; ++index)
            sum1Tensor = sum1Tensor + segmentNormals[i][index] * transcendentalExpressions[index].ln;
        //endregion

        //region 4. Step: Sum 2 (shared by all result parameters)
        FloatType sum2 = 0.0;
        for (unsigned int index = 0; index < 3; ++index)
            sum2 += segmentNormalOrientations[index] * transcendentalExpressions[index].an;
        //endregion

        //region 5. Step: Sum for potential and acceleration
        const FloatType planeSumPotentialAcceleration = sum1PotentialAcceleration + planeDistance * sum2 + singularities.a;
        //endregion

        //region 6. Step: Sum for tensor
        const Array3 subSum = (sum1Tensor + (normals[i] * (planeNormalOrientation * sum2))) + singularities.b;
        const Array3 first = normals[i] * subSum;
        const Array3 reorderedNp = {normals[i][0], normals[i][0], normals[i][1]};
        const Array3 reorderedSubSum = {subSum[1], subSum[2], subSum[2]};
        const Array3 second = reorderedNp * reorderedSubSum;
        //endregion

        //region 7. Step: Store the per-face contribution (prefix applied on the host)
        results[i] = GravityModelResult{
                planeNormalOrientation * planeDistance * planeSumPotentialAcceleration,
                normals[i] * planeSumPotentialAcceleration,
                concat(first, second)};
        //endregion
    }
};

} // namespace

GlobalResources::GlobalResources(int &argc, char *argv[]) {}
GlobalResources::~GlobalResources() = default;

class GravityEvaluable : public GravityEvaluableBase {
public:
    GravityEvaluable(
            const std::vector<Array3> &Vertices,
            const std::vector<IndexArray3> &Faces,
            const double density)
        : GravityEvaluableBase(Vertices, Faces, density)
        , host(alpaka::getDevByIdx(PlatformHost{}, 0))
        , device(alpaka::getDevByIdx(PlatformAcc{}, 0))
        , queue(device)
        , _vertices_d(alpaka::allocBuf<Array3, Idx>(device, extentOf(Vertices.size())))
        , _faces_d(alpaka::allocBuf<IndexArray3, Idx>(device, extentOf(Faces.size())))
        , _normals_d(alpaka::allocBuf<Array3, Idx>(device, extentOf(Faces.size())))
        , _segmentVectors_d(alpaka::allocBuf<Array3Triplet, Idx>(device, extentOf(Faces.size())))
        , _segmentNormals_d(alpaka::allocBuf<Array3Triplet, Idx>(device, extentOf(Faces.size())))
        , _results_d(alpaka::allocBuf<GravityModelResult, Idx>(device, extentOf(Faces.size()))) {
    }

    GravityModelResult evaluate(const Array3 &Point) override {
        if (!_initialized) init();

        const Idx num_faces = _faces.size();
        const auto extent = extentOf(num_faces);

        EvalKernel evalKernel;
        const auto workDiv = alpaka::getValidWorkDiv(
                alpaka::KernelCfg<Acc>{extent, alpaka::Vec<Dim, Idx>::ones()},
                device,
                evalKernel,
                alpaka::getPtrNative(_vertices_d),
                alpaka::getPtrNative(_faces_d),
                alpaka::getPtrNative(_normals_d),
                alpaka::getPtrNative(_segmentVectors_d),
                alpaka::getPtrNative(_segmentNormals_d),
                alpaka::getPtrNative(_results_d),
                num_faces,
                Point[0], Point[1], Point[2]);

        auto const taskKernel = alpaka::createTaskKernel<Acc>(
                workDiv,
                evalKernel,
                alpaka::getPtrNative(_vertices_d),
                alpaka::getPtrNative(_faces_d),
                alpaka::getPtrNative(_normals_d),
                alpaka::getPtrNative(_segmentVectors_d),
                alpaka::getPtrNative(_segmentNormals_d),
                alpaka::getPtrNative(_results_d),
                num_faces,
                Point[0], Point[1], Point[2]);

        alpaka::enqueue(queue, taskKernel);

        // Copy the per-face contributions back and reduce them on the host.
        std::vector<GravityModelResult> hostResults(num_faces);
        auto resultView = alpaka::createView(host, hostResults.data(), extent);
        alpaka::memcpy(queue, resultView, _results_d, extent);
        alpaka::wait(queue);

        GravityModelResult result{};
        for (Idx i = 0; i < num_faces; ++i) {
            result += hostResults[i];
        }

        const double prefix = GRAVITATIONAL_CONSTANT * _density;
        result.potential = (result.potential * prefix) / 2.0;
        result.acceleration = result.acceleration * (-1.0 * prefix);
        result.gradiometricTensor = result.gradiometricTensor * prefix;
        return result;
    }

private:
    void init() {
        const Idx num_vertices = _vertices.size();
        const Idx num_faces = _faces.size();

        auto verticesView = alpaka::createView(host, const_cast<Array3 *>(_vertices.data()), extentOf(num_vertices));
        auto facesView = alpaka::createView(host, const_cast<IndexArray3 *>(_faces.data()), extentOf(num_faces));
        alpaka::memcpy(queue, _vertices_d, verticesView, extentOf(num_vertices));
        alpaka::memcpy(queue, _faces_d, facesView, extentOf(num_faces));

        const auto extent = extentOf(num_faces);
        InitKernel initKernel;
        const auto workDiv = alpaka::getValidWorkDiv(
                alpaka::KernelCfg<Acc>{extent, alpaka::Vec<Dim, Idx>::ones()},
                device,
                initKernel,
                alpaka::getPtrNative(_vertices_d),
                alpaka::getPtrNative(_faces_d),
                alpaka::getPtrNative(_normals_d),
                alpaka::getPtrNative(_segmentVectors_d),
                alpaka::getPtrNative(_segmentNormals_d),
                num_faces);

        auto const taskKernel = alpaka::createTaskKernel<Acc>(
                workDiv,
                initKernel,
                alpaka::getPtrNative(_vertices_d),
                alpaka::getPtrNative(_faces_d),
                alpaka::getPtrNative(_normals_d),
                alpaka::getPtrNative(_segmentVectors_d),
                alpaka::getPtrNative(_segmentNormals_d),
                num_faces);

        alpaka::enqueue(queue, taskKernel);
        alpaka::wait(queue);

        _initialized = true;
    }

    Host host;
    DeviceAcc device;
    Queue queue;

    BufAcc<Array3> _vertices_d;
    BufAcc<IndexArray3> _faces_d;
    BufAcc<Array3> _normals_d;
    BufAcc<Array3Triplet> _segmentVectors_d;
    BufAcc<Array3Triplet> _segmentNormals_d;
    BufAcc<GravityModelResult> _results_d;
};

std::unique_ptr<GravityEvaluableBase> create_gravity_evaluable(
        const std::vector<Array3> &Vertices,
        const std::vector<IndexArray3> &Faces,
        double density) {
    return std::make_unique<GravityEvaluable>(Vertices, Faces, density);
}
