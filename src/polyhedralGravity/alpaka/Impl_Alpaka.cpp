#include "common/Marker.h"
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
/** Pinned host memory where the platform supports it, regular host memory otherwise. */
template<typename T>
using BufHostMapped = decltype(alpaka::allocMappedBufIfSupported<T, Idx>(
        std::declval<Host>(), std::declval<PlatformAcc>(), std::declval<alpaka::Vec<Dim, Idx>>()));

/**
 * The accelerator platform, shared by the whole process. On the SYCL backend every platform object creates its own
 * SYCL context: the device, the pinned host buffer and the compiled kernels have to belong to the same one.
 */
PlatformAcc &getPlatformAcc() {
    static PlatformAcc platform{};
    return platform;
}

inline alpaka::Vec<Dim, Idx> extentOf(const Idx n) {
    return alpaka::Vec<Dim, Idx>{n};
}

/**
 * Precomputes the plane unit normals, the only per-face geometry that is cached
 * rather than recomputed for every evaluation point.
 */
struct InitKernel {
    template<typename TAcc>
    ALPAKA_FN_ACC void operator()(
            const TAcc &acc,
            const Array3 *vertices,
            const IndexArray3 *faces,
            Array3 *normals,
            const Idx num_faces) const {
        const Idx i = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0];
        if (i >= num_faces) {
            return;
        }

        const Array3Triplet face = {
                vertices[faces[i][0]],
                vertices[faces[i][1]],
                vertices[faces[i][2]]};

        normals[i] = normal(face[1] - face[0], face[2] - face[1]);
    }
};

/**
 * Evaluates the polyhedral gravity model for a single face with respect to the
 * computation point and stores the (not yet reduced) per-face contribution.
 */
struct EvalKernel {
    template<typename TAcc>
    ALPAKA_FN_ACC void operator()(
            const TAcc &acc,
            const Array3 *vertices,
            const IndexArray3 *faces,
            const Array3 *normals,
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
        const Array3 planeUnitNormal = normals[i];

        //region 1-01 Step: Compute the segment vectors G_pq (recomputed rather than read from a cache)
        const Array3Triplet segmentVectors = {face[1] - face[0], face[2] - face[1], face[0] - face[2]};
        //endregion

        //region 1-04 to 1-07 Step: Compute sigma_p, h_p and P' from the projection of the face onto N_p
        const FloatType planeProjection = dot(planeUnitNormal, face[0]);
        const FloatType planeNormalOrientation = sgn(planeProjection);
        const FloatType planeDistance = std::abs(planeProjection);
        const Array3 orthogonalProjectionPointOnPlane = planeUnitNormal * planeProjection;
        //endregion

        //region 1-08 to 1-12 Step: Compute sigma_pq, h_pq, l1, l2, s1, s2 and the norms of P' - v_q without forming P''
        const Array3 vertexNorms = {euclideanNorm(face[0]), euclideanNorm(face[1]), euclideanNorm(face[2])};
        Array3Triplet segmentUnitNormals{};
        Array3 segmentNormalOrientations{};
        Array3 segmentDistances{};
        Array3 projectionPointVertexNorms{};
        Distance distances[3];
        for (unsigned int index = 0; index < 3; ++index) {
            const Array3 relativeProjectionPoint = orthogonalProjectionPointOnPlane - face[index];
            projectionPointVertexNorms[index] = euclideanNorm(relativeProjectionPoint);

            // n_pq is normalized by |G_pq|, which is |G_pq x N_p| since N_p is a unit vector perpendicular to G_pq
            const FloatType squaredSegmentNorm = dot(segmentVectors[index], segmentVectors[index]);
            const FloatType inverseSegmentNorm = 1 / std::sqrt(squaredSegmentNorm);
            const FloatType segmentNorm = squaredSegmentNorm * inverseSegmentNorm;
            segmentUnitNormals[index] = cross(segmentVectors[index], planeUnitNormal) * inverseSegmentNorm;

            const FloatType normalProjection = dot(segmentUnitNormals[index], relativeProjectionPoint);
            segmentNormalOrientations[index] = -sgn(normalProjection);
            segmentDistances[index] = std::abs(normalProjection);

            const FloatType alongSegment = dot(relativeProjectionPoint, segmentVectors[index]) * inverseSegmentNorm;
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
        //endregion

        //region 1-13 Step: Compute the transcendental expressions LN_pq and AN_pq, evaluated unconditionally and selected
        TranscendentalExpression transcendentalExpressions[3];
        for (unsigned int index = 0; index < 3; ++index) {
            const Distance &distance = distances[index];
            const FloatType r1Norm = projectionPointVertexNorms[(index + 1) % 3];
            const FloatType r2Norm = projectionPointVertexNorms[index];

            const bool logarithmVanishes =
                    (segmentNormalOrientations[index] == 0 && (r1Norm < EPSILON_ZERO || r2Norm < EPSILON_ZERO)) ||
                    (std::abs(distance.s1 + distance.s2) < EPSILON_ZERO && std::abs(distance.l1 + distance.l2) < EPSILON_ZERO);
            const FloatType logarithm = std::log((distance.s2 + distance.l2) / (distance.s1 + distance.l1));
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
        //endregion

        //region 1-14 Step: Compute the singularities sing A = factor * h_p and sing B = factor * sigma_p * N_p
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
            const FloatType squaredSegmentNorm = dot(segmentVectors[index], segmentVectors[index]);
            const bool atVertex = r1Norm < EPSILON_ZERO || r2Norm < EPSILON_ZERO;
            anyOnLine |= !atVertex && r1Norm * r1Norm < squaredSegmentNorm && r2Norm * r2Norm < squaredSegmentNorm;
            vertexSegment = anyAtVertex ? vertexSegment : index;
            vertexIsSegmentEnd = anyAtVertex ? vertexIsSegmentEnd : r1Norm < EPSILON_ZERO;
            anyAtVertex |= atVertex;
        }
        FloatType vertexAngle = 0;
        if (anyAtVertex) {
            const Array3 &g1 = vertexIsSegmentEnd ? segmentVectors[vertexSegment] : segmentVectors[(vertexSegment + 2) % 3];
            const Array3 &g2 = vertexIsSegmentEnd ? segmentVectors[(vertexSegment + 1) % 3] : segmentVectors[vertexSegment];
            const FloatType gdot = -dot(g1, g2);
            vertexAngle = gdot == 0 ? PI_2 : std::acos(gdot / (euclideanNorm(g1) * euclideanNorm(g2)));
        }
        const FloatType singularityFactor = allInside ? -PI2 : anyOnLine ? -PI : -vertexAngle;
        const Singularity singularities{singularityFactor * planeDistance, planeUnitNormal * (singularityFactor * planeNormalOrientation)};
        //endregion

        //region 2. Step: Sum 1 for potential and acceleration
        FloatType sum1PotentialAcceleration = 0;
        for (unsigned int index = 0; index < 3; ++index)
            sum1PotentialAcceleration += segmentNormalOrientations[index] * segmentDistances[index] * transcendentalExpressions[index].ln;
        //endregion

        //region 3. Step: Sum 1 for the gradiometric tensor
        Array3 sum1Tensor{};
        for (unsigned int index = 0; index < 3; ++index)
            sum1Tensor = sum1Tensor + segmentUnitNormals[index] * transcendentalExpressions[index].ln;
        //endregion

        //region 4. Step: Sum 2 (shared by all result parameters)
        FloatType sum2 = 0;
        for (unsigned int index = 0; index < 3; ++index)
            sum2 += segmentNormalOrientations[index] * transcendentalExpressions[index].an;
        //endregion

        //region 5. Step: Sum for potential and acceleration
        const FloatType planeSumPotentialAcceleration = sum1PotentialAcceleration + planeDistance * sum2 + singularities.a;
        //endregion

        //region 6. Step: Sum for tensor
        const Array3 subSum = (sum1Tensor + (planeUnitNormal * (planeNormalOrientation * sum2))) + singularities.b;
        const Array3 first = planeUnitNormal * subSum;
        const Array3 reorderedNp = {planeUnitNormal[0], planeUnitNormal[0], planeUnitNormal[1]};
        const Array3 reorderedSubSum = {subSum[1], subSum[2], subSum[2]};
        const Array3 second = reorderedNp * reorderedSubSum;
        //endregion

        //region 7. Step: Store the per-face contribution (prefix applied on the host)
        results[i] = GravityModelResult{
                planeNormalOrientation * planeDistance * planeSumPotentialAcceleration,
                planeUnitNormal * planeSumPotentialAcceleration,
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
        , device(alpaka::getDevByIdx(getPlatformAcc(), 0))
        , queue(device)
        , _vertices_d(alpaka::allocBuf<Array3, Idx>(device, extentOf(Vertices.size())))
        , _faces_d(alpaka::allocBuf<IndexArray3, Idx>(device, extentOf(Faces.size())))
        , _normals_d(alpaka::allocBuf<Array3, Idx>(device, extentOf(Faces.size())))
        , _results_d(alpaka::allocBuf<GravityModelResult, Idx>(device, extentOf(Faces.size())))
        , _results_h(alpaka::allocMappedBufIfSupported<GravityModelResult, Idx>(host, getPlatformAcc(), extentOf(Faces.size()))) {
    }

    GravityModelResult evaluate(const Array3 &Point) override {
        if (!_initialized) init();
        PPB_MARKER_GPU_SCOPE("evaluate");

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
                alpaka::getPtrNative(_results_d),
                num_faces,
                Point[0], Point[1], Point[2]);

        auto const taskKernel = alpaka::createTaskKernel<Acc>(
                workDiv,
                evalKernel,
                alpaka::getPtrNative(_vertices_d),
                alpaka::getPtrNative(_faces_d),
                alpaka::getPtrNative(_normals_d),
                alpaka::getPtrNative(_results_d),
                num_faces,
                Point[0], Point[1], Point[2]);

        alpaka::enqueue(queue, taskKernel);

        // Copy the per-face contributions back into the host buffer allocated once, and reduce them on the host.
        alpaka::memcpy(queue, _results_h, _results_d, extent);
        alpaka::wait(queue);

        const GravityModelResult *hostResults = alpaka::getPtrNative(_results_h);
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
        PPB_MARKER_GPU_SCOPE("init");
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
                num_faces);

        auto const taskKernel = alpaka::createTaskKernel<Acc>(
                workDiv,
                initKernel,
                alpaka::getPtrNative(_vertices_d),
                alpaka::getPtrNative(_faces_d),
                alpaka::getPtrNative(_normals_d),
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
    BufAcc<GravityModelResult> _results_d;
    BufHostMapped<GravityModelResult> _results_h;
};

std::unique_ptr<GravityEvaluableBase> create_gravity_evaluable(
        const std::vector<Array3> &Vertices,
        const std::vector<IndexArray3> &Faces,
        double density) {
    return std::make_unique<GravityEvaluable>(Vertices, Faces, density);
}
