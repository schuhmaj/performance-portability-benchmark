#include "common/Marker.h"
#include "polyhedralGravity/PolyhedralGravityDefinitions.h"

#include <sycl/sycl.hpp>

GlobalResources::GlobalResources(int &argc, char *argv[]) {
}
GlobalResources::~GlobalResources() = default;

// The launch bounds of Kokkos' EvaluationPolicy: an explicit nd_range with 256 threads per group. A plain
// range<1> reduction is streamed by AdaptiveCpp: it launches only a few groups and strides each thread through the faces.
constexpr size_t groupSize = 256;

class GravityEvaluable : public GravityEvaluableBase {
public:
    GravityEvaluable(
            const std::vector<Array3> &Vertices,
            const std::vector<IndexArray3> &Faces,
            const double density)
        : GravityEvaluableBase(Vertices, Faces, density),
          queue{sycl::default_selector_v, sycl::property_list{sycl::property::queue::in_order()}} {
        const size_t numVertices = _vertices.size();
        const size_t numFaces = _faces.size();

        // Modern USM model: allocate the geometry on the device and stream the
        // host data over instead of wrapping it in sycl::buffers.
        _vertices_device = sycl::malloc_device<Array3>(numVertices, queue);
        _faces_device = sycl::malloc_device<IndexArray3>(numFaces, queue);
        _normals = sycl::malloc_device<Array3>(numFaces, queue);
        _result_device = sycl::malloc_device<GravityModelResult>(1, queue);

        queue.copy(_vertices.data(), _vertices_device, numVertices);
        queue.copy(_faces.data(), _faces_device, numFaces);
    }

    ~GravityEvaluable() override {
        sycl::free(_vertices_device, queue);
        sycl::free(_faces_device, queue);
        sycl::free(_normals, queue);
        sycl::free(_result_device, queue);
    }

    GravityModelResult evaluate(const Array3 &Point) override {
        if (!_initialized) init();
        PPB_MARKER_GPU_SCOPE("evaluate");

        const size_t numFaces = _faces.size();
        const Array3 point = Point;

        // Capture the device pointers as plain values for the kernel.
        const Array3 *V = _vertices_device;
        const IndexArray3 *F = _faces_device;
        const Array3 *N = _normals;

        // The result buffer is allocated once (see constructor) and reused across
        // calls - allocating/freeing USM on every evaluate() is a heavyweight,
        // synchronizing operation that dominates the (small) per-point kernel.
        const size_t globalSize = (numFaces + groupSize - 1) / groupSize * groupSize;

        queue.submit([&](sycl::handler &h) {
             auto reduction = sycl::reduction(_result_device, GravityModelResult{}, sycl::plus<GravityModelResult>(),
                                              sycl::property::reduction::initialize_to_identity{});

             h.parallel_for(sycl::nd_range<1>(globalSize, groupSize), reduction, [=](const sycl::nd_item<1> &item, auto &reducer) {
                 const size_t i = item.get_global_id(0);
                 if (i >= numFaces) return;
                 const Array3Triplet face = {
                         V[F[i][0]] - point,
                         V[F[i][1]] - point,
                         V[F[i][2]] - point};
                 const Array3 planeUnitNormal = N[i];

                 // Recomputed rather than cached: three subtractions are cheaper than reading 36 more bytes per face
                 const Array3Triplet segmentVectors = {face[1] - face[0], face[2] - face[1], face[0] - face[2]};

                 // N_p is a unit vector, so N_p * v_0 is the signed distance of P to the plane and P' is N_p scaled by it
                 const FloatType planeProjection = dot(planeUnitNormal, face[0]);
                 const FloatType planeNormalOrientation = sgn(planeProjection);
                 const FloatType planeDistance = std::abs(planeProjection);
                 const Array3 orthogonalProjectionPointOnPlane = planeUnitNormal * planeProjection;

                 // sigma_pq, h_pq, l1, l2, s1, s2 and |P' - v_q| follow from projecting P' - v_q onto n_pq and onto G_pq
                 const Array3 vertexNorms = {euclideanNorm(face[0]), euclideanNorm(face[1]), euclideanNorm(face[2])};
                 Array3Triplet segmentUnitNormals{};
                 Array3 segmentNormalOrientations{};
                 Array3 segmentDistances{};
                 Array3 projectionPointVertexNorms{};
                 std::array<Distance, 3> distances{};
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

                 // Both transcendental expressions are evaluated unconditionally and then selected
                 std::array<TranscendentalExpression, 3> transcendentalExpressions{};
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

                 // The singularities are sing A = factor * h_p and sing B = factor * sigma_p * N_p in all four cases
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

                 FloatType sum1PotentialAcceleration = 0;
                 for (unsigned int index = 0; index < 3; ++index)
                     sum1PotentialAcceleration += segmentNormalOrientations[index] * segmentDistances[index] *
                                                  transcendentalExpressions[index].ln;

                 Array3 sum1Tensor{};
                 for (unsigned int index = 0; index < 3; ++index)
                     sum1Tensor = sum1Tensor + segmentUnitNormals[index] * transcendentalExpressions[index].ln;

                 FloatType sum2 = 0;
                 for (unsigned int index = 0; index < 3; ++index)
                     sum2 += segmentNormalOrientations[index] * transcendentalExpressions[index].an;

                 const FloatType planeSumPotentialAcceleration =
                         sum1PotentialAcceleration + planeDistance * sum2 + singularities.a;

                 const Array3 subSum = (sum1Tensor + (planeUnitNormal * (planeNormalOrientation * sum2))) + singularities.b;

                 const Array3 first = planeUnitNormal * subSum;

                 const Array3 reorderedNp = {planeUnitNormal[0], planeUnitNormal[0], planeUnitNormal[1]};
                 const Array3 reorderedSubSum = {subSum[1], subSum[2], subSum[2]};
                 const Array3 second = reorderedNp * reorderedSubSum;

                 reducer.combine(GravityModelResult{
                         planeNormalOrientation * planeDistance * planeSumPotentialAcceleration,
                         planeUnitNormal * planeSumPotentialAcceleration,
                         concat(first, second)});
             });
         });

        GravityModelResult result{};
        queue.copy(_result_device, &result, 1).wait();

        const double prefix = GRAVITATIONAL_CONSTANT * _density;

        result.potential = (result.potential * prefix) / 2.0;
        result.acceleration = result.acceleration * (-1.0 * prefix);
        result.gradiometricTensor = result.gradiometricTensor * prefix;
        return result;
    }

private:
    void init() {
        PPB_MARKER_GPU_SCOPE("init");
        const size_t numFaces = _faces.size();

        const Array3 *V = _vertices_device;
        const IndexArray3 *F = _faces_device;
        Array3 *normals = _normals;

        // The plane unit normals N_p are the only per-face property that is cached
        queue.submit([&](sycl::handler &h) {
             h.parallel_for(sycl::range<1>(numFaces), [=](const sycl::id<1> &id) {
                 const size_t i = id[0];
                 const Array3Triplet face = {V[F[i][0]], V[F[i][1]], V[F[i][2]]};
                 normals[i] = normal(face[1] - face[0], face[2] - face[1]);
             });
         }).wait();

        _initialized = true;
    }

    sycl::queue queue;

    Array3 *_vertices_device = nullptr;
    IndexArray3 *_faces_device = nullptr;

    Array3 *_normals = nullptr;

    // Reused across evaluate() calls to avoid per-call USM allocation overhead.
    GravityModelResult *_result_device = nullptr;
};

std::unique_ptr<GravityEvaluableBase> create_gravity_evaluable(
        const std::vector<Array3> &Vertices,
        const std::vector<IndexArray3> &Faces,
        double density) {
    return std::make_unique<GravityEvaluable>(Vertices, Faces, density);
}
