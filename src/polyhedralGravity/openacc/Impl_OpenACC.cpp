#include "common/Marker.h"
#include <memory>

#include "polyhedralGravity/PolyhedralGravityDefinitions.h"
#include <openacc.h>

GlobalResources::GlobalResources(int &argc, char *argv[]) {
}
GlobalResources::~GlobalResources() = default;

template<typename T>
T *allocateOpenACC(size_t n_elem) {
    return static_cast<T *>(acc_malloc(sizeof(T) * n_elem));
}

#pragma omp declare reduction(+ : GravityModelResult : omp_out += omp_in) initializer(omp_priv = GravityModelResult())

class GravityEvaluable : public GravityEvaluableBase {
public:
    GravityEvaluable(
            const std::vector<Array3> &Vertices,
            const std::vector<IndexArray3> &Faces,
            const double density)
        : GravityEvaluableBase(Vertices, Faces, density),
          _facesDevice(allocateOpenACC<IndexArray3>(_faces.size())),
          _verticesDevice(allocateOpenACC<Array3>(_vertices.size())),
          _normals(allocateOpenACC<Array3>(_faces.size())) {
        acc_memcpy_to_device(_facesDevice, (void *) (_faces.data()), sizeof(IndexArray3) * _faces.size());
        acc_memcpy_to_device(_verticesDevice, (void *) _vertices.data(), sizeof(Array3) * _vertices.size());
    }

    ~GravityEvaluable() override {
        acc_free(_facesDevice);
        acc_free(_verticesDevice);
        acc_free(_normals);
    }

    GravityModelResult evaluate(const Array3 &Point) override {
        if (!_initialized) init();
        PPB_MARKER_GPU_SCOPE("evaluate");

        // Every face adds its contribution to the reduction directly, so no per-face results have to be stored
        // and summed up on the host. OpenACC reduces arithmetic scalars and arrays of them, but no structs.
        FloatType potential = 0;
        FloatType acceleration[3] = {0, 0, 0};
        FloatType gradiometricTensor[6] = {0, 0, 0, 0, 0, 0};
        size_t face_count = _faces.size();
#pragma acc parallel loop reduction(+ : potential, acceleration[0:3], gradiometricTensor[0:6])
        for (size_t i = 0; i < face_count; ++i) {
            const Array3Triplet face = {
                    _verticesDevice[_facesDevice[i][0]] - Point,
                    _verticesDevice[_facesDevice[i][1]] - Point,
                    _verticesDevice[_facesDevice[i][2]] - Point};
            const Array3 planeUnitNormal = _normals[i];

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
                sum1PotentialAcceleration += segmentNormalOrientations[index] * segmentDistances[index] * transcendentalExpressions[index].ln;

            Array3 sum1Tensor{};
            for (unsigned int index = 0; index < 3; ++index)
                sum1Tensor = sum1Tensor + segmentUnitNormals[index] * transcendentalExpressions[index].ln;

            FloatType sum2 = 0;
            for (unsigned int index = 0; index < 3; ++index)
                sum2 += segmentNormalOrientations[index] * transcendentalExpressions[index].an;

            const FloatType planeSumPotentialAcceleration = sum1PotentialAcceleration + planeDistance * sum2 + singularities.a;

            const Array3 subSum = (sum1Tensor + (planeUnitNormal * (planeNormalOrientation * sum2))) + singularities.b;
            const Array3 first = planeUnitNormal * subSum;
            const Array3 reorderedNp = {planeUnitNormal[0], planeUnitNormal[0], planeUnitNormal[1]};
            const Array3 reorderedSubSum = {subSum[1], subSum[2], subSum[2]};
            const Array3 second = reorderedNp * reorderedSubSum;

            potential += planeNormalOrientation * planeDistance * planeSumPotentialAcceleration;
            for (unsigned int index = 0; index < 3; ++index) {
                acceleration[index] += planeUnitNormal[index] * planeSumPotentialAcceleration;
                gradiometricTensor[index] += first[index];
                gradiometricTensor[index + 3] += second[index];
            }
        }

        GravityModelResult result{potential,
                                  {acceleration[0], acceleration[1], acceleration[2]},
                                  {gradiometricTensor[0], gradiometricTensor[1], gradiometricTensor[2], gradiometricTensor[3], gradiometricTensor[4], gradiometricTensor[5]}};

        const double prefix = GRAVITATIONAL_CONSTANT * _density;

        result.potential = (result.potential * prefix) / 2.0;
        result.acceleration = result.acceleration * (-1.0 * prefix);
        result.gradiometricTensor = result.gradiometricTensor * prefix;
        return result;
    }

private:
    void init() {
        PPB_MARKER_GPU_SCOPE("init");
        size_t face_count = _faces.size();
        // The plane unit normals N_p are the only per-face property that is cached
#pragma acc parallel loop
        for (size_t i = 0; i < face_count; ++i) {
            const Array3Triplet face = {
                    _verticesDevice[_facesDevice[i][0]],
                    _verticesDevice[_facesDevice[i][1]],
                    _verticesDevice[_facesDevice[i][2]]};
            _normals[i] = normal(face[1] - face[0], face[2] - face[1]);
        }
        _initialized = true;
    }
    IndexArray3 *_facesDevice;
    Array3 *_verticesDevice;

    Array3 *_normals;
};

std::unique_ptr<GravityEvaluableBase> create_gravity_evaluable(
        const std::vector<Array3> &Vertices,
        const std::vector<IndexArray3> &Faces,
        double density) {
    return std::make_unique<GravityEvaluable>(Vertices, Faces, density);
}
