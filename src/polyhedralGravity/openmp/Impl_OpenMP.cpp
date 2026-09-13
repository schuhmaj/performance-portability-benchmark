#include "common/Marker.h"
#include <memory>

#include "polyhedralGravity/PolyhedralGravityDefinitions.h"
#include "omp.h"

GlobalResources::GlobalResources(int &argc, char *argv[]) {
}
GlobalResources::~GlobalResources() = default;

template<typename T>
T *allocateOpenMp(size_t n_elem, int device_num) {
    return static_cast<T *>(omp_target_alloc(sizeof(T) * n_elem, device_num));
}

#pragma omp declare reduction(+ : GravityModelResult : omp_out += omp_in) initializer(omp_priv = GravityModelResult())

class GravityEvaluable : public GravityEvaluableBase {
public:
    GravityEvaluable(
            const std::vector<Array3> &Vertices,
            const std::vector<IndexArray3> &Faces,
            const double density)
        : GravityEvaluableBase(Vertices, Faces, density),
          device_num((omp_get_num_devices() > 3) ? 2 : 0),
          _facesDevice(allocateOpenMp<IndexArray3>(_faces.size(), device_num)),
          _verticesDevice(allocateOpenMp<Array3>(_vertices.size(), device_num)),
          _normals(allocateOpenMp<Array3>(_faces.size(), device_num)) {
        int src_device_num = omp_get_initial_device();

        std::cout << "Device number: " << device_num << std::endl;

        omp_target_memcpy(_facesDevice, _faces.data(), sizeof(IndexArray3) * _faces.size(), 0, 0, device_num, src_device_num);
        omp_target_memcpy(_verticesDevice, _vertices.data(), sizeof(Array3) * _vertices.size(), 0, 0, device_num, src_device_num);
    }

    ~GravityEvaluable() override {
        omp_target_free(_facesDevice, device_num);
        omp_target_free(_verticesDevice, device_num);
        omp_target_free(_normals, device_num);
    }

    GravityModelResult evaluate(const Array3 &Point) override {
        if (!_initialized) init();
        PPB_MARKER_GPU_SCOPE("evaluate");

        GravityModelResult result{};
        size_t face_count = _faces.size();
        // Every face adds its contribution to the reduction directly, so no per-face results have to be stored
        // and summed up by a second kernel
#pragma omp target teams distribute parallel for reduction(+ : result) device(device_num)
        for (size_t i = 0; i < face_count; ++i) {
            const Array3Triplet face = {
                    _verticesDevice[_facesDevice[i][0]] - Point,
                    _verticesDevice[_facesDevice[i][1]] - Point,
                    _verticesDevice[_facesDevice[i][2]] - Point};
            const Array3 planeUnitNormal = _normals[i];

            //region 1-01 Step: Compute the segment vectors G_pq
            // Recomputed rather than cached: three subtractions are cheaper than reading 36 more bytes per face
            const Array3Triplet segmentVectors = {face[1] - face[0], face[2] - face[1], face[0] - face[2]};
            //endregion

            //region 1-04 to 1-07 Step: Compute sigma_p, h_p and P' from the projection of the face onto N_p
            // N_p is a unit vector, so N_p * v_0 is the signed distance of P to the plane and P' is N_p scaled by it.
            // This is the value the Hessian normal form and the sign rules of Tsoulis' (22) arrive at.
            const FloatType planeProjection = dot(planeUnitNormal, face[0]);
            const FloatType planeNormalOrientation = sgn(planeProjection);
            const FloatType planeDistance = std::abs(planeProjection);
            const Array3 orthogonalProjectionPointOnPlane = planeUnitNormal * planeProjection;
            //endregion

            //region 1-08 to 1-12 Step: Compute sigma_pq, h_pq, the distances l1, l2, s1, s2 and the norms of P' - v_q
            // All of them follow from projecting P' - v_q onto n_pq and onto G_pq, so P'' is never formed
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

                // The projection onto n_pq has the sign -sigma_pq and the magnitude h_pq
                const FloatType normalProjection = dot(segmentUnitNormals[index], relativeProjectionPoint);
                segmentNormalOrientations[index] = -sgn(normalProjection);
                segmentDistances[index] = std::abs(normalProjection);

                // The projection onto G_pq is the signed position u of P'' along the segment, from its first endpoint
                const FloatType alongSegment = dot(relativeProjectionPoint, segmentVectors[index]) * inverseSegmentNorm;
                Distance &distance = distances[index];
                distance.l1 = vertexNorms[index];
                distance.l2 = vertexNorms[(index + 1) % 3];
                distance.s1 = std::abs(alongSegment);
                distance.s2 = std::abs(alongSegment - segmentNorm);

                // The signs of Tsoulis (2021): the 1., 2. and 3. Option all amount to s1 = -u and s2 = |G_pq| - u,
                // only the 4. Option (P is located on the line of the segment) is told apart
                if (std::abs(distance.s1 - distance.l1) >= EPSILON_ZERO || std::abs(distance.s2 - distance.l2) >= EPSILON_ZERO) {
                    distance.s1 = -alongSegment;
                    distance.s2 = segmentNorm - alongSegment;
                } else if (distance.s2 < distance.s1) {
                    // 4. Option - Case 2: P is located on the segment from its right side
                    distance.s1 = -distance.s1;
                    distance.s2 = -distance.s2;
                    distance.l1 = -distance.l1;
                    distance.l2 = -distance.l2;
                } else if (std::abs(distance.s2 - distance.s1) < EPSILON_ZERO) {
                    // 4. Option - Case 1: P is located inside the segment
                    distance.s1 = -distance.s1;
                    distance.l1 = -distance.l1;
                }
                // 4. Option - Case 3: P is located on the segment from its left side --> Nothing to do!
            }
            //endregion

            //region 1-13 Step: Compute the transcendental expressions LN_pq and AN_pq
            // Both are evaluated unconditionally and then selected: their guards only hold for degenerate positions of P'
            std::array<TranscendentalExpression, 3> transcendentalExpressions{};
            for (unsigned int index = 0; index < 3; ++index) {
                const Distance &distance = distances[index];
                const FloatType r1Norm = projectionPointVertexNorms[(index + 1) % 3];
                const FloatType r2Norm = projectionPointVertexNorms[index];

                // LN_pq according to (14), zero if P' lies on a vertex of the segment or P on the line of the segment
                const bool logarithmVanishes =
                        (segmentNormalOrientations[index] == 0 && (r1Norm < EPSILON_ZERO || r2Norm < EPSILON_ZERO)) ||
                        (std::abs(distance.s1 + distance.s2) < EPSILON_ZERO && std::abs(distance.l1 + distance.l2) < EPSILON_ZERO);
                const FloatType logarithm = std::log((distance.s2 + distance.l2) / (distance.s1 + distance.l1));
                transcendentalExpressions[index].ln = logarithmVanishes ? 0 : logarithm;

                // AN_pq according to (15), zero if h_p == 0 or h_pq == 0
                // The difference of the two arc tangents is taken as one, atan(x) - atan(y) = atan((x - y) / (1 + xy)),
                // which misses a whole PI (with the sign of x) exactly when 1 + xy is negative
                const bool arcTangentVanishes = planeDistance < EPSILON_ZERO || segmentDistances[index] < EPSILON_ZERO;
                const FloatType upper = (planeDistance * distance.s2) / (segmentDistances[index] * distance.l2);
                const FloatType lower = (planeDistance * distance.s1) / (segmentDistances[index] * distance.l1);
                const FloatType denominator = 1 + upper * lower;
                const FloatType branchOffset = denominator < 0 ? (upper < 0 ? -PI : PI) : 0;
                const FloatType arcTangent = std::atan((upper - lower) / denominator) + branchOffset;
                transcendentalExpressions[index].an = arcTangentVanishes ? 0 : arcTangent;
            }
            //endregion

            //region 1-14 Step: Compute the singularities sing A and sing B if P' is located in the plane, on a segment or on a vertex
            // All four cases share the shape sing A = factor * h_p and sing B = factor * sigma_p * N_p
            // 1. Case: If all sigma_pq are 1 then P' lies inside the plane S_p
            const bool allInside = segmentNormalOrientations[0] == 1 && segmentNormalOrientations[1] == 1 && segmentNormalOrientations[2] == 1;
            // 2. Case: If sigma_pq == 0 and P' is closer than |G_pq| to both endpoints, P' lies on the segment, not on a vertex
            // 3. Case: If sigma_pq == 0 and P' coincides with one of the endpoints, P' lies on a vertex
            bool anyOnLine = false;
            bool anyAtVertex = false;
            unsigned int vertexSegment = 0;
            bool vertexIsSegmentEnd = false;
            for (unsigned int index = 0; index < 3; ++index) {
                // A branch rather than a selection: on nearly every face P' lies on the line of none of the segments
                if (segmentNormalOrientations[index] != 0) {
                    continue;
                }
                const FloatType r1Norm = projectionPointVertexNorms[(index + 1) % 3];
                const FloatType r2Norm = projectionPointVertexNorms[index];
                // Both sides are non-negative, so comparing the squares saves the square root of |G_pq|
                const FloatType squaredSegmentNorm = dot(segmentVectors[index], segmentVectors[index]);
                const bool atVertex = r1Norm < EPSILON_ZERO || r2Norm < EPSILON_ZERO;
                anyOnLine |= !atVertex && r1Norm * r1Norm < squaredSegmentNorm && r2Norm * r2Norm < squaredSegmentNorm;
                // The first segment P' is found on decides which vertex it is
                vertexSegment = anyAtVertex ? vertexSegment : index;
                vertexIsSegmentEnd = anyAtVertex ? vertexIsSegmentEnd : r1Norm < EPSILON_ZERO;
                anyAtVertex |= atVertex;
            }
            // The angle theta of the 3. Case needs the only arc cosine, so it stays guarded
            FloatType vertexAngle = 0;
            if (anyAtVertex) {
                const Array3 &g1 = vertexIsSegmentEnd ? segmentVectors[vertexSegment] : segmentVectors[(vertexSegment + 2) % 3];
                const Array3 &g2 = vertexIsSegmentEnd ? segmentVectors[(vertexSegment + 1) % 3] : segmentVectors[vertexSegment];
                // theta = arccos((G_2 * -G_1) / (|G_2| * |G_1|))
                const FloatType gdot = -dot(g1, g2);
                vertexAngle = gdot == 0 ? PI_2 : std::acos(gdot / (euclideanNorm(g1) * euclideanNorm(g2)));
            }
            // The 1. Case takes precedence over the 2., the 2. over the 3., and in the 4. Case there is no singularity
            const FloatType singularityFactor = allInside ? -PI2 : anyOnLine ? -PI : -vertexAngle;
            const Singularity singularities{singularityFactor * planeDistance, planeUnitNormal * (singularityFactor * planeNormalOrientation)};
            //endregion

            //region 2. Step: Compute Sum 1 used for potential and acceleration (first derivative)
            // sum over: sigma_pq * h_pq * LN_pq
            // --> Equation 11/12 the first summation in the brackets
            FloatType sum1PotentialAcceleration = 0;
            for (unsigned int index = 0; index < 3; ++index)
                sum1PotentialAcceleration += segmentNormalOrientations[index] * segmentDistances[index] * transcendentalExpressions[index].ln;
            //endregion

            //region 3. Step: Compute Sum 1 used for the gradiometric tensor (second derivative)
            // sum over: n_pq * LN_pq
            // --> Equation 13 the first summation in the brackets
            Array3 sum1Tensor{};
            for (unsigned int index = 0; index < 3; ++index)
                sum1Tensor = sum1Tensor + segmentUnitNormals[index] * transcendentalExpressions[index].ln;
            //endregion

            //region 4. Step: Compute Sum 2 which is the same for every result parameter
            FloatType sum2 = 0;
            for (unsigned int index = 0; index < 3; ++index)
                sum2 += segmentNormalOrientations[index] * transcendentalExpressions[index].an;
            //endregion

            //region 5. Step: Sum for potential and acceleration
            // consisting of: sum1 + h_p * sum2 + sing A
            // --> Equation 11/12 the total sum of the brackets
            const FloatType planeSumPotentialAcceleration = sum1PotentialAcceleration + planeDistance * sum2 + singularities.a;
            //endregion

            //region 6. Step: Sum for tensor
            // consisting of: sum1 + sigma_p * N_p * sum2 + sing B
            // --> Equation 13 the total sum of the brackets
            const Array3 subSum = (sum1Tensor + (planeUnitNormal * (planeNormalOrientation * sum2))) + singularities.b;
            // first component: trivial case Vxx, Vyy, Vzz --> just N_p * subSum
            // 00, 11, 22 --> xx, yy, zz with x as 0, y as 1, z as 2
            const Array3 first = planeUnitNormal * subSum;
            // second component: reordering required to build Vxy, Vxz, Vyz
            // 01, 02, 12 --> xy, xz, yz with x as 0, y as 1, z as 2
            const Array3 reorderedNp = {planeUnitNormal[0], planeUnitNormal[0], planeUnitNormal[1]};
            const Array3 reorderedSubSum = {subSum[1], subSum[2], subSum[2]};
            const Array3 second = reorderedNp * reorderedSubSum;
            //endregion

            //region 7. Step: Multiply with prefix
            result += GravityModelResult{
                    // Equation (11): sigma_p * h_p * sum
                    planeNormalOrientation * planeDistance * planeSumPotentialAcceleration,

                    // Equation (12): N_p * sum
                    planeUnitNormal * planeSumPotentialAcceleration,

                    // Equation (13): already done above, just concat the two components for later summation
                    concat(first, second)};
            //endregion
        }

        // 9. Step: Compute prefix consisting of GRAVITATIONAL_CONSTANT * density
        const double prefix = GRAVITATIONAL_CONSTANT * _density;

        // 10. Step: Final expressions after application of the prefix (and a division by 2 for the potential)
        result.potential = (result.potential * prefix) / 2.0;
        result.acceleration = result.acceleration * (-1.0 * prefix);
        result.gradiometricTensor = result.gradiometricTensor * prefix;
        return result;
    }

private:
    void init() {
        PPB_MARKER_GPU_SCOPE("init");
        size_t face_count = _faces.size();
        // 1-02 Step: The plane unit normals N_p are the only per-face property that is cached
#pragma omp target teams distribute parallel for device(device_num)
        for (size_t i = 0; i < face_count; ++i) {
            const Array3Triplet face = {
                    _verticesDevice[_facesDevice[i][0]],
                    _verticesDevice[_facesDevice[i][1]],
                    _verticesDevice[_facesDevice[i][2]]};
            _normals[i] = normal(face[1] - face[0], face[2] - face[1]);
        }

        _initialized = true;
    }
    int device_num = 0;

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
