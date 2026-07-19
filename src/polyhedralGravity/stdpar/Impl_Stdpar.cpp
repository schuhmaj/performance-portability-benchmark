#include <algorithm>
#include <execution>
#include <functional>
#include <memory>
#include <numeric>
#include <ranges>
#include <vector>

#include "polyhedralGravity/PolyhedralGravityDefinitions.h"

GlobalResources::GlobalResources(int &argc, char *argv[]) {
}
GlobalResources::~GlobalResources() = default;

/**
 * Polyhedral gravity model parallelized with the ISO C++ standard parallel algorithms.
 * All buffers are plain std::vector allocations: the stdpar-offloading toolchains
 * (NVHPC -stdpar=gpu, ROCm --hipstdpar, Intel -fsycl-pstl-offload=gpu, AdaptiveCpp --acpp-stdpar)
 * place them in unified/managed memory, so no explicit host/device transfers are needed.
 * The lambdas capture raw pointers by value to avoid dereferencing `this` on the device.
 */
class GravityEvaluable : public GravityEvaluableBase {
public:
    GravityEvaluable(
            const std::vector<Array3> &Vertices,
            const std::vector<IndexArray3> &Faces,
            const double density)
        : GravityEvaluableBase(Vertices, Faces, density), _normals(Faces.size()), _segmentVectors(Faces.size()), _segmentNormals(Faces.size()), _results(Faces.size()) {
    }

    GravityModelResult evaluate(const Array3 &Point) override {
        if (!_initialized) init();

        const size_t faceCount = _faces.size();
        const Array3 *vertices = _vertices.data();
        const IndexArray3 *faces = _faces.data();
        const Array3 *normals = _normals.data();
        const Array3Triplet *segmentVectors = _segmentVectors.data();
        const Array3Triplet *segmentNormals = _segmentNormals.data();
        GravityModelResult *results = _results.data();
        const Array3 point = Point;

        const auto indices = std::views::iota(size_t{0}, faceCount);
        std::for_each(std::execution::par_unseq, indices.begin(), indices.end(), [=](const size_t i) {
            Array3Triplet face = {
                    vertices[faces[i][0]] - point,
                    vertices[faces[i][1]] - point,
                    vertices[faces[i][2]] - point};

            //region 1-04 Step: Compute Plane Normal Orientation sigma_p
            int planeNormalOrientation = sgn(dot(normals[i], face[0]));
            //endregion

            //region 1-05 Step: Compute Hessian Normal Plane Representation
            HessianPlane hessianPlane{};
            {
                // No constexpr Array3 origin{0,0,0} here: nvc++ -stdpar=gpu materializes it as a
                // `__staticinit` device constant with malformed NVVM IR (NVHPC 26.3); negate directly.
                const auto crossProduct = cross(face[0] - face[1], face[0] - face[2]);
                const auto res = crossProduct * face[0];
                const auto d = -(res[0] + res[1] + res[2]);

                hessianPlane = {crossProduct[0], crossProduct[1], crossProduct[2], d};
            }
            //endregion

            //region 1-06 Step: Compute distance h_p between P and P'
            auto planeDistance = std::abs(hessianPlane.d / std::sqrt(
                                                                   hessianPlane.a * hessianPlane.a + hessianPlane.b * hessianPlane.b + hessianPlane.c * hessianPlane.c));
            //endregion

            //region 1-07 Step: Compute the actual position of P' (projection of P on the plane)

            //Calculate the projection point by (22) P'_ = N_i / norm(N_i) * h_i
            // norm(N_i) is always 1 since N_i is a "normed" vector --> we do not need this division

            Array3 orthogonalProjectionPointOnPlane = normals[i] * planeDistance;
            {
                //Calculate alpha, beta and gamma as D/A, D/B and D/C (Notice that we "forget" the minus before those
                // divisions. In consequence, the conditions for signs are reversed below!!!)
                // These values represent the intersections of each polygonal plane with the axes
                // Comparison x == 0.0 is ok, since we only want to avoid nan values
                Array3 intersections = {hessianPlane.a == 0.0 ? static_cast<FloatType>(0.0) : hessianPlane.d / hessianPlane.a,
                                        hessianPlane.b == 0.0 ? static_cast<FloatType>(0.0) : hessianPlane.d / hessianPlane.b,
                                        hessianPlane.c == 0.0 ? static_cast<FloatType>(0.0) : hessianPlane.d / hessianPlane.c};

                //Determine the signs of the coordinates of P' according to the intersection values alpha, beta, gamma
                // denoted as __ below, i.e. -alpha, -beta, -gamma denoted -__
                for (unsigned int index = 0; index < 3; ++index) {
                    if (intersections[index] < 0) {
                        //If -__ >= 0 --> __ < 0 then coordinates are positive, we calculate abs(orthogonalProjectionPoint[..])
                        orthogonalProjectionPointOnPlane[index] = std::abs(orthogonalProjectionPointOnPlane[index]);
                    } else {
                        //The coordinates need to be controlled
                        if (orthogonalProjectionPointOnPlane[index] > 0) {
                            //If -__ < 0 --> __ >= 0 then the coordinate is negative -orthogonalProjectionPoint[..]
                            orthogonalProjectionPointOnPlane[index] = -1.0 * orthogonalProjectionPointOnPlane[index];
                        } else {
                            //Else the coordinate is positive orthogonalProjectionPoint[..]
                            orthogonalProjectionPointOnPlane[index] = orthogonalProjectionPointOnPlane[index];
                        }
                    }
                }
            }
            //endregion

            //region 1-08 Step: Compute the segment normal orientation sigma_pq (direction of n_pq in relation to P')
            std::array<int, 3> segmentNormalOrientations{};
            for (unsigned int index = 0; index < 3; ++index) {
                segmentNormalOrientations[index] = -sgn(dot(segmentNormals[i][index], orthogonalProjectionPointOnPlane - face[index]));
            }
            //endregion

            //region 1-09 Step: Compute the orthogonal projection point P'' of P' on each segment
            Array3Triplet orthogonalProjectionPointsOnSegmentsForPlane{};
            for (unsigned int index = 0; index < 3; ++index) {
                if (segmentNormalOrientations[index] == 0) {
                    //Geometrically trivial case, in neither of the half space --> already on segment
                    orthogonalProjectionPointsOnSegmentsForPlane[index] = orthogonalProjectionPointOnPlane;
                } else {
                    // In one of the half space, evaluate the projection point P'' for the segment with the endpoints v1 and v2
                    const auto &vertex1 = face[index];
                    const auto &vertex2 = face[(index + 1) % 3];

                    const Array3 matrixRow1 = vertex2 - vertex1;
                    const Array3 matrixRow2 = cross(vertex1 - orthogonalProjectionPointOnPlane, matrixRow1);
                    const Array3 matrixRow3 = cross(matrixRow2, matrixRow1);
                    const Array3 d = {dot(matrixRow1, orthogonalProjectionPointOnPlane),
                                      dot(matrixRow2, orthogonalProjectionPointOnPlane), dot(matrixRow3, vertex1)};
                    Matrix columnMatrix = transpose({matrixRow1, matrixRow2, matrixRow3});

                    // Calculation and solving the equations of above
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

            //region 1-11 Step: Compute the 3D distances l1, l2 (between P and vertices) and 1D distances s1, s2 (between P'' and vertices)
            std::array<Distance, 3> distances{};
            for (unsigned int index = 0; index < 3; ++index) {
                // Calculate the 3D distances between P (0, 0, 0) and
                // the segment endpoints face[j] and face[(j + 1) % 3])
                distances[index].l1 = euclideanNorm(face[index]);
                distances[index].l2 = euclideanNorm(face[(index + 1) % 3]);

                // Calculate the 1D distances between P'' (every segment has its own) and
                // the segment endpoints face[j] and face[(j + 1) % 3])
                distances[index].s1 = euclideanNorm(orthogonalProjectionPointsOnSegmentsForPlane[index] - face[index]);
                distances[index].s2 = euclideanNorm(orthogonalProjectionPointsOnSegmentsForPlane[index] - face[(index + 1) % 3]);

                /*
         * Additional remark:
         * Details on these conditions are in the second paper referenced in the README.md (Tsoulis, 2021)
         * The numbering of these conditions is equal to the numbering scheme of the paper
         * Assign a sign to those magnitudes depending on the relative position of P'' to the two
         * segment endpoints
        */

                //4. Option: |s1 - l1| == 0 && |s2 - l2| == 0 Computation point P is located from the beginning on
                // the direction of a specific segment (P coincides with P' and P'')
                if (std::abs(distances[index].s1 - distances[index].l1) < EPSILON_ZERO_OFFSET &&
                    std::abs(distances[index].s2 - distances[index].l2) < EPSILON_ZERO_OFFSET) {
                    //4. Option - Case 2: P is located on the segment from its right side
                    // s1 = -|s1|, s2 = -|s2|, l1 = -|l1|, l2 = -|l2|
                    if (distances[index].s2 < distances[index].s1) {
                        distances[index].s1 *= -1.0;
                        distances[index].s2 *= -1.0;
                        distances[index].l1 *= -1.0;
                        distances[index].l2 *= -1.0;
                    } else if (std::abs(distances[index].s2 - distances[index].s1) < EPSILON_ZERO_OFFSET) {
                        //4. Option - Case 1: P is located inside the segment (s2 == s1)
                        // s1 = -|s1|, s2 = |s2|, l1 = -|l1|, l2 = |l2|
                        distances[index].s1 *= -1.0;
                        distances[index].l1 *= -1.0;
                    }
                    //4. Option - Case 3: P is located on the segment from its left side
                    // s1 = |s1|, s2 = |s2|, l1 = |l1|, l2 = |l2| --> Nothing to do!
                } else {
                    const auto norm = euclideanNorm(segmentVectors[i][index]);
                    if (distances[index].s1 < norm && distances[index].s2 < norm) {
                        //1. Option: |s1| < |G_ij| && |s2| < |G_ij| Point P'' is situated inside the segment
                        // s1 = -|s1|, s2 = |s2|, l1 = |l1|, l2 = |l2|
                        distances[index].s1 *= -1.0;
                    } else if (distances[index].s2 < distances[index].s1) {
                        //2. Option: |s2| < |s1| Point P'' is on the right side of the segment
                        // s1 = -|s1|, s2 = -|s2|, l1 = |l1|, l2 = |l2|
                        distances[index].s1 *= -1.0;
                        distances[index].s2 *= -1.0;
                    }
                    //3. Option: |s1| < |s2| Point P'' is on the left side of the segment
                    // s1 = |s1|, s2 = |s2|, l1 = |l1|, l2 = |l2| --> Nothing to do!
                }
            }
            //endregion

            //region 1-12 Step: Compute the euclidian Norms of the vectors consisting of P and the vertices they are later used for determining the position of P in relation to the plane
            Array3 projectionPointVertexNorms{
                    euclideanNorm(orthogonalProjectionPointOnPlane - face[0]),
                    euclideanNorm(orthogonalProjectionPointOnPlane - face[1]),
                    euclideanNorm(orthogonalProjectionPointOnPlane - face[2]),
            };
            //endregion

            //region 1-13 Step: Compute the transcendental Expressions LN_pq and AN_pq
            std::array<TranscendentalExpression, 3> transcendentalExpressions{};
            for (unsigned int index = 0; index < 3; ++index) {
                // Computation of the norm of P' and segment endpoints
                // If the one of the norms == 0 then P' lies on the corresponding vertex and coincides with P''
                const auto r1Norm = projectionPointVertexNorms[(index + 1) % 3];
                const auto r2Norm = projectionPointVertexNorms[index];

                //Compute LN_pq according to (14)
                // If sigma_pq == 0 && either of the distances of P' to the two segment endpoints == 0 OR
                // the 1D and 3D distances are smaller than some EPSILON
                // then LN_pq can be set to zero
                if ((segmentNormalOrientations[index] == 0 && (r1Norm < EPSILON_ZERO_OFFSET || r2Norm < EPSILON_ZERO_OFFSET)) ||
                    (std::abs(distances[index].s1 + distances[index].s2) < EPSILON_ZERO_OFFSET &&
                     std::abs(distances[index].l1 + distances[index].l2) < EPSILON_ZERO_OFFSET)) {
                    transcendentalExpressions[index].ln = 0.0;
                } else {
                    //Implementation of
                    // log((s2_pq + l2_pq) / (s1_pq + l1_pq))
                    FloatType inner_num = distances[index].s2 + distances[index].l2;
                    FloatType inner_denom = distances[index].s1 + distances[index].l1;

                    if (inner_num <= 0.0 || inner_denom <= 0.0) {// TODO: figure out why, to avoid -inf and -nan
                        transcendentalExpressions[index].ln = 0.0;
                    } else {
                        transcendentalExpressions[index].ln = std::log(inner_num / inner_denom);
                    }
                }

                //Compute AN_pq according to (15)
                // If h_p == 0 or h_pq == 0 then AN_pq is zero, too (distances are always positive!)
                if (planeDistance < EPSILON_ZERO_OFFSET || segmentDistances[index] < EPSILON_ZERO_OFFSET) {
                    transcendentalExpressions[index].an = 0.0;
                } else {
                    //Implementation of:
                    // atan(h_p * s2_pq / h_pq * l2_pq) - atan(h_p * s1_pq / h_pq * l1_pq)

                    auto frac1 = (planeDistance * distances[index].s2) / (segmentDistances[index] * distances[index].l2);
                    auto frac2 = (planeDistance * distances[index].s1) / (segmentDistances[index] * distances[index].l1);

                    transcendentalExpressions[index].an = std::atan(frac1) - std::atan(frac2);
                }
            }
            //endregion

            //region 1-14 Step: Compute the singularities sing A and sing B if P' is located in the plane, on any vertex, or on one segment (G_pq)
            Singularity singularities{};

            do {
                // 1. Case: If all sigma_pq for a given plane p are 1.0 then P' lies inside the plane S_p
                bool allInside = true;
                for (unsigned int index = 0; index < 3; ++index) {
                    allInside &= segmentNormalOrientations[index] == 1;
                }
                if (allInside) {
                    //sing alpha = -2pi*h_p
                    singularities.a = -1.0 * PI2 * planeDistance;

                    //sing beta  = -2pi*sigma_p*N_p
                    singularities.b = normals[i] * (-1.0 * PI2 * planeNormalOrientation);
                    break;
                }

                // 2. Case: If sigma_pq == 0 AND norm(P' - v1) < norm(G_ij) && norm(P' - v2) < norm(G_ij) with G_ij
                // as the vector of v1 and v2
                // then P' is located on one line segment G_p of plane p, but not on any of its vertices
                bool anyOnLine = false;
                for (unsigned int index = 0; index < 3; ++index) {
                    if (segmentNormalOrientations[index] != 0) {
                        continue;
                    }
                    const auto segmentVectorNorm = euclideanNorm(segmentVectors[i][index]);
                    anyOnLine |= projectionPointVertexNorms[(index + 1) % 3] < segmentVectorNorm && projectionPointVertexNorms[index] < segmentVectorNorm;
                }

                if (anyOnLine) {
                    singularities.a = -1.0 * PI * planeDistance;                        //sing alpha = -pi*h_p
                    singularities.b = normals[i] * (-1.0 * PI * planeNormalOrientation);//sing beta  = -pi*sigma_p*N_p
                    break;
                }

                // 3. Case If sigma_pq == 0 AND norm(P' - v1) < 0 || norm(P' - v2) < 0
                // then P' is located at one of G_p's vertices
                bool anyAtVertex = false;

                for (unsigned int index = 0; index < 3; ++index) {
                    if (segmentNormalOrientations[index] != 0) {
                        continue;
                    }

                    auto r1Norm = projectionPointVertexNorms[(index + 1) % 3];
                    auto r2Norm = projectionPointVertexNorms[index];

                    if (!(r1Norm < EPSILON_ZERO_OFFSET || r2Norm < EPSILON_ZERO_OFFSET)) {
                        continue;
                    }

                    const Array3 &g1 = r1Norm == 0.0 ? segmentVectors[i][index] : segmentVectors[i][(index - 1 + 3) % 3];
                    const Array3 &g2 = r1Norm == 0.0 ? segmentVectors[i][(index + 1) % 3] : segmentVectors[i][index];
                    // theta = arcos((G_2 * -G_1) / (|G_2| * |G_1|))
                    const FloatType gdot = dot(g1 * -1.0, g2);
                    const FloatType theta = gdot == 0.0 ? PI_2 : std::acos(gdot / (euclideanNorm(g1) * euclideanNorm(g2)));

                    singularities.a = -1.0 * theta * planeDistance;                        //sing alpha = -theta*h_p
                    singularities.b = normals[i] * (-1.0 * theta * planeNormalOrientation);//sing beta  = -theta*sigma_p*N_p
                    anyAtVertex = true;
                    break;
                }

                if (!anyAtVertex) {
                    //4. Case Otherwise P' is located outside the plane S_p and then the singularity equals zero
                    singularities.a = 0.0;
                    singularities.b = Array3{};
                }
            } while (false);
            //endregion

            //region 2. Step: Compute Sum 1 used for potential and acceleration (first derivative)
            // sum over: sigma_pq * h_pq * LN_pq
            // --> Equation 11/12 the first summation in the brackets
            FloatType sum1PotentialAcceleration = 0.0;
            for (unsigned int index = 0; index < 3; ++index)
                sum1PotentialAcceleration += segmentNormalOrientations[index] * segmentDistances[index] * transcendentalExpressions[index].ln;
            //endregion

            //region 3. Step: Compute Sum 1 used for the gradiometric tensor (second derivative)
            // sum over: n_pq * LN_pq
            // --> Equation 13 the first summation in the brackets
            Array3 sum1Tensor{};
            for (unsigned int index = 0; index < 3; ++index)
                sum1Tensor = sum1Tensor + segmentNormals[i][index] * transcendentalExpressions[index].ln;
            //endregion

            //region 4. Step: Compute Sum 2 which is the same for every result parameter
            FloatType sum2 = 0.0;
            for (unsigned int index = 0; index < 3; ++index)
                sum2 += segmentNormalOrientations[index] * transcendentalExpressions[index].an;

            //    if (isCriticalDifference(planeDistance, sum2)) {
            //          WARN()
            //    }
            //endregion

            //region 5. Step: Sum for potential and acceleration
            // consisting of: sum1 + h_p * sum2 + sing A
            // --> Equation 11/12 the total sum of the brackets
            const FloatType planeSumPotentialAcceleration = sum1PotentialAcceleration + planeDistance * sum2 + singularities.a;
            //endregion

            //region 6. Step: Sum for tensor
            // consisting of: sum1 + sigma_p * N_p * sum2 + sing B
            // --> Equation 13 the total sum of the brackets
            const Array3 subSum = (sum1Tensor + (normals[i] * (planeNormalOrientation * sum2))) + singularities.b;
            // first component: trivial case Vxx, Vyy, Vzz --> just N_p * subSum
            // 00, 11, 22 --> xx, yy, zz with x as 0, y as 1, z as 2
            const Array3 first = normals[i] * subSum;
            // second component: reordering required to build Vxy, Vxz, Vyz
            // 01, 02, 12 --> xy, xz, yz with x as 0, y as 1, z as 2
            const Array3 reorderedNp = {normals[i][0], normals[i][0], normals[i][1]};
            const Array3 reorderedSubSum = {subSum[1], subSum[2], subSum[2]};
            const Array3 second = reorderedNp * reorderedSubSum;
            //endregion

            //region 7. Step: Multiply with prefix
            results[i] = {
                    // Equation (11): sigma_p * h_p * sum
                    planeNormalOrientation * planeDistance * planeSumPotentialAcceleration,

                    // Equation (12): N_p * sum
                    normals[i] * planeSumPotentialAcceleration,

                    // Equation (13): already done above, just concat the two components for later summation
                    concat(first, second)};
            //endregion
        });

        // 8. Step: Reduce the per-face contributions to the final result
        GravityModelResult result = std::reduce(std::execution::par_unseq, _results.begin(), _results.end(),
                                                GravityModelResult{}, std::plus<>{});

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
        const size_t faceCount = _faces.size();
        const Array3 *vertices = _vertices.data();
        const IndexArray3 *faces = _faces.data();
        Array3 *normals = _normals.data();
        Array3Triplet *segmentVectors = _segmentVectors.data();
        Array3Triplet *segmentNormals = _segmentNormals.data();

        const auto indices = std::views::iota(size_t{0}, faceCount);
        std::for_each(std::execution::par_unseq, indices.begin(), indices.end(), [=](const size_t i) {
            Array3Triplet Face = {
                    vertices[faces[i][0]],
                    vertices[faces[i][1]],
                    vertices[faces[i][2]]};

            Array3Triplet SV{Face[1] - Face[0], Face[2] - Face[1], Face[0] - Face[2]};
            segmentVectors[i] = SV;

            Array3 Normal = normal(SV[0], SV[1]);
            normals[i] = Normal;
            segmentNormals[i] = {
                    normal(SV[0], Normal),
                    normal(SV[1], Normal),
                    normal(SV[2], Normal),
            };
        });

        _initialized = true;
    }

    std::vector<Array3> _normals;
    std::vector<Array3Triplet> _segmentVectors;
    std::vector<Array3Triplet> _segmentNormals;
    std::vector<GravityModelResult> _results;
};

std::unique_ptr<GravityEvaluableBase> create_gravity_evaluable(
        const std::vector<Array3> &Vertices,
        const std::vector<IndexArray3> &Faces,
        double density) {
    return std::make_unique<GravityEvaluable>(Vertices, Faces, density);
}
