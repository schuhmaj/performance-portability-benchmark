#include "polyhedralGravity/PolyhedralGravityDefinitions.h"

#include <sycl/sycl.hpp>

GlobalResources::GlobalResources(int &argc, char *argv[]) {
}
GlobalResources::~GlobalResources() = default;

struct GravityModelResultAcpp {
    FloatType data[10];

    GravityModelResultAcpp operator+(const GravityModelResultAcpp &other) const {
        GravityModelResultAcpp result = *this;
        for (int i = 0; i < 10; i++) { result.data[i] += other.data[i]; }
        return result;
    }

    GravityModelResultAcpp &operator+=(const GravityModelResultAcpp &other) {
        for (int i = 0; i < 10; i++) { data[i] += other.data[i]; }
        return *this;
    }
};

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
        _segmentVectors = sycl::malloc_device<Array3Triplet>(numFaces, queue);
        _segmentNormals = sycl::malloc_device<Array3Triplet>(numFaces, queue);
        _result_device = sycl::malloc_device<GravityModelResultAcpp>(1, queue);

        queue.copy(_vertices.data(), _vertices_device, numVertices);
        queue.copy(_faces.data(), _faces_device, numFaces);
    }

    ~GravityEvaluable() override {
        sycl::free(_vertices_device, queue);
        sycl::free(_faces_device, queue);
        sycl::free(_normals, queue);
        sycl::free(_segmentVectors, queue);
        sycl::free(_segmentNormals, queue);
        sycl::free(_result_device, queue);
    }

    GravityModelResult evaluate(const Array3 &Point) override {
        if (!_initialized) init();

        const size_t numFaces = _faces.size();
        const Array3 point = Point;

        // Capture the device pointers as plain values for the kernel.
        const Array3 *V = _vertices_device;
        const IndexArray3 *F = _faces_device;
        const Array3 *N = _normals;
        const Array3Triplet *SV = _segmentVectors;
        const Array3Triplet *SN = _segmentNormals;

        // The result buffer is allocated once (see constructor) and reused across
        // calls - allocating/freeing USM on every evaluate() is a heavyweight,
        // synchronizing operation that dominates the (small) per-point kernel.
        // Reset to the reduction identity; the in-order queue keeps this ordered
        // before the kernel below.
        queue.memset(_result_device, 0, sizeof(GravityModelResultAcpp));

        queue.submit([&](sycl::handler &h) {
             auto reduction = sycl::reduction(_result_device, GravityModelResultAcpp{},
                                              [](const GravityModelResultAcpp &a, const GravityModelResultAcpp &b) {
                                                  return a + b;// Custom reduction operation
                                              });

             h.parallel_for(sycl::range<1>(numFaces), reduction, [=](const sycl::id<1> &id, auto &reducer) {
                 const size_t i = id[0];
                 Array3Triplet face = {
                         V[F[i][0]] - point,
                         V[F[i][1]] - point,
                         V[F[i][2]] - point};

                 int planeNormalOrientation = sgn(dot(N[i], face[0]));

                 HessianPlane hessianPlane{};
                 {
                     constexpr Array3 origin{0.0, 0.0, 0.0};
                     const auto crossProduct = cross(face[0] - face[1], face[0] - face[2]);
                     const auto res = crossProduct * (origin - face[0]);
                     const auto d = res[0] + res[1] + res[2];

                     hessianPlane = {crossProduct[0], crossProduct[1], crossProduct[2], d};
                 }

                 auto planeDistance = std::abs(hessianPlane.d / std::sqrt(
                                                                        hessianPlane.a * hessianPlane.a + hessianPlane.b * hessianPlane.b +
                                                                        hessianPlane.c * hessianPlane.c));

                 Array3 orthogonalProjectionPointOnPlane = N[i] * planeDistance;
                 {
                     Array3 intersections = {hessianPlane.a == 0.0 ? static_cast<FloatType>(0.0) : hessianPlane.d / hessianPlane.a,
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

                 std::array<int, 3> segmentNormalOrientations{};
                 for (unsigned int index = 0; index < 3; ++index) {
                     segmentNormalOrientations[index] = -sgn(
                             dot(SN[i][index], orthogonalProjectionPointOnPlane - face[index]));
                 }

                 Array3Triplet orthogonalProjectionPointsOnSegmentsForPlane{};
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
                         Matrix columnMatrix = transpose({matrixRow1, matrixRow2, matrixRow3});

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

                 Array3 segmentDistances{};
                 for (unsigned int index = 0; index < 3; ++index) {
                     segmentDistances[index] = euclideanNorm(
                             orthogonalProjectionPointsOnSegmentsForPlane[index] - orthogonalProjectionPointOnPlane);
                 }

                 std::array<Distance, 3> distances{};
                 for (unsigned int index = 0; index < 3; ++index) {
                     distances[index].l1 = euclideanNorm(face[index]);
                     distances[index].l2 = euclideanNorm(face[(index + 1) % 3]);

                     distances[index].s1 = euclideanNorm(orthogonalProjectionPointsOnSegmentsForPlane[index] - face[index]);
                     distances[index].s2 = euclideanNorm(
                             orthogonalProjectionPointsOnSegmentsForPlane[index] - face[(index + 1) % 3]);

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
                         const auto norm = euclideanNorm(SV[i][index]);
                         if (distances[index].s1 < norm && distances[index].s2 < norm) {
                             distances[index].s1 *= -1.0;
                         } else if (distances[index].s2 < distances[index].s1) {
                             distances[index].s1 *= -1.0;
                             distances[index].s2 *= -1.0;
                         }
                     }
                 }

                 Array3 projectionPointVertexNorms{
                         euclideanNorm(orthogonalProjectionPointOnPlane - face[0]),
                         euclideanNorm(orthogonalProjectionPointOnPlane - face[1]),
                         euclideanNorm(orthogonalProjectionPointOnPlane - face[2]),
                 };
                 std::array<TranscendentalExpression, 3> transcendentalExpressions{};
                 for (unsigned int index = 0; index < 3; ++index) {
                     const auto r1Norm = projectionPointVertexNorms[(index + 1) % 3];
                     const auto r2Norm = projectionPointVertexNorms[index];

                     if ((segmentNormalOrientations[index] == 0 &&
                          (r1Norm < EPSILON_ZERO_OFFSET || r2Norm < EPSILON_ZERO_OFFSET)) ||
                         (std::abs(distances[index].s1 + distances[index].s2) < EPSILON_ZERO_OFFSET &&
                          std::abs(distances[index].l1 + distances[index].l2) < EPSILON_ZERO_OFFSET)) {
                         transcendentalExpressions[index].ln = 0.0;
                     } else {
                         FloatType inner_num = distances[index].s2 + distances[index].l2;
                         FloatType inner_denom = distances[index].s1 + distances[index].l1;

                         if (inner_num <= 0.0 || inner_denom <= 0.0) {
                             transcendentalExpressions[index].ln = 0.0;
                         } else {
                             transcendentalExpressions[index].ln = std::log(inner_num / inner_denom);
                         }
                     }

                     if (planeDistance < EPSILON_ZERO_OFFSET || segmentDistances[index] < EPSILON_ZERO_OFFSET) {
                         transcendentalExpressions[index].an = 0.0;
                     } else {
                         FloatType frac1 =
                                 (planeDistance * distances[index].s2) / (segmentDistances[index] * distances[index].l2);
                         FloatType frac2 =
                                 (planeDistance * distances[index].s1) / (segmentDistances[index] * distances[index].l1);

                         transcendentalExpressions[index].an = std::atan(frac1) - std::atan(frac2);
                     }
                 }

                 Singularity singularities{};

                 do {
                     bool allInside = true;
                     for (unsigned int index = 0; index < 3; ++index) {
                         allInside &= segmentNormalOrientations[index] == 1;
                     }
                     if (allInside) {
                         singularities.a = -1.0 * PI2 * planeDistance;
                         singularities.b = N[i] * (-1.0 * PI2 * planeNormalOrientation);
                         break;
                     }

                     bool anyOnLine = false;
                     for (unsigned int index = 0; index < 3; ++index) {
                         if (segmentNormalOrientations[index] != 0) {
                             continue;
                         }
                         const auto segmentVectorNorm = euclideanNorm(SV[i][index]);
                         anyOnLine |= projectionPointVertexNorms[(index + 1) % 3] < segmentVectorNorm &&
                                      projectionPointVertexNorms[index] < segmentVectorNorm;
                     }

                     if (anyOnLine) {
                         singularities.a = -1.0 * PI * planeDistance;                  //sing alpha = -pi*h_p
                         singularities.b = N[i] * (-1.0 * PI * planeNormalOrientation);//sing beta  = -pi*sigma_p*N_p
                         break;
                     }

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

                         const Array3 &g1 = r1Norm == 0.0 ? SV[i][index] : SV[i][(index - 1 + 3) % 3];
                         const Array3 &g2 = r1Norm == 0.0 ? SV[i][(index + 1) % 3] : SV[i][index];

                         const FloatType gdot = dot(g1 * -1.0, g2);
                         const FloatType theta = gdot == 0.0 ? PI_2 : std::acos(gdot / (euclideanNorm(g1) * euclideanNorm(g2)));

                         singularities.a = -1.0 * theta * planeDistance;
                         singularities.b = N[i] * (-1.0 * theta * planeNormalOrientation);
                         anyAtVertex = true;
                         break;
                     }

                     if (!anyAtVertex) {
                         singularities.a = 0.0;
                         singularities.b = {0.0, 0.0, 0.0};
                     }
                 } while (false);

                 FloatType sum1PotentialAcceleration = 0.0;
                 for (unsigned int index = 0; index < 3; ++index)
                     sum1PotentialAcceleration += segmentNormalOrientations[index] * segmentDistances[index] *
                                                  transcendentalExpressions[index].ln;

                 Array3 sum1Tensor{0.0, 0.0, 0.0};
                 for (unsigned int index = 0; index < 3; ++index)
                     sum1Tensor = sum1Tensor + SN[i][index] * transcendentalExpressions[index].ln;

                 FloatType sum2 = 0.0;
                 for (unsigned int index = 0; index < 3; ++index)
                     sum2 += segmentNormalOrientations[index] * transcendentalExpressions[index].an;

                 const FloatType planeSumPotentialAcceleration =
                         sum1PotentialAcceleration + planeDistance * sum2 + singularities.a;

                 const Array3 subSum = (sum1Tensor + (N[i] * (planeNormalOrientation * sum2))) + singularities.b;

                 const Array3 first = N[i] * subSum;

                 const Array3 reorderedNp = {N[i][0], N[i][0], N[i][1]};
                 const Array3 reorderedSubSum = {subSum[1], subSum[2], subSum[2]};
                 const Array3 second = reorderedNp * reorderedSubSum;

                 const Array3 acc = N[i] * planeSumPotentialAcceleration;

                 GravityModelResultAcpp r2{};
                 r2.data[0] = planeNormalOrientation * planeDistance * planeSumPotentialAcceleration;
                 r2.data[1] = acc[0];
                 r2.data[2] = acc[1];
                 r2.data[3] = acc[2];

                 r2.data[4] = first[0];
                 r2.data[5] = first[1];
                 r2.data[6] = first[2];

                 r2.data[7] = second[0];
                 r2.data[8] = second[1];
                 r2.data[9] = second[2];

                 reducer.combine(r2);
             });
         });

        GravityModelResultAcpp acpp_result{};
        queue.copy(_result_device, &acpp_result, 1).wait();

        GravityModelResult result{};
        result.potential = acpp_result.data[0];
        result.acceleration[0] = acpp_result.data[1];
        result.acceleration[1] = acpp_result.data[2];
        result.acceleration[2] = acpp_result.data[3];
        for (int i = 0; i < 6; ++i) {
            result.gradiometricTensor.data[i] = acpp_result.data[i + 4];
        }

        const double prefix = GRAVITATIONAL_CONSTANT * _density;

        result.potential = (result.potential * prefix) / 2.0;
        result.acceleration = result.acceleration * (-1.0 * prefix);
        result.gradiometricTensor = result.gradiometricTensor * prefix;
        return result;
    }

private:
    void init() {
        const size_t numFaces = _faces.size();

        const Array3 *V = _vertices_device;
        const IndexArray3 *F = _faces_device;
        Array3 *normals = _normals;
        Array3Triplet *segmentVectors = _segmentVectors;
        Array3Triplet *segmentNormals = _segmentNormals;

        queue.submit([&](sycl::handler &h) {
             h.parallel_for(sycl::range<1>(numFaces), [=](const sycl::id<1> &id) {
                 const size_t i = id[0];
                 Array3Triplet Face = {V[F[i][0]], V[F[i][1]], V[F[i][2]]};
                 Array3Triplet SegV = {Face[1] - Face[0], Face[2] - Face[1], Face[0] - Face[2]};
                 Array3 Normal = normal(SegV[0], SegV[1]);

                 segmentVectors[i] = SegV;
                 normals[i] = Normal;
                 segmentNormals[i] = {
                         normal(SegV[0], Normal),
                         normal(SegV[1], Normal),
                         normal(SegV[2], Normal),
                 };
             });
         }).wait();

        _initialized = true;
    }

    sycl::queue queue;

    Array3 *_vertices_device = nullptr;
    IndexArray3 *_faces_device = nullptr;

    Array3 *_normals = nullptr;
    Array3Triplet *_segmentVectors = nullptr;
    Array3Triplet *_segmentNormals = nullptr;

    // Reused across evaluate() calls to avoid per-call USM allocation overhead.
    GravityModelResultAcpp *_result_device = nullptr;
};

std::unique_ptr<GravityEvaluableBase> create_gravity_evaluable(
        const std::vector<Array3> &Vertices,
        const std::vector<IndexArray3> &Faces,
        double density) {
    return std::make_unique<GravityEvaluable>(Vertices, Faces, density);
}
