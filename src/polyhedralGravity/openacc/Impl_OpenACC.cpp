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
          _normals(allocateOpenACC<Array3>(_faces.size())),
          _segmentVectors(allocateOpenACC<Array3Triplet>(_faces.size())),
          _segmentNormals(allocateOpenACC<Array3Triplet>(_faces.size())),
          _resultsDevice(allocateOpenACC<GravityModelResult>(_faces.size())), _results_cpu(_faces.size()) {
        acc_memcpy_to_device(_facesDevice, (void *) (_faces.data()), sizeof(IndexArray3) * _faces.size());
        acc_memcpy_to_device(_verticesDevice, (void *) _vertices.data(), sizeof(Array3) * _vertices.size());
    }

    ~GravityEvaluable() override {
        acc_free(_facesDevice);
        acc_free(_verticesDevice);
        acc_free(_normals);
        acc_free(_segmentVectors);
        acc_free(_segmentNormals);
        acc_free(_resultsDevice);
    }

    GravityModelResult evaluate(const Array3 &Point) override {
        if (!_initialized) init();

        size_t face_count = _faces.size();
#pragma acc parallel loop
        for (size_t i = 0; i < face_count; ++i) {
            Array3Triplet face = {
                    _verticesDevice[_facesDevice[i][0]] - Point,
                    _verticesDevice[_facesDevice[i][1]] - Point,
                    _verticesDevice[_facesDevice[i][2]] - Point};

            int planeNormalOrientation = sgn(dot(_normals[i], face[0]));

            HessianPlane hessianPlane{};
            {
                constexpr Array3 origin{0.0, 0.0, 0.0};
                const auto crossProduct = cross(face[0] - face[1], face[0] - face[2]);
                const auto res = crossProduct * (origin - face[0]);
                const auto d = res[0] + res[1] + res[2];

                hessianPlane = {crossProduct[0], crossProduct[1], crossProduct[2], d};
            }

            auto planeDistance = std::abs(hessianPlane.d / std::sqrt(
                                                                   hessianPlane.a * hessianPlane.a + hessianPlane.b * hessianPlane.b + hessianPlane.c * hessianPlane.c));

            Array3 orthogonalProjectionPointOnPlane = _normals[i] * planeDistance;
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
                segmentNormalOrientations[index] = -sgn(dot(_segmentNormals[i][index], orthogonalProjectionPointOnPlane - face[index]));
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
                segmentDistances[index] = euclideanNorm(orthogonalProjectionPointsOnSegmentsForPlane[index] - orthogonalProjectionPointOnPlane);
            }

            std::array<Distance, 3> distances{};
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
                    const auto norm = euclideanNorm(_segmentVectors[i][index]);
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

                if ((segmentNormalOrientations[index] == 0 && (r1Norm < EPSILON_ZERO_OFFSET || r2Norm < EPSILON_ZERO_OFFSET)) ||
                    (std::abs(distances[index].s1 + distances[index].s2) < EPSILON_ZERO_OFFSET &&
                     std::abs(distances[index].l1 + distances[index].l2) < EPSILON_ZERO_OFFSET)) {
                    transcendentalExpressions[index].ln = 0.0;
                } else {
                    FloatType inner_num = distances[index].s2 + distances[index].l2;
                    FloatType inner_denom = distances[index].s1 + distances[index].l1;

                    if (inner_num <= 0.0 || inner_denom <= 0.0) {// TODO: figure out why, to avoid -inf and -nan
                        transcendentalExpressions[index].ln = 0.0;
                    } else {
                        transcendentalExpressions[index].ln = std::log(inner_num / inner_denom);
                    }
                }

                if (planeDistance < EPSILON_ZERO_OFFSET || segmentDistances[index] < EPSILON_ZERO_OFFSET) {
                    transcendentalExpressions[index].an = 0.0;
                } else {
                    auto frac1 = (planeDistance * distances[index].s2) / (segmentDistances[index] * distances[index].l2);
                    auto frac2 = (planeDistance * distances[index].s1) / (segmentDistances[index] * distances[index].l1);

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

                    singularities.b = _normals[i] * (-1.0 * PI2 * planeNormalOrientation);
                    break;
                }

                bool anyOnLine = false;
                for (unsigned int index = 0; index < 3; ++index) {
                    if (segmentNormalOrientations[index] != 0) {
                        continue;
                    }
                    const auto segmentVectorNorm = euclideanNorm(_segmentVectors[i][index]);
                    anyOnLine |= projectionPointVertexNorms[(index + 1) % 3] < segmentVectorNorm && projectionPointVertexNorms[index] < segmentVectorNorm;
                }

                if (anyOnLine) {
                    singularities.a = -1.0 * PI * planeDistance;
                    singularities.b = _normals[i] * (-1.0 * PI * planeNormalOrientation);
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

                    const Array3 &g1 = r1Norm == 0.0 ? _segmentVectors[i][index] : _segmentVectors[i][(index - 1 + 3) % 3];
                    const Array3 &g2 = r1Norm == 0.0 ? _segmentVectors[i][(index + 1) % 3] : _segmentVectors[i][index];

                    const FloatType gdot = dot(g1 * -1.0, g2);
                    const FloatType theta = gdot == 0.0 ? PI_2 : std::acos(gdot / (euclideanNorm(g1) * euclideanNorm(g2)));

                    singularities.a = -1.0 * theta * planeDistance;
                    singularities.b = _normals[i] * (-1.0 * theta * planeNormalOrientation);
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
                sum1PotentialAcceleration += segmentNormalOrientations[index] * segmentDistances[index] * transcendentalExpressions[index].ln;

            Array3 sum1Tensor{0.0, 0.0, 0.0};
            for (unsigned int index = 0; index < 3; ++index)
                sum1Tensor = sum1Tensor + _segmentNormals[i][index] * transcendentalExpressions[index].ln;

            FloatType sum2 = 0.0;
            for (unsigned int index = 0; index < 3; ++index)
                sum2 += segmentNormalOrientations[index] * transcendentalExpressions[index].an;

            const FloatType planeSumPotentialAcceleration = sum1PotentialAcceleration + planeDistance * sum2 + singularities.a;

            const Array3 subSum = (sum1Tensor + (_normals[i] * (planeNormalOrientation * sum2))) + singularities.b;
            const Array3 first = _normals[i] * subSum;
            const Array3 reorderedNp = {_normals[i][0], _normals[i][0], _normals[i][1]};
            const Array3 reorderedSubSum = {subSum[1], subSum[2], subSum[2]};
            const Array3 second = reorderedNp * reorderedSubSum;

            _resultsDevice[i] = {
                    planeNormalOrientation * planeDistance * planeSumPotentialAcceleration,
                    _normals[i] * planeSumPotentialAcceleration,
                    concat(first, second)};
        }

        acc_memcpy_from_device(_results_cpu.data(), _resultsDevice, sizeof(_results_cpu[0]) * _results_cpu.size());

        GravityModelResult result{};

        for (size_t i = 0; i < face_count; ++i) {
            result += _results_cpu[i];
        }

        const double prefix = GRAVITATIONAL_CONSTANT * _density;

        result.potential = (result.potential * prefix) / 2.0;
        result.acceleration = result.acceleration * (-1.0 * prefix);
        result.gradiometricTensor = result.gradiometricTensor * prefix;
        return result;
    }

private:
    void init() {
        size_t face_count = _faces.size();
#pragma acc parallel loop
        for (size_t i = 0; i < face_count; ++i) {
            Array3Triplet Face = {
                    _verticesDevice[_facesDevice[i][0]],
                    _verticesDevice[_facesDevice[i][1]],
                    _verticesDevice[_facesDevice[i][2]]};

            Array3Triplet SV{Face[1] - Face[0], Face[2] - Face[1], Face[0] - Face[2]};
            _segmentVectors[i] = SV;

            Array3 Normal = normal(SV[0], SV[1]);
            _normals[i] = Normal;
            _segmentNormals[i] = {
                    normal(SV[0], Normal),
                    normal(SV[1], Normal),
                    normal(SV[2], Normal),
            };
        }
        _initialized = true;
    }
    IndexArray3 *_facesDevice;
    Array3 *_verticesDevice;

    Array3 *_normals;
    Array3Triplet *_segmentVectors;
    Array3Triplet *_segmentNormals;
    GravityModelResult *_resultsDevice;

    std::vector<GravityModelResult> _results_cpu;
};

std::unique_ptr<GravityEvaluableBase> create_gravity_evaluable(
        const std::vector<Array3> &Vertices,
        const std::vector<IndexArray3> &Faces,
        double density) {
    return std::make_unique<GravityEvaluable>(Vertices, Faces, density);
}
