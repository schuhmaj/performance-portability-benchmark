#version 460
#extension GL_EXT_shader_atomic_float : require

#ifndef FloatType
#define FloatType double
#define FloatType3 double3
#define FloatType4 double4
#define FloatTypeM dmat3
#endif

struct Result {
    FloatType4 res;
    FloatType4 first;
    FloatType4 second;
    FloatType4 _padding;
};

StructuredBuffer<FloatType3> vertices;
StructuredBuffer<uvec3> faces;

RWStructuredBuffer<FloatType3> normals;
RWStructuredBuffer<FloatType3> segmentVectors;
RWStructuredBuffer<FloatType3> segmentNormals;

RWStructuredBuffer<Result> results;

struct Params {
    FloatType3 point;
    uint num_faces;
    uint init_done;
};

[[vk::push_constant]] ConstantBuffer<Params> params;

struct Distance {
    FloatType l1;
    FloatType l2;
    FloatType s1;
    FloatType s2;
};

struct TranscendentalExpression {
    FloatType ln;
    FloatType an;
};

struct Singularity {
    FloatType a;
    FloatType3 b;
};

const FloatType EPSILON_ZERO_OFFSET = 1e-14;
const FloatType PI = 3.1415926535897932384626433832795028841971693993751058209749445923;
const FloatType PI2 = 6.2831853071795864769252867665590057683943387987502116419498891846;
const FloatType PI_2 = 1.5707963267948966192313216916397514420985846996875529104874722961;

int sgn(FloatType val) {
    return val < -EPSILON_ZERO_OFFSET ? -1 : val > EPSILON_ZERO_OFFSET ? 1 : 0;
}

FloatType3 normal(FloatType3 first, FloatType3 second) {
    return normalize(cross(first, second));
}

FloatType custom_dot(FloatType3 lhs, FloatType3 rhs) {
    return FloatType(FloatType(lhs.x) *  FloatType(rhs.x) + FloatType(lhs.y) * FloatType(rhs.y) + FloatType(lhs.z) * FloatType(rhs.z));
}

FloatType euclideanNorm(FloatType3 a) {
    return FloatType(sqrt(custom_dot(a, a)));
}

FloatType custom_log(FloatType x) {
    #ifdef TARGET_CUDA
    return log(x);
    #else
    return log(float(x));
    #endif
}

FloatType custom_atan(FloatType x) {
    #ifdef TARGET_CUDA
    return atan(x);
    #else
    return atan(float(x));
    #endif
}

FloatType custom_acos(FloatType x) {
    #ifdef TARGET_CUDA
    return acos(x);
    #else
    return acos(float(x));
    #endif
}

FloatType compute_singularities(
    uint face_index,
    ivec3 segmentNormalOrientations,
    FloatType3 projectionPointVertexNorms,
) {
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
        FloatType segmentVectorNorm = length(segmentVectors[face_index * 3 + index]);
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

        FloatType3 g1 = r1Norm == 0.0 ? segmentVectors[face_index * 3 + index] : segmentVectors[face_index * 3 + (index - 1 + 3) % 3];
        FloatType3 g2 = r1Norm == 0.0 ? segmentVectors[face_index * 3 + (index + 1) % 3] : segmentVectors[face_index * 3 + index];

        FloatType gdot = dot(g1 * -1.0, g2);
        FloatType theta = gdot == 0.0 ? PI_2 : custom_acos(gdot / (length(g1) * length(g2)));
        return theta;
    }

    return 0.0;
}

FloatType det_custom(FloatTypeM matrix) {
    return matrix[0][0] * matrix[1][1] * matrix[2][2] +
           matrix[0][1] * matrix[1][2] * matrix[2][0] +
           matrix[0][2] * matrix[1][0] * matrix[2][1] -
           matrix[0][2] * matrix[1][1] * matrix[2][0] -
           matrix[0][0] * matrix[1][2] * matrix[2][1] -
           matrix[0][1] * matrix[1][0] * matrix[2][2];
}

void do_init(uint index) {
    FloatType3 Face[3] = {
        vertices[faces[index][0]],
        vertices[faces[index][1]],
        vertices[faces[index][2]]
    };

    segmentVectors[index * 3 + 0] = Face[1] - Face[0];
    segmentVectors[index * 3 + 1] = Face[2] - Face[1];
    segmentVectors[index * 3 + 2] = Face[0] - Face[2];

    normals[index] = normal(segmentVectors[index* 3 + 0], segmentVectors[index * 3 + 1]);

    segmentNormals[index * 3 + 0] = normal(segmentVectors[index * 3 + 0], normals[index]);
    segmentNormals[index * 3 + 1] = normal(segmentVectors[index * 3 + 1], normals[index]);
    segmentNormals[index * 3 + 2] = normal(segmentVectors[index * 3 + 2], normals[index]);
}

[shader("compute")]
[numthreads(32, 1, 1)]
void computeMain(uint3 threadIdx : SV_DispatchThreadID, uint3 groupThreadIdx : SV_GroupThreadID, uint3 groupIdx : SV_GroupID) {
    uint face_index = threadIdx.x;

    if (face_index >= params.num_faces) {
        return;
    }

    if (params.init_done == 0) {
        do_init(face_index);
    }

    FloatType3 face[3] = {
        vertices[faces[face_index][0]] - params.point,
        vertices[faces[face_index][1]] - params.point,
        vertices[faces[face_index][2]] - params.point
    };

    int planeNormalOrientation = sgn(dot(normals[face_index], face[0]));

    FloatType4 hessianPlane = {};
    {
        FloatType3 origin = {0.0, 0.0, 0.0};
        FloatType3 crossP =  cross(face[0] - face[1], face[0] - face[2]);
        FloatType3 res = (origin - face[0]) * crossP;
        FloatType d = res.x + res.y + res.z;

        hessianPlane.xyz = crossP;
        hessianPlane.w = d;
    }


    FloatType planeDistance = abs(hessianPlane.w / sqrt(hessianPlane.x * hessianPlane.x + hessianPlane.y * hessianPlane.y + hessianPlane.z * hessianPlane.z));
    FloatType3 orthogonalProjectionPointOnPlane = normals[face_index] * planeDistance;

    {
        FloatType3 intersections = FloatType3(
            hessianPlane.x == 0.0 ? 0.0 : hessianPlane.w / hessianPlane.x,
            hessianPlane.y == 0.0 ? 0.0 : hessianPlane.w / hessianPlane.y,
            hessianPlane.z == 0.0 ? 0.0 : hessianPlane.w / hessianPlane.z
        );

        for (uint index = 0; index < 3; ++index) {
            if (intersections[index] < 0) {
                orthogonalProjectionPointOnPlane[index] = abs(orthogonalProjectionPointOnPlane[index]);
            } else {
                if (orthogonalProjectionPointOnPlane[index] > 0) {
                    orthogonalProjectionPointOnPlane[index] = -orthogonalProjectionPointOnPlane[index];
                } else {
                    orthogonalProjectionPointOnPlane[index] = orthogonalProjectionPointOnPlane[index];
                }
            }
        }
    }

    ivec3 segmentNormalOrientations = {};
    for (uint index = 0; index < 3; ++index) {
        FloatType inner = custom_dot(segmentNormals[face_index * 3 + index], orthogonalProjectionPointOnPlane - face[index]);
        segmentNormalOrientations[index] = -sgn(inner);
    }

    FloatType3 orthogonalProjectionPointsOnSegmentsForPlane[3];
    for (uint index = 0; index < 3; ++index) {
        if (segmentNormalOrientations[index] == 0) {
            orthogonalProjectionPointsOnSegmentsForPlane[index] = orthogonalProjectionPointOnPlane;
        } else {
            FloatType3 vertex1 = face[index];
            FloatType3 vertex2 = face[(index + 1) % 3];

            FloatType3 matrixRow1 = vertex2 - vertex1;
            FloatType3 matrixRow2 = cross(vertex1 - orthogonalProjectionPointOnPlane, matrixRow1);
            FloatType3 matrixRow3 = cross(matrixRow2, matrixRow1);

            FloatType3 d = {
                dot(matrixRow1, orthogonalProjectionPointOnPlane),
                dot(matrixRow2, orthogonalProjectionPointOnPlane),
                dot(matrixRow3, vertex1),
            };

            FloatTypeM m = FloatTypeM(matrixRow1, matrixRow2, matrixRow3);
            FloatTypeM columnMatrix = transpose(m);

            FloatType determinant_ = det_custom(columnMatrix);

            if (determinant_ != 0.0) {
                FloatType3 res = {
                    det_custom(FloatTypeM(d, columnMatrix[1], columnMatrix[2])),
                    det_custom(FloatTypeM(columnMatrix[0], d, columnMatrix[2])),
                    det_custom(FloatTypeM(columnMatrix[0], columnMatrix[1], d)),
                };
                orthogonalProjectionPointsOnSegmentsForPlane[index] = res / determinant_;
            }
        }
    }

    FloatType segmentDistances[3];
    for (uint index = 0; index < 3; ++index) {
        segmentDistances[index] = length(orthogonalProjectionPointsOnSegmentsForPlane[index] - orthogonalProjectionPointOnPlane);
    }

    FloatType4 distances[3];
    for (uint index = 0; index < 3; ++index) {
        distances[index].x = length(face[index]);
        distances[index].y = length(face[(index + 1) % 3]);

        distances[index].z = length(orthogonalProjectionPointsOnSegmentsForPlane[index] - face[index]);
        distances[index].w = length(orthogonalProjectionPointsOnSegmentsForPlane[index] - face[(index + 1) % 3]);

        if (abs(distances[index].z - distances[index].x) < EPSILON_ZERO_OFFSET &&
            abs(distances[index].w - distances[index].y) < EPSILON_ZERO_OFFSET) {

            if (distances[index].w < distances[index].z) {
                distances[index].z *= -1.0;
                distances[index].w *= -1.0;
                distances[index].x *= -1.0;
                distances[index].y *= -1.0;
            } else if(abs(distances[index].w - distances[index].z) < EPSILON_ZERO_OFFSET) {
                distances[index].z *= -1.0;
                distances[index].x *= -1.0;
            }
        } else {
            FloatType norm = length(segmentVectors[face_index * 3 + index]);
            if (distances[index].z < norm && distances[index].w < norm) {
                distances[index].z *= -1.0;
            } else if (distances[index].w < distances[index].z) {
                distances[index].z *= -1.0;
                distances[index].w *= -1.0;
            }
        }
    }

    FloatType3 projectionPointVertexNorms = {
        euclideanNorm(orthogonalProjectionPointOnPlane - face[0]),
        euclideanNorm(orthogonalProjectionPointOnPlane - face[1]),
        euclideanNorm(orthogonalProjectionPointOnPlane - face[2]),
    };

    TranscendentalExpression transcendentalExpressions[3];
    for (uint index = 0; index < 3; ++index) {
        FloatType r1Norm = projectionPointVertexNorms[(index + 1) % 3];
        FloatType r2Norm = projectionPointVertexNorms[index];

        if ((segmentNormalOrientations[index] == 0 && (r1Norm < EPSILON_ZERO_OFFSET || r2Norm < EPSILON_ZERO_OFFSET)) ||
            (abs(distances[index].z + distances[index].w) < EPSILON_ZERO_OFFSET &&
            abs(distances[index].x + distances[index].y) < EPSILON_ZERO_OFFSET))
        {
            transcendentalExpressions[index].ln = 0.0;
        } else {
            FloatType inner_num = distances[index].w + distances[index].y;
            FloatType inner_denom = distances[index].z + distances[index].x;

            if (inner_num <= 0.0 || inner_denom <= 0.0) {
                transcendentalExpressions[index].ln = 0.0;
            } else {
                FloatType inner_frac = inner_num / inner_denom;
                transcendentalExpressions[index].ln = custom_log(inner_frac);
            }
        }

        if (planeDistance < EPSILON_ZERO_OFFSET || segmentDistances[index] < EPSILON_ZERO_OFFSET) {
            transcendentalExpressions[index].an = 0.0;
        } else {
            FloatType frac1 = (planeDistance * distances[index].w) / (segmentDistances[index] * distances[index].y);
            FloatType frac2 = (planeDistance * distances[index].z) / (segmentDistances[index] * distances[index].x);

            transcendentalExpressions[index].an = custom_atan(frac1) - custom_atan(frac2);
        }
    }

    FloatType sing_theta = compute_singularities(face_index, segmentNormalOrientations, projectionPointVertexNorms);
    FloatType sing_alpha = -planeDistance * sing_theta;
    FloatType3 sing_beta = normals[face_index] * (-1.0 * sing_theta * planeNormalOrientation);

    FloatType sum1PotentialAcceleration = 0.0;
    for (uint index = 0; index < 3; ++index) {
        sum1PotentialAcceleration += segmentNormalOrientations[index] * segmentDistances[index] * transcendentalExpressions[index].ln;
    }

    FloatType3 sum1Tensor = {0.0, 0.0, 0.0};
    for (uint index = 0; index < 3; ++index) {
        sum1Tensor += segmentNormals[face_index * 3 + index] * transcendentalExpressions[index].ln;
    }

    FloatType sum2 = 0.0;
    for (uint index = 0; index < 3; ++index) {
        sum2 += segmentNormalOrientations[index] * transcendentalExpressions[index].an;
    }

    FloatType planeSumPotentialAcceleration = sum1PotentialAcceleration + planeDistance * sum2 + sing_alpha;
    FloatType3 subSum = (sum1Tensor + (normals[face_index] * (planeNormalOrientation * sum2))) + sing_beta;

    FloatType3 first = normals[face_index] * subSum;
    FloatType3 second = normals[face_index].xxy * subSum.yzz;

    Result res;
    res.res.w = planeNormalOrientation * planeDistance * planeSumPotentialAcceleration;
    res.res.xyz = normals[face_index] * planeSumPotentialAcceleration;
    res.first.xyz = first;
    res.second.xyz = second;

#ifdef TARGET_CUDA
    results[face_index] = res;
#else
    FloatType4 groupSum0 = WaveActiveSum(res.res);
    FloatType4 groupSum1 = WaveActiveSum(res.first);
    FloatType4 groupSum2 = WaveActiveSum(res.second);

    if (groupThreadIdx.x == 0) {
        atomicAdd(results[0].res.x, groupSum0.x);
        atomicAdd(results[0].res.y, groupSum0.y);
        atomicAdd(results[0].res.z, groupSum0.z);
        atomicAdd(results[0].res.w, groupSum0.w);

        atomicAdd(results[0].first.x, groupSum1.x);
        atomicAdd(results[0].first.y, groupSum1.y);
        atomicAdd(results[0].first.z, groupSum1.z);

        atomicAdd(results[0].second.x, groupSum2.x);
        atomicAdd(results[0].second.y, groupSum2.y);
        atomicAdd(results[0].second.z, groupSum2.z);
    }
#endif
}
