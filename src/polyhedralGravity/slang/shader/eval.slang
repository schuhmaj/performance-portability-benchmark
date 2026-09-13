#version 460
#extension GL_EXT_shader_atomic_float : require

#ifndef FloatType
#define FloatType double
#define FloatType3 double3
#define FloatType4 double4
#define FloatTypeM dmat3
#define FLOAT_BITS 64
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

const FloatType PI = 3.1415926535897932384626433832795028841971693993751058209749445923;
const FloatType PI2 = 6.2831853071795864769252867665590057683943387987502116419498891846;
const FloatType PI_2 = 1.5707963267948966192313216916397514420985846996875529104874722961;
// The radius around zero which is treated as zero: 1e-14 scaled to the resolution of FloatType
#if FLOAT_BITS == 32
const FloatType EPSILON_ZERO = 5.3687091e-6;
#else
const FloatType EPSILON_ZERO = 1e-14;
#endif

FloatType sgn(FloatType val) {
    return val < -EPSILON_ZERO ? -1.0 : val > EPSILON_ZERO ? 1.0 : 0.0;
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

void do_init(uint index) {
    FloatType3 Face[3] = {
        vertices[faces[index][0]],
        vertices[faces[index][1]],
        vertices[faces[index][2]]
    };

    // The plane unit normals N_p are the only per-face property that is cached
    normals[index] = normalize(cross(Face[1] - Face[0], Face[2] - Face[1]));
}

[shader("compute")]
[numthreads(256, 1, 1)]
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
    FloatType3 planeUnitNormal = normals[face_index];

    // Recomputed rather than cached: three subtractions are cheaper than reading 36 more bytes per face
    FloatType3 segmentVectors[3] = {face[1] - face[0], face[2] - face[1], face[0] - face[2]};

    // N_p is a unit vector, so N_p * v_0 is the signed distance of P to the plane and P' is N_p scaled by it
    FloatType planeProjection = dot(planeUnitNormal, face[0]);
    FloatType planeNormalOrientation = sgn(planeProjection);
    FloatType planeDistance = abs(planeProjection);
    FloatType3 orthogonalProjectionPointOnPlane = planeUnitNormal * planeProjection;

    // sigma_pq, h_pq, l1, l2, s1, s2 and |P' - v_q| follow from projecting P' - v_q onto n_pq and onto G_pq
    FloatType3 vertexNorms = {length(face[0]), length(face[1]), length(face[2])};
    FloatType3 segmentUnitNormals[3];
    FloatType3 segmentNormalOrientations = {};
    FloatType3 segmentDistances = {};
    FloatType3 projectionPointVertexNorms = {};
    Distance distances[3];
    for (uint index = 0; index < 3; ++index) {
        FloatType3 relativeProjectionPoint = orthogonalProjectionPointOnPlane - face[index];
        projectionPointVertexNorms[index] = length(relativeProjectionPoint);

        // n_pq is normalized by |G_pq|, which is |G_pq x N_p| since N_p is a unit vector perpendicular to G_pq
        FloatType squaredSegmentNorm = dot(segmentVectors[index], segmentVectors[index]);
        FloatType inverseSegmentNorm = rsqrt(squaredSegmentNorm);
        FloatType segmentNorm = squaredSegmentNorm * inverseSegmentNorm;
        segmentUnitNormals[index] = cross(segmentVectors[index], planeUnitNormal) * inverseSegmentNorm;

        FloatType normalProjection = dot(segmentUnitNormals[index], relativeProjectionPoint);
        segmentNormalOrientations[index] = -sgn(normalProjection);
        segmentDistances[index] = abs(normalProjection);

        FloatType alongSegment = dot(relativeProjectionPoint, segmentVectors[index]) * inverseSegmentNorm;
        Distance distance;
        distance.l1 = vertexNorms[index];
        distance.l2 = vertexNorms[(index + 1) % 3];
        distance.s1 = abs(alongSegment);
        distance.s2 = abs(alongSegment - segmentNorm);

        // The 1., 2. and 3. Option of Tsoulis (2021) all amount to s1 = -u and s2 = |G_pq| - u
        if (abs(distance.s1 - distance.l1) >= EPSILON_ZERO || abs(distance.s2 - distance.l2) >= EPSILON_ZERO) {
            distance.s1 = -alongSegment;
            distance.s2 = segmentNorm - alongSegment;
        } else if (distance.s2 < distance.s1) {
            distance.s1 = -distance.s1;
            distance.s2 = -distance.s2;
            distance.l1 = -distance.l1;
            distance.l2 = -distance.l2;
        } else if (abs(distance.s2 - distance.s1) < EPSILON_ZERO) {
            distance.s1 = -distance.s1;
            distance.l1 = -distance.l1;
        }
        distances[index] = distance;
    }

    // Both transcendental expressions are evaluated unconditionally and then selected
    TranscendentalExpression transcendentalExpressions[3];
    for (uint index = 0; index < 3; ++index) {
        Distance distance = distances[index];
        FloatType r1Norm = projectionPointVertexNorms[(index + 1) % 3];
        FloatType r2Norm = projectionPointVertexNorms[index];

        bool logarithmVanishes =
            (segmentNormalOrientations[index] == 0.0 && (r1Norm < EPSILON_ZERO || r2Norm < EPSILON_ZERO)) ||
            (abs(distance.s1 + distance.s2) < EPSILON_ZERO && abs(distance.l1 + distance.l2) < EPSILON_ZERO);
        FloatType logarithm = custom_log((distance.s2 + distance.l2) / (distance.s1 + distance.l1));
        transcendentalExpressions[index].ln = logarithmVanishes ? 0.0 : logarithm;

        // atan(x) - atan(y) = atan((x - y) / (1 + xy)), which misses a whole PI (with the sign of x) if 1 + xy < 0
        bool arcTangentVanishes = planeDistance < EPSILON_ZERO || segmentDistances[index] < EPSILON_ZERO;
        FloatType upper = (planeDistance * distance.s2) / (segmentDistances[index] * distance.l2);
        FloatType lower = (planeDistance * distance.s1) / (segmentDistances[index] * distance.l1);
        FloatType denominator = 1.0 + upper * lower;
        FloatType branchOffset = denominator < 0.0 ? (upper < 0.0 ? -PI : PI) : 0.0;
        FloatType arcTangent = custom_atan((upper - lower) / denominator) + branchOffset;
        transcendentalExpressions[index].an = arcTangentVanishes ? 0.0 : arcTangent;
    }

    // The singularities are sing A = factor * h_p and sing B = factor * sigma_p * N_p in all four cases
    bool allInside = segmentNormalOrientations[0] == 1.0 && segmentNormalOrientations[1] == 1.0 && segmentNormalOrientations[2] == 1.0;
    bool anyOnLine = false;
    bool anyAtVertex = false;
    uint vertexSegment = 0;
    bool vertexIsSegmentEnd = false;
    for (uint index = 0; index < 3; ++index) {
        if (segmentNormalOrientations[index] != 0.0) {
            continue;
        }
        FloatType r1Norm = projectionPointVertexNorms[(index + 1) % 3];
        FloatType r2Norm = projectionPointVertexNorms[index];
        FloatType squaredSegmentNorm = dot(segmentVectors[index], segmentVectors[index]);
        bool atVertex = r1Norm < EPSILON_ZERO || r2Norm < EPSILON_ZERO;
        anyOnLine = anyOnLine || (!atVertex && r1Norm * r1Norm < squaredSegmentNorm && r2Norm * r2Norm < squaredSegmentNorm);
        vertexSegment = anyAtVertex ? vertexSegment : index;
        vertexIsSegmentEnd = anyAtVertex ? vertexIsSegmentEnd : r1Norm < EPSILON_ZERO;
        anyAtVertex = anyAtVertex || atVertex;
    }
    FloatType vertexAngle = 0.0;
    if (anyAtVertex) {
        FloatType3 g1 = segmentVectors[vertexIsSegmentEnd ? vertexSegment : (vertexSegment + 2) % 3];
        FloatType3 g2 = segmentVectors[vertexIsSegmentEnd ? (vertexSegment + 1) % 3 : vertexSegment];
        FloatType gdot = -dot(g1, g2);
        vertexAngle = gdot == 0.0 ? PI_2 : custom_acos(gdot / (length(g1) * length(g2)));
    }
    FloatType singularityFactor = allInside ? -PI2 : anyOnLine ? -PI : -vertexAngle;
    FloatType sing_alpha = singularityFactor * planeDistance;
    FloatType3 sing_beta = planeUnitNormal * (singularityFactor * planeNormalOrientation);

    FloatType sum1PotentialAcceleration = 0.0;
    for (uint index = 0; index < 3; ++index) {
        sum1PotentialAcceleration += segmentNormalOrientations[index] * segmentDistances[index] * transcendentalExpressions[index].ln;
    }

    FloatType3 sum1Tensor = {0.0, 0.0, 0.0};
    for (uint index = 0; index < 3; ++index) {
        sum1Tensor += segmentUnitNormals[index] * transcendentalExpressions[index].ln;
    }

    FloatType sum2 = 0.0;
    for (uint index = 0; index < 3; ++index) {
        sum2 += segmentNormalOrientations[index] * transcendentalExpressions[index].an;
    }

    FloatType planeSumPotentialAcceleration = sum1PotentialAcceleration + planeDistance * sum2 + sing_alpha;
    FloatType3 subSum = (sum1Tensor + (planeUnitNormal * (planeNormalOrientation * sum2))) + sing_beta;

    FloatType3 first = planeUnitNormal * subSum;
    FloatType3 second = planeUnitNormal.xxy * subSum.yzz;

    Result res;
    res.res.w = planeNormalOrientation * planeDistance * planeSumPotentialAcceleration;
    res.res.xyz = planeUnitNormal * planeSumPotentialAcceleration;
    res.first.xyz = first;
    res.second.xyz = second;

#ifdef TARGET_CUDA
    results[face_index] = res;
#else
    // WaveActiveSum reduces per wave (not per group); with a 256-thread group the
    // first active lane of EVERY wave has to contribute its wave's partial sum.
    FloatType4 groupSum0 = WaveActiveSum(res.res);
    FloatType4 groupSum1 = WaveActiveSum(res.first);
    FloatType4 groupSum2 = WaveActiveSum(res.second);

    if (WaveIsFirstLane()) {
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
