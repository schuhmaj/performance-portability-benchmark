
const FloatType PI = 3.1415926535897932384626433832795028841971693993751058209749445923;
const FloatType PI2 = 6.2831853071795864769252867665590057683943387987502116419498891846;
const FloatType PI_2 = 1.5707963267948966192313216916397514420985846996875529104874722961;
// The radius around zero which is treated as zero: 1e-14 scaled to the resolution of FloatType
const FloatType EPSILON_ZERO = sizeof(FloatType) == sizeof(float) ? 5.3687091e-6 : 1e-14;

FloatType sgn(FloatType val) {
    if (val < -EPSILON_ZERO) return -1;
    if (val > EPSILON_ZERO) return 1;
    return 0;
}

typedef struct {
    FloatType l1;
    FloatType l2;
    FloatType s1;
    FloatType s2;
} Distance;

typedef struct {
    FloatType ln;
    FloatType an;
} TranscendentalExpression;

kernel void vecadd(
    global const FloatType3* vertices,
    global const int3* faces,
    global const FloatType3* normals,
    global FloatType16* results,
    int num_faces,
    FloatType p1,
    FloatType p2,
    FloatType p3,
    local FloatType* partialSums
   ){
    FloatType3 point = {p1, p2, p3};

    // Out-of-range work-items must NOT return early: every work-item of the work-group has
    // to reach the barriers of the local-memory reduction below, otherwise the others
    // deadlock. Instead the out-of-range items redundantly evaluate face 0 and contribute
    // zero to the reduction, which is the same guard opencl_sum.cl uses.
    const int global_index = get_global_id(0);
    const bool inRange = global_index < num_faces;
    const int face_index = inRange ? global_index : 0;

    const FloatType3 face[3] = {
        vertices[faces[face_index][0]] - point,
        vertices[faces[face_index][1]] - point,
        vertices[faces[face_index][2]] - point,
    };
    const FloatType3 planeUnitNormal = normals[face_index];

    // Recomputed rather than cached: three subtractions are cheaper than reading 36 more bytes per face
    const FloatType3 segmentVectors[3] = {face[1] - face[0], face[2] - face[1], face[0] - face[2]};

    // N_p is a unit vector, so N_p * v_0 is the signed distance of P to the plane and P' is N_p scaled by it
    const FloatType planeProjection = dot(planeUnitNormal, face[0]);
    const FloatType planeNormalOrientation = sgn(planeProjection);
    const FloatType planeDistance = fabs(planeProjection);
    const FloatType3 orthogonalProjectionPointOnPlane = planeUnitNormal * planeProjection;

    // sigma_pq, h_pq, l1, l2, s1, s2 and |P' - v_q| follow from projecting P' - v_q onto n_pq and onto G_pq
    const FloatType3 vertexNorms = {length(face[0]), length(face[1]), length(face[2])};
    FloatType3 segmentUnitNormals[3];
    FloatType3 segmentNormalOrientations;
    FloatType3 segmentDistances;
    FloatType3 projectionPointVertexNorms;
    Distance distances[3];
    for (int index = 0; index < 3; ++index) {
        const FloatType3 relativeProjectionPoint = orthogonalProjectionPointOnPlane - face[index];
        projectionPointVertexNorms[index] = length(relativeProjectionPoint);

        // n_pq is normalized by |G_pq|, which is |G_pq x N_p| since N_p is a unit vector perpendicular to G_pq
        const FloatType squaredSegmentNorm = dot(segmentVectors[index], segmentVectors[index]);
        const FloatType inverseSegmentNorm = rsqrt(squaredSegmentNorm);
        const FloatType segmentNorm = squaredSegmentNorm * inverseSegmentNorm;
        segmentUnitNormals[index] = cross(segmentVectors[index], planeUnitNormal) * inverseSegmentNorm;

        const FloatType normalProjection = dot(segmentUnitNormals[index], relativeProjectionPoint);
        segmentNormalOrientations[index] = -sgn(normalProjection);
        segmentDistances[index] = fabs(normalProjection);

        const FloatType alongSegment = dot(relativeProjectionPoint, segmentVectors[index]) * inverseSegmentNorm;
        Distance distance;
        distance.l1 = vertexNorms[index];
        distance.l2 = vertexNorms[(index + 1) % 3];
        distance.s1 = fabs(alongSegment);
        distance.s2 = fabs(alongSegment - segmentNorm);

        // The 1., 2. and 3. Option of Tsoulis (2021) all amount to s1 = -u and s2 = |G_pq| - u
        if (fabs(distance.s1 - distance.l1) >= EPSILON_ZERO || fabs(distance.s2 - distance.l2) >= EPSILON_ZERO) {
            distance.s1 = -alongSegment;
            distance.s2 = segmentNorm - alongSegment;
        } else if (distance.s2 < distance.s1) {
            distance.s1 = -distance.s1;
            distance.s2 = -distance.s2;
            distance.l1 = -distance.l1;
            distance.l2 = -distance.l2;
        } else if (fabs(distance.s2 - distance.s1) < EPSILON_ZERO) {
            distance.s1 = -distance.s1;
            distance.l1 = -distance.l1;
        }
        distances[index] = distance;
    }

    // Both transcendental expressions are evaluated unconditionally and then selected
    TranscendentalExpression transcendentalExpressions[3];
    for (int index = 0; index < 3; ++index) {
        const Distance distance = distances[index];
        const FloatType r1Norm = projectionPointVertexNorms[(index + 1) % 3];
        const FloatType r2Norm = projectionPointVertexNorms[index];

        const bool logarithmVanishes =
            (segmentNormalOrientations[index] == 0 && (r1Norm < EPSILON_ZERO || r2Norm < EPSILON_ZERO)) ||
            (fabs(distance.s1 + distance.s2) < EPSILON_ZERO && fabs(distance.l1 + distance.l2) < EPSILON_ZERO);
        const FloatType logarithm = log((distance.s2 + distance.l2) / (distance.s1 + distance.l1));
        transcendentalExpressions[index].ln = logarithmVanishes ? 0 : logarithm;

        // atan(x) - atan(y) = atan((x - y) / (1 + xy)), which misses a whole PI (with the sign of x) if 1 + xy < 0
        const bool arcTangentVanishes = planeDistance < EPSILON_ZERO || segmentDistances[index] < EPSILON_ZERO;
        const FloatType upper = (planeDistance * distance.s2) / (segmentDistances[index] * distance.l2);
        const FloatType lower = (planeDistance * distance.s1) / (segmentDistances[index] * distance.l1);
        const FloatType denominator = 1 + upper * lower;
        const FloatType branchOffset = denominator < 0 ? (upper < 0 ? -PI : PI) : 0;
        const FloatType arcTangent = atan((upper - lower) / denominator) + branchOffset;
        transcendentalExpressions[index].an = arcTangentVanishes ? 0 : arcTangent;
    }

    // The singularities are sing A = factor * h_p and sing B = factor * sigma_p * N_p in all four cases
    const bool allInside = segmentNormalOrientations[0] == 1 && segmentNormalOrientations[1] == 1 && segmentNormalOrientations[2] == 1;
    bool anyOnLine = false;
    bool anyAtVertex = false;
    int vertexSegment = 0;
    bool vertexIsSegmentEnd = false;
    for (int index = 0; index < 3; ++index) {
        if (segmentNormalOrientations[index] != 0) {
            continue;
        }
        const FloatType r1Norm = projectionPointVertexNorms[(index + 1) % 3];
        const FloatType r2Norm = projectionPointVertexNorms[index];
        const FloatType squaredSegmentNorm = dot(segmentVectors[index], segmentVectors[index]);
        const bool atVertex = r1Norm < EPSILON_ZERO || r2Norm < EPSILON_ZERO;
        anyOnLine = anyOnLine || (!atVertex && r1Norm * r1Norm < squaredSegmentNorm && r2Norm * r2Norm < squaredSegmentNorm);
        vertexSegment = anyAtVertex ? vertexSegment : index;
        vertexIsSegmentEnd = anyAtVertex ? vertexIsSegmentEnd : r1Norm < EPSILON_ZERO;
        anyAtVertex = anyAtVertex || atVertex;
    }
    FloatType vertexAngle = 0;
    if (anyAtVertex) {
        const FloatType3 g1 = vertexIsSegmentEnd ? segmentVectors[vertexSegment] : segmentVectors[(vertexSegment + 2) % 3];
        const FloatType3 g2 = vertexIsSegmentEnd ? segmentVectors[(vertexSegment + 1) % 3] : segmentVectors[vertexSegment];
        const FloatType gdot = -dot(g1, g2);
        vertexAngle = gdot == 0 ? PI_2 : acos(gdot / (length(g1) * length(g2)));
    }
    const FloatType singularityFactor = allInside ? -PI2 : anyOnLine ? -PI : -vertexAngle;
    const FloatType sing_alpha = singularityFactor * planeDistance;
    const FloatType3 sing_beta = planeUnitNormal * (singularityFactor * planeNormalOrientation);

    FloatType sum1PotentialAcceleration = 0;
    for (int index = 0; index < 3; ++index)
        sum1PotentialAcceleration += segmentNormalOrientations[index] * segmentDistances[index] * transcendentalExpressions[index].ln;

    FloatType3 sum1Tensor = (FloatType3)(0);
    for (int index = 0; index < 3; ++index)
        sum1Tensor = sum1Tensor + segmentUnitNormals[index] * transcendentalExpressions[index].ln;

    FloatType sum2 = 0;
    for (int index = 0; index < 3; ++index)
        sum2 += segmentNormalOrientations[index] * transcendentalExpressions[index].an;

    FloatType planeSumPotentialAcceleration = sum1PotentialAcceleration + planeDistance * sum2 + sing_alpha;
    FloatType3 subSum = (sum1Tensor + (planeUnitNormal * (planeNormalOrientation * sum2))) + sing_beta;
    FloatType3 first = planeUnitNormal * subSum;

    FloatType3 reorderedNp = {planeUnitNormal[0], planeUnitNormal[0], planeUnitNormal[1]};
    FloatType3 reorderedSubSum = {subSum[1], subSum[2], subSum[2]};
    FloatType3 second = reorderedNp * reorderedSubSum;

    FloatType16 result_value = (FloatType16)(0.0);
    if (inRange) {
        result_value.w = planeNormalOrientation * planeDistance * planeSumPotentialAcceleration;
        result_value.xyz = planeUnitNormal * planeSumPotentialAcceleration;
        result_value.s456 = first;
        result_value.s789 = second;
    }

    // Shared-memory reduction over the ten used components of the work-group
    const uint localId = get_local_id(0);
    for (uint i = 0; i < 10; ++i) partialSums[localId * 10 + i] = result_value[i];
    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint stride = get_local_size(0) / 2; stride > 0; stride /= 2) {
        if (localId < stride) {
            for (uint i = 0; i < 10; ++i) partialSums[localId * 10 + i] += partialSums[(localId + stride) * 10 + i];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (localId == 0) {
        for (uint i = 0; i < 10; ++i) result_value[i] = partialSums[i];
        results[get_group_id(0)] = result_value;
    }
}
