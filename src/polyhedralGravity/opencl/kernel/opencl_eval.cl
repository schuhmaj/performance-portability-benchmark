
const FloatType EPSILON_ZERO_OFFSET = 1e-14;
const FloatType PI = 3.1415926535897932384626433832795028841971693993751058209749445923;
const FloatType PI2 = 6.2831853071795864769252867665590057683943387987502116419498891846;
const FloatType PI_2 = 1.5707963267948966192313216916397514420985846996875529104874722961;

int sgn(FloatType val) {
    if (val < -EPSILON_ZERO_OFFSET) return -1;
    if (val > EPSILON_ZERO_OFFSET) return 1;
    return 0;
}

void transpose(FloatType3 matrix[3]) {
    FloatType3 copy[3] = {
        matrix[0], matrix[1], matrix[2],
    };

    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            matrix[i][j] = copy[j][i];
        }
    }
}

FloatType det(FloatType3 matrix[3]) {
    return matrix[0][0] * matrix[1][1] * matrix[2][2] +
           matrix[0][1] * matrix[1][2] * matrix[2][0] +
           matrix[0][2] * matrix[1][0] * matrix[2][1] -
           matrix[0][2] * matrix[1][1] * matrix[2][0] -
           matrix[0][0] * matrix[1][2] * matrix[2][1] -
           matrix[0][1] * matrix[1][0] * matrix[2][2];
}

FloatType det_v(FloatType3 a, FloatType3 b, FloatType3 c) {
    FloatType3 matrix[3] = {a, b, c};
    return det(matrix);
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

FloatType compute_singularities(
    int face_index,
    int3 segmentNormalOrientations,
    FloatType3 projectionPointVertexNorms,
    global const FloatType3* segmentVectors
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

        FloatType gdot = dot(-g1, g2);
        FloatType theta = gdot == 0.0 ? PI_2 : acos(gdot / (length(g1) * length(g2)));
        return theta;
    }

    return 0.0;
}

kernel void vecadd(
    global const FloatType3* vertices,
    global const int3* faces,
    global const FloatType3* normals,
    global const FloatType3* segmentVectors,
    global const FloatType3* segmentNormals,
    global FloatType16* results,
    int num_faces,
    FloatType p1,
    FloatType p2,
    FloatType p3
   ){
    FloatType3 point = {p1, p2, p3};

    const int face_index = get_global_id(0);

    if (face_index >= num_faces) {
        return;
    }

    FloatType3 face[3] = {
        vertices[faces[face_index][0]] - point,
        vertices[faces[face_index][1]] - point,
        vertices[faces[face_index][2]] - point,
    };

    int planeNormalOrientation = sgn(dot(normals[face_index], face[0]));

    FloatType4 hessianPlane;
    {
        FloatType3 origin = {0.0, 0.0, 0.0};
        FloatType3 crossProduct = cross(face[0] - face[1], face[0] - face[2]);
        FloatType3 res = (origin - face[0]) * crossProduct;
        FloatType d = res[0] + res[1] + res[2];

        hessianPlane.xyz = crossProduct;
        hessianPlane.w = d;
    }

    FloatType planeDistance = fabs(hessianPlane.w / sqrt(hessianPlane.x * hessianPlane.x + hessianPlane.y * hessianPlane.y + hessianPlane.z * hessianPlane.z));

    FloatType3 orthogonalProjectionPointOnPlane = normals[face_index] * planeDistance;
    {
        FloatType3 intersections = {
            hessianPlane.x == 0.0 ? 0.0 : hessianPlane.w / hessianPlane.x,
            hessianPlane.y == 0.0 ? 0.0 : hessianPlane.w / hessianPlane.y,
            hessianPlane.z == 0.0 ? 0.0 : hessianPlane.w / hessianPlane.z,
        };

        for (unsigned int index = 0; index < 3; ++index) {
            if (intersections[index] < 0) {
                orthogonalProjectionPointOnPlane[index] = fabs(orthogonalProjectionPointOnPlane[index]);
            } else {
                if (orthogonalProjectionPointOnPlane[index] > 0) {
                    orthogonalProjectionPointOnPlane[index] = -orthogonalProjectionPointOnPlane[index];
                } else {
                    orthogonalProjectionPointOnPlane[index] = orthogonalProjectionPointOnPlane[index];
                }
            }
        }
    }

    int3 segmentNormalOrientations;
    for (unsigned int index = 0; index < 3; ++index) {
        FloatType inner = dot(segmentNormals[face_index * 3 + index], orthogonalProjectionPointOnPlane - face[index]);
        segmentNormalOrientations[index] = -sgn(inner);
    }

    FloatType3 orthogonalProjectionPointsOnSegmentsForPlane[3];
    for (unsigned int index = 0; index < 3; ++index) {
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
                dot(matrixRow3, vertex1)
            };

            FloatType3 columnMatrix[3] = {
                matrixRow1,
                matrixRow2,
                matrixRow3
            };
            transpose(columnMatrix);

            FloatType determinant = det(columnMatrix);

            if (determinant != 0.0) {
                FloatType3 r = {
                     det_v(d, columnMatrix[1], columnMatrix[2]),
                     det_v(columnMatrix[0], d, columnMatrix[2]),
                     det_v(columnMatrix[0], columnMatrix[1], d),
                };
                orthogonalProjectionPointsOnSegmentsForPlane[index] = r / determinant;
            }
        }
    }

    FloatType3 segmentDistances;
    for (unsigned int index = 0; index < 3; ++index) {
        segmentDistances[index] = length(orthogonalProjectionPointsOnSegmentsForPlane[index] - orthogonalProjectionPointOnPlane);
    }

    Distance distances[3];
    for (unsigned int index = 0; index < 3; ++index) {
        distances[index].l1 = length(face[index]);
        distances[index].l2 = length(face[(index + 1) % 3]);

        distances[index].s1 = length(orthogonalProjectionPointsOnSegmentsForPlane[index] - face[index]);
        distances[index].s2 = length(orthogonalProjectionPointsOnSegmentsForPlane[index] - face[(index + 1) % 3]);

        if (fabs(distances[index].s1 - distances[index].l1) < EPSILON_ZERO_OFFSET && fabs(distances[index].s2 - distances[index].l2) < EPSILON_ZERO_OFFSET) {
            if (distances[index].s2 < distances[index].s1) {
                distances[index].s1 *= -1.0;
                distances[index].s2 *= -1.0;
                distances[index].l1 *= -1.0;
                distances[index].l2 *= -1.0;
            } else if (fabs(distances[index].s2 - distances[index].s1) < EPSILON_ZERO_OFFSET) {
                distances[index].s1 *= -1.0;
                distances[index].l1 *= -1.0;
            }
        } else {
            FloatType norm = length(segmentVectors[face_index * 3 + index]);
            if (distances[index].s1 < norm && distances[index].s2 < norm) {
                distances[index].s1 *= -1.0;
            } else if (distances[index].s2 < distances[index].s1) {
                distances[index].s1 *= -1.0;
                distances[index].s2 *= -1.0;
            }
        }
    }

    FloatType3 projectionPointVertexNorms = {
        length(orthogonalProjectionPointOnPlane - face[0]),
        length(orthogonalProjectionPointOnPlane - face[1]),
        length(orthogonalProjectionPointOnPlane - face[2]),
    };

    TranscendentalExpression transcendentalExpressions[3];
    for (unsigned int index = 0; index < 3; ++index) {
        FloatType r1Norm = projectionPointVertexNorms[(index + 1) % 3];
        FloatType r2Norm = projectionPointVertexNorms[index];

        if ((segmentNormalOrientations[index] == 0 && (r1Norm < EPSILON_ZERO_OFFSET || r2Norm < EPSILON_ZERO_OFFSET)) ||
            (fabs(distances[index].s1 + distances[index].s2) < EPSILON_ZERO_OFFSET &&
            fabs(distances[index].l1 + distances[index].l2) < EPSILON_ZERO_OFFSET)) {
            transcendentalExpressions[index].ln = 0.0;
        } else {
            FloatType inner_num = distances[index].s2 + distances[index].l2;
            FloatType inner_denom = distances[index].s1 + distances[index].l1;

            if (inner_num <= 0.0 || inner_denom <= 0.0) {
                transcendentalExpressions[index].ln = 0.0;
            } else {
                transcendentalExpressions[index].ln = log(inner_num / inner_denom);
            }
        }

        if (planeDistance < EPSILON_ZERO_OFFSET || segmentDistances[index] < EPSILON_ZERO_OFFSET) {
            transcendentalExpressions[index].an = 0.0;
        } else {
            FloatType frac1 = (planeDistance * distances[index].s2) / (segmentDistances[index] * distances[index].l2);
            FloatType frac2 = (planeDistance * distances[index].s1) / (segmentDistances[index] * distances[index].l1);

            transcendentalExpressions[index].an = atan(frac1) - atan(frac2);
        }
    }

    FloatType sing_theta = compute_singularities(face_index, segmentNormalOrientations, projectionPointVertexNorms, segmentVectors);
    FloatType sing_alpha = -planeDistance * sing_theta;
    FloatType3 sing_beta = normals[face_index] * ((FloatType) -1.0 * sing_theta * planeNormalOrientation);

    FloatType sum1PotentialAcceleration = 0.0;
    for (unsigned int index = 0; index < 3; ++index)
        sum1PotentialAcceleration += segmentNormalOrientations[index] * segmentDistances[index] * transcendentalExpressions[index].ln;

    FloatType3 sum1Tensor = {0.0, 0.0, 0.0};
    for (unsigned int index = 0; index < 3; ++index)
        sum1Tensor = sum1Tensor + segmentNormals[face_index * 3 + index] * transcendentalExpressions[index].ln;

    FloatType sum2 = 0.0;
    for (unsigned int index = 0; index < 3; ++index)
        sum2 += segmentNormalOrientations[index] * transcendentalExpressions[index].an;

    FloatType planeSumPotentialAcceleration = sum1PotentialAcceleration + planeDistance * sum2 + sing_alpha;
    FloatType3 subSum = (sum1Tensor + (normals[face_index] * (planeNormalOrientation * sum2))) + sing_beta;
    FloatType3 first = normals[face_index] * subSum;

    FloatType3 reorderedNp = {normals[face_index][0], normals[face_index][0], normals[face_index][1]};
    FloatType3 reorderedSubSum = {subSum[1], subSum[2], subSum[2]};
    FloatType3 second = reorderedNp * reorderedSubSum;

    FloatType16 result_value;
    result_value.w = planeNormalOrientation * planeDistance * planeSumPotentialAcceleration;
    result_value.xyz = normals[face_index] * planeSumPotentialAcceleration;
    result_value.s456 = first;
    result_value.s789 = second;

    for (uint i = 0; i<10; ++i) result_value[i] = work_group_reduce_add(result_value[i]);
    if (get_local_id(0) == 0) { results[get_group_id(0)] = result_value; }
}
