@group(0) @binding(0) var<storage, read_write> vertices: array<vec3f>;
@group(0) @binding(1) var<storage, read_write> faces: array<vec3u>;

@group(0) @binding(2) var<storage, read_write> normals: array<vec3f>;
@group(0) @binding(3) var<storage, read_write> segment_vectors: array<vec3f>;
@group(0) @binding(4) var<storage, read_write> segment_normals: array<vec3f>;

@group(0) @binding(5) var<storage, read_write> results: array<f32>;

@group(0) @binding(6) var<storage, read_write> settings: array<vec4f>;

const EPSILON_ZERO_OFFSET = 1e-14;
const PI = 3.1415926535897932384626433832795028841971693993751058209749445923;
const PI2 = 6.2831853071795864769252867665590057683943387987502116419498891846;
const PI_2 = 1.5707963267948966192313216916397514420985846996875529104874722961;

fn sgn(val: f32) -> i32 {
    if (val < -EPSILON_ZERO_OFFSET) {return -1;}
    if (val > EPSILON_ZERO_OFFSET) {return 1;}
    return 0;
}

fn compute_singularities(face_index: u32, segmentNormalOrientations: vec3i, projectionPointVertexNorms :vec3f) -> f32 {
    var allInside = true;
    for (var index = 0; index < 3; index+=1) {
        allInside &= segmentNormalOrientations[index] == 1;
    }
    if (allInside) { return PI2; }

    var anyOnLine = false;
    for (var index = 0u; index < 3u; index+=1u) {
        if (segmentNormalOrientations[index] != 0) {
            continue;
        }
        var segmentVectorNorm = length(segment_vectors[face_index * 3 + index]);
        anyOnLine |= projectionPointVertexNorms[(index + 1) % 3] < segmentVectorNorm && projectionPointVertexNorms[index] < segmentVectorNorm;
    }
    if (anyOnLine) { return PI; }

    for (var index = 0u; index < 3u; index+=1u) {
        if (segmentNormalOrientations[index] != 0) {
            continue;
        }

        let r1Norm = projectionPointVertexNorms[(index + 1) % 3];
        let r2Norm = projectionPointVertexNorms[index];

        if (!(r1Norm < EPSILON_ZERO_OFFSET || r2Norm < EPSILON_ZERO_OFFSET)) {
            continue;
        }

        var g1 = segment_vectors[face_index * 3 + index];
        if (r1Norm != 0.0) {
            g1 = segment_vectors[face_index * 3 + (index - 1 + 3) % 3];
        }

        var g2 = segment_vectors[face_index * 3 + (index + 1) % 3];
        if (r1Norm != 0.0) {
            g2 = segment_vectors[face_index * 3 + index];
        }

        let gdot = dot(g1 * -1.0, g2);
        var theta = PI_2;

        if (gdot != 0.0) { theta = acos(gdot / (length(g1) * length(g2))); }
        return theta;
    }

    return 0.0;
}

@compute
@workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    // x-dimension is capped at 65534 workgroups (each 256 threads) by the host
    var face_index: u32 = global_id.x + global_id.y * 65534u * 256u;

    if (face_index >= u32(settings[0].w)) { return; };

    var face = array<vec3f, 3>(
        vertices[faces[face_index][0]] - settings[0].x,
        vertices[faces[face_index][1]] - settings[0].y,
        vertices[faces[face_index][2]] - settings[0].z
    );

    let planeNormalOrientation = sgn(dot(normals[face_index], face[0]));

    var hessianPlane = vec4f();
    {
        let origin = vec3f(0.0, 0.0, 0.0);
        let crossP =  cross(face[0] - face[1], face[0] - face[2]);
        let res = (face[0] * -1.0) * crossP;
        let d = res.x + res.y + res.z;

        hessianPlane.x = crossP.x;
        hessianPlane.y = crossP.y;
        hessianPlane.z = crossP.z;
        hessianPlane.w = d;
    }

    let planeDistance = abs(hessianPlane.w / sqrt(hessianPlane.x * hessianPlane.x + hessianPlane.y * hessianPlane.y + hessianPlane.z * hessianPlane.z));
    var orthogonalProjectionPointOnPlane = normals[face_index] * planeDistance;

    {
        var intersections = vec3f(0.0, 0.0, 0.0);
        if (hessianPlane.x != 0.0) { intersections.x = hessianPlane.w / hessianPlane.x; }
        if (hessianPlane.y != 0.0) { intersections.y = hessianPlane.w / hessianPlane.y; }
        if (hessianPlane.z != 0.0) { intersections.z = hessianPlane.w / hessianPlane.z; }

        for (var index = 0; index < 3; index+=1) {
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

    var segmentNormalOrientations = vec3i(0, 0, 0);
    segmentNormalOrientations[0] = -sgn(dot(segment_normals[face_index * 3 + 0], orthogonalProjectionPointOnPlane - face[0]));
    segmentNormalOrientations[1] = -sgn(dot(segment_normals[face_index * 3 + 1], orthogonalProjectionPointOnPlane - face[1]));
    segmentNormalOrientations[2] = -sgn(dot(segment_normals[face_index * 3 + 2], orthogonalProjectionPointOnPlane - face[2]));

    var orthogonalProjectionPointsOnSegmentsForPlane = array<vec3f, 3>();

    for (var index = 0; index < 3; index+=1) {
        if (segmentNormalOrientations[index] == 0) {
            orthogonalProjectionPointsOnSegmentsForPlane[index] = orthogonalProjectionPointOnPlane;
        } else {
            let vertex1 = face[index];
            let vertex2 = face[(index + 1) % 3];

            let matrixRow1 = vertex2 - vertex1;
            let matrixRow2 = cross(vertex1 - orthogonalProjectionPointOnPlane, matrixRow1);
            let matrixRow3 = cross(matrixRow2, matrixRow1);

            let d = vec3f(
                dot(matrixRow1, orthogonalProjectionPointOnPlane),
                dot(matrixRow2, orthogonalProjectionPointOnPlane),
                dot(matrixRow3, vertex1)
            );

            var m = mat3x3f(
                matrixRow1, matrixRow2, matrixRow3
            );
            var columnMatrix = transpose(m);

            let determinant_ = determinant(columnMatrix);

            if (determinant_ != 0.0) {
                var res = vec3f(
                    determinant(mat3x3f(d, columnMatrix[1], columnMatrix[2])),
                    determinant(mat3x3f(columnMatrix[0], d, columnMatrix[2])),
                    determinant(mat3x3f(columnMatrix[0], columnMatrix[1], d)),
                );

                orthogonalProjectionPointsOnSegmentsForPlane[index].x = res.x  / determinant_;
                orthogonalProjectionPointsOnSegmentsForPlane[index].y = res.y  / determinant_;
                orthogonalProjectionPointsOnSegmentsForPlane[index].z = res.z  / determinant_;
            }
        }
    }

    var segmentDistances = vec3f();
    for (var index = 0; index < 3; index+=1) {
        segmentDistances[index] = length(orthogonalProjectionPointsOnSegmentsForPlane[index] - orthogonalProjectionPointOnPlane);
    }

    var distances = array<vec4f, 3>();
    for (var index = 0u; index < 3u; index+=1u) {
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
            let norm = length(segment_vectors[face_index * 3 + index]);
            if (distances[index].z < norm && distances[index].w < norm) {
                distances[index].z *= -1.0;
            } else if (distances[index].w < distances[index].z) {
                distances[index].z *= -1.0;
                distances[index].w *= -1.0;
            }
        }
    }

    var projectionPointVertexNorms = vec3f(
        length(orthogonalProjectionPointOnPlane - face[0]),
        length(orthogonalProjectionPointOnPlane - face[1]),
        length(orthogonalProjectionPointOnPlane - face[2])
    );

    var transcendentalExpressions = array<vec2f, 3>();
    for (var index = 0u; index < 3u; index+=1u) {
        let r1Norm = projectionPointVertexNorms[(index + 1) % 3];
        let r2Norm = projectionPointVertexNorms[index];

        if ((segmentNormalOrientations[index] == 0 && (r1Norm < EPSILON_ZERO_OFFSET || r2Norm < EPSILON_ZERO_OFFSET)) ||
            (abs(distances[index].z + distances[index].w) < EPSILON_ZERO_OFFSET &&
            abs(distances[index].x + distances[index].y) < EPSILON_ZERO_OFFSET))
        {
            transcendentalExpressions[index].x = 0.0;
        } else {
            let inner_num = distances[index].w + distances[index].y;
            let inner_denom = distances[index].z + distances[index].x;

            if (inner_num <= 0.0 || inner_denom <= 0.0) {
                transcendentalExpressions[index].x = 0.0;
            } else {
                let inner_frac = inner_num / inner_denom;
                transcendentalExpressions[index].x = log(inner_frac);
            }
        }

        if (planeDistance < EPSILON_ZERO_OFFSET || segmentDistances[index] < EPSILON_ZERO_OFFSET) {
            transcendentalExpressions[index].y = 0.0;
        } else {
            let frac1 = (planeDistance * distances[index].w) / (segmentDistances[index] * distances[index].y);
            let frac2 = (planeDistance * distances[index].z) / (segmentDistances[index] * distances[index].x);

            transcendentalExpressions[index].y = atan(frac1) - atan(frac2);
        }
    }

    let sing_theta = compute_singularities(face_index, segmentNormalOrientations, projectionPointVertexNorms);
    let sing_alpha = -planeDistance * sing_theta;
    let sing_beta = normals[face_index] * (-1.0 * sing_theta * f32(planeNormalOrientation));

    var sum1PotentialAcceleration = 0.0;
    for (var index = 0u; index < 3u; index+=1u) {
        sum1PotentialAcceleration += f32(segmentNormalOrientations[index]) * segmentDistances[index] * transcendentalExpressions[index].x;
    }

    var sum1Tensor = vec3f(0.0, 0.0, 0.0);
    for (var index = 0u; index < 3u; index+=1u) {
        sum1Tensor += segment_normals[face_index * 3 + index] * transcendentalExpressions[index].x;
    }

    var sum2 = 0.0;
    for (var index = 0u; index < 3u; index+=1u) {
        sum2 += f32(segmentNormalOrientations[index]) * transcendentalExpressions[index].y;
    }

    let planeSumPotentialAcceleration = sum1PotentialAcceleration + planeDistance * sum2 + sing_alpha;

    let subSum1 = sum1Tensor + (normals[face_index] * (f32(planeNormalOrientation) * sum2));
    let subSum = subSum1 + sing_beta;

    let first = normals[face_index] * subSum;
    let reorderedNp = normals[face_index].xxy;
    let reorderedSubSum = subSum.yzz;
    let second = reorderedNp * reorderedSubSum;

    let res = normals[face_index] * planeSumPotentialAcceleration;

    results[face_index * 10 + 0] = res.x;
    results[face_index * 10 + 1] = res.y;
    results[face_index * 10 + 2] = res.z;
    results[face_index * 10 + 3] = f32(planeNormalOrientation) * planeDistance * planeSumPotentialAcceleration;

    results[face_index * 10 + 4] = first.x;
    results[face_index * 10 + 5] = first.y;
    results[face_index * 10 + 6] = first.z;

    results[face_index * 10 + 7] = second.x;
    results[face_index * 10 + 8] = second.y;
    results[face_index * 10 + 9] = second.z;

    if (global_id.z == 1234) {
        vertices[0].x = 0.0;
        faces[0].x = 0u;
        normals[0].x = 0.0;
        segment_vectors[0].x = 0.0;
        segment_normals[0].x = 0.0;
        results[0] = 0.0;
        settings[0].x = 0.0;
    }
}
