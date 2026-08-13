@group(0) @binding(0) var<storage, read_write> vertices: array<vec3f>;
@group(0) @binding(1) var<storage, read_write> faces: array<vec3u>;

@group(0) @binding(2) var<storage, read_write> normals: array<vec3f>;
@group(0) @binding(3) var<storage, read_write> segment_vectors: array<vec3f>;
@group(0) @binding(4) var<storage, read_write> segment_normals: array<vec3f>;

@group(0) @binding(5) var<storage, read_write> results: array<vec4f>;

@group(0) @binding(6) var<storage, read_write> settings: array<vec4f>;

fn normal(first: vec3f, second: vec3f) -> vec3f {
    return normalize(cross(first, second));
}

@compute
@workgroup_size(1, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    var index: u32 = global_id.x + global_id.y * 65534u;

    if (index >= u32(settings[0].w)) { return; };

    results[index] = vec4(0.0, 0.0, 0.0, 0.0);

    let Face = array<vec3f, 3>(
        vertices[faces[index][0]],
        vertices[faces[index][1]],
        vertices[faces[index][2]]
    );

    segment_vectors[index * 3 + 0] = Face[1] - Face[0];
    segment_vectors[index * 3 + 1] = Face[2] - Face[1];
    segment_vectors[index * 3 + 2] = Face[0] - Face[2];

    normals[index] = normal(segment_vectors[index* 3 + 0], segment_vectors[index * 3 + 1]);

    segment_normals[index * 3 + 0] = normal(segment_vectors[index * 3 + 0], normals[index]);
    segment_normals[index * 3 + 1] = normal(segment_vectors[index * 3 + 1], normals[index]);
    segment_normals[index * 3 + 2] = normal(segment_vectors[index * 3 + 2], normals[index]);

    if (global_id.z == 1234) {
        vertices[0].x = 0.0;
        faces[0].x = 0u;
        normals[0].x = 0.0;
        segment_vectors[0].x = 0.0;
        segment_normals[0].x = 0.0;
        results[0].x = 0.0;
        settings[0].x = 0.0;
    }
}
