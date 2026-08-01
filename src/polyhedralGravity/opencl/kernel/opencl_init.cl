kernel void vecadd(
    global const FloatType3* vertices,
    global const int3* faces,
    global FloatType3* normals,
    global FloatType3* segmentVectors,
    global FloatType3* segmentNormals,
    int num_faces){

    const int index = get_global_id(0);

    if (index >= num_faces) {
        return;
    }

    FloatType3 face[3] = {
        vertices[faces[index].x],
        vertices[faces[index].y],
        vertices[faces[index].z],
    };

    FloatType3 sv[3] = {
        face[1] - face[0], face[2] - face[1], face[0] -face[2]
    };

    FloatType3 n = normalize(cross(sv[0], sv[1]));
    normals[index] = n;

    segmentVectors[index * 3 + 0] = sv[0];
    segmentVectors[index * 3 + 1] = sv[1];
    segmentVectors[index * 3 + 2] = sv[2];

    segmentNormals[index * 3 + 0] = normalize(cross(sv[0], n));
    segmentNormals[index * 3 + 1] = normalize(cross(sv[1], n));
    segmentNormals[index * 3 + 2] = normalize(cross(sv[2], n));
}
