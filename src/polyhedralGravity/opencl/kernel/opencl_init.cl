kernel void vecadd(
    global const FloatType3* vertices,
    global const int3* faces,
    global FloatType3* normals,
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

    // The plane unit normals N_p are the only per-face property that is cached
    normals[index] = normalize(cross(face[1] - face[0], face[2] - face[1]));
}
