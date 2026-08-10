#include <cstdio>
#include <stdexcept>

#define SLANG_CUDA_STRUCTURED_BUFFER_NO_COUNT
#include "eval.cuh"

#if FLOAT_BITS == 32
using FloatType = float;
#elif FLOAT_BITS == 64
using FloatType = double;
#else
#error "Invliad float bits size"
#endif

void checkCudaError(cudaError_t err, int line) {
    if (err != cudaSuccess) {
        // Get the error string
        const char *errorString = cudaGetErrorString(err);

        // Throw a runtime_error with the error string
        throw std::runtime_error("CUDA error: " + std::string(errorString) + ", line: " + std::to_string(line));
    }
}
#define CHECK(call) checkCudaError(call, __LINE__)

__host__ void wrapper_eval(
        void *vertices,
        void *faces,
        void *normals,
        void *segmentVectors,
        void *segmentNormals,
        void *results,
        void *settings,
        unsigned int num_faces,
        FloatType p1,
        FloatType p2,
        FloatType p3,
        bool init) {
    GlobalParams_0 params{};
    params.vertices_0.data = reinterpret_cast<typeof(params.vertices_0.data)>(vertices);
    params.faces_0.data = reinterpret_cast<typeof(params.faces_0.data)>(faces);
    params.normals_0.data = reinterpret_cast<typeof(params.normals_0.data)>(normals);
    params.segmentVectors_0.data = reinterpret_cast<typeof(params.segmentVectors_0.data)>(segmentVectors);
    params.segmentNormals_0.data = reinterpret_cast<typeof(params.segmentNormals_0.data)>(segmentNormals);
    params.results_0.data = reinterpret_cast<typeof(params.results_0.data)>(results);
    params.params_0 = reinterpret_cast<typeof(params.params_0)>(settings);

    cudaMemcpyToSymbol(SLANG_globalParams, &params, sizeof(GlobalParams_0));

    Params_0 params_cpu{};
    params_cpu.num_faces_0 = num_faces;
    params_cpu.point_0.x = p1;
    params_cpu.point_0.y = p2;
    params_cpu.point_0.z = p3;
    params_cpu.init_done_0 = init;

    cudaMemcpy(settings, &params_cpu, sizeof(params_cpu), cudaMemcpyHostToDevice);

    dim3 blockSize(256, 1, 1);
    dim3 gridSize((num_faces + blockSize.x - 1) / blockSize.x,
                  1,
                  1);

    run_eval<<<gridSize, blockSize>>>();

    cudaDeviceSynchronize();
    CHECK(cudaGetLastError());
}
