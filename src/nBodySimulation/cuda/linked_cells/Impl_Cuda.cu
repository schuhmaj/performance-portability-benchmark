#include "Impl_Cuda.cuh"
#include <iostream>
#include <cuda_runtime.h>
#include <math.h>
#include <algorithm>
#include <thrust/scan.h>
#include <thrust/execution_policy.h>

#ifdef PPB_ENABLE_VTK
#include <vtkCellArray.h>
#include <vtkDoubleArray.h>
#include <vtkFloatArray.h>
#include <vtkIntArray.h>
#include <vtkPointData.h>
#include <vtkXMLUnstructuredGridWriter.h>
#include <vtkSmartPointer.h>
#include <vtkUnstructuredGrid.h>
#include <iomanip>
#include <sstream>
#endif

//taken from https://leimao.github.io/blog/Proper-CUDA-Error-Checking/ (last accessed 13.6.26, 19:44)
#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)
void check(cudaError_t err, char const* func, char const* file, int line)
{
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line
                  << std::endl;
        std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
        // We don't exit when we encounter CUDA errors in this example.
        // std::exit(EXIT_FAILURE);
    }
}

namespace ppb {

    template <typename FloatType>
    CudaParticleSoA<FloatType>::CudaParticleSoA(const std::vector<Particle<FloatType>> &particles)
        : positionsHost{particles.size()}
        , velocitiesHost{particles.size()}
        , forcesHost{particles.size()}
        , _ref{particles}
    {
        const size_t size = particles.size();
        for (size_t i = 0; i < size; ++i) {
            positionsHost[i] = {particles[i].getPosition()[0], particles[i].getPosition()[1], particles[i].getPosition()[2]};
            velocitiesHost[i] = {particles[i].getVelocity()[0], particles[i].getVelocity()[1], particles[i].getVelocity()[2]};
            forcesHost[i] = {particles[i].getForce()[0], particles[i].getForce()[1], particles[i].getForce()[2]};
        }

        CHECK_CUDA_ERROR(cudaMalloc(&positions, sizeof(float3) * size));
        CHECK_CUDA_ERROR(cudaMalloc(&velocities, sizeof(float3) * size));
        CHECK_CUDA_ERROR(cudaMalloc(&forces, sizeof(float3) * size));
        CHECK_CUDA_ERROR(cudaMalloc(&oldForces, sizeof(float3) * size));

        CHECK_CUDA_ERROR(cudaMemcpy(positions, positionsHost.data(), sizeof(float3) * size, cudaMemcpyHostToDevice));
        CHECK_CUDA_ERROR(cudaMemcpy(velocities, velocitiesHost.data(), sizeof(float3) * size, cudaMemcpyHostToDevice));
        CHECK_CUDA_ERROR(cudaMemcpy(forces, forcesHost.data(), sizeof(float3) * size, cudaMemcpyHostToDevice));
        CHECK_CUDA_ERROR(cudaMemset(oldForces, 0.0, sizeof(float3) * size));
    }

    template <typename FloatType>
    CudaParticleSoA<FloatType>::~CudaParticleSoA() {
        CHECK_CUDA_ERROR(cudaFree(positions));
        CHECK_CUDA_ERROR(cudaFree(velocities));
        CHECK_CUDA_ERROR(cudaFree(forces));
        CHECK_CUDA_ERROR(cudaFree(oldForces));
    }

    template <typename FloatType>
    std::vector<Particle<FloatType>> CudaParticleSoA<FloatType>::toParticles() {
        std::vector<Particle<FloatType>> particles{_ref};
        CHECK_CUDA_ERROR(cudaMemcpy(positionsHost.data(), positions, sizeof(float3) * _ref.size(), cudaMemcpyDeviceToHost));
        CHECK_CUDA_ERROR(cudaMemcpy(velocitiesHost.data(), velocities, sizeof(float3) * _ref.size(), cudaMemcpyDeviceToHost));
        CHECK_CUDA_ERROR(cudaMemcpy(forcesHost.data(), forces, sizeof(float3) * _ref.size(), cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < particles.size(); ++i) {
            const float3& position = positionsHost[i];
            const float3& velocity = velocitiesHost[i];
            const float3& force = forcesHost[i];
            particles[i].setPosition({position.x, position.y, position.z});
            particles[i].setVelocity({velocity.x, velocity.y, velocity.z});
            particles[i].setForce({force.x, force.y, force.z});
        }
        return particles;
    }

    template class CudaParticleSoA<float>;
    

    __device__ inline float3 make_float3_add(const float3 a, const float3 b) {
        return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
    }

    __device__ inline float3 make_float3_sub(const float3 a, const float3 b) {
        return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
    }

    __device__ inline float3 make_float3_scale(const float3 v, const float s) {
        return make_float3(v.x * s, v.y * s, v.z * s);
    }

    __device__ inline float dot3(const float3 a, const float3 b) {
        return a.x * b.x + a.y * b.y + a.z * b.z;
    } 

    __global__ void update_cells(int* cells, int* tmp, int* cell_offsets, int* starts, size_t numParticles) {
        const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= numParticles) {
            return;
        } 
        
        size_t idx = starts[tmp[i]];
        size_t offset = cell_offsets[i];
        size_t position = idx + offset;
        cells[position] = i;
    }

    __device__ inline bool is_in_bounds(size_t idx, size_t offset, int x_dim, int y_dim, int z_dim) {
        size_t i = blockIdx.x * blockDim.x + threadIdx.x;
        size_t offset_idx = idx + offset;
        size_t x_idx = idx % x_dim;
        size_t y_idx = (idx / x_dim) % y_dim;
        size_t z_idx = (idx / (x_dim * y_dim));
        size_t x_offset = offset_idx % x_dim;
        size_t y_offset = (offset_idx / x_dim) % y_dim;
        size_t z_offset = (offset_idx / (x_dim * y_dim));

        //printf("THREAD %lu: x_idx: %lu, y_idx: %lu, z_idx: %lu, x_offset: %lu, y_offset: %lu, z_offset: %lu\n", i, x_idx, y_idx, z_idx, x_offset, y_offset, z_offset);
        if (std::abs((int)(x_idx - x_offset)) > 1) return false;
        else if (std::abs((int)(y_idx - y_offset)) > 1) return false;
        else if (std::abs((int)(z_idx - z_offset)) > 1) return false;
        else if (x_offset >= x_dim) return false;
        else if (y_offset >= y_dim) return false;
        else if (z_offset >= z_dim) return false;
        return true;
    }

    //taken from: https://github.com/dangets/cuda_examples/blob/master/clamp_function.cu (last accessed 14.6.26)
    template <typename T>
    inline __device__ T clamp(T val, T vMin, T vMax) {
        return min(max(val, vMin), vMax);
    }


    template<typename FloatType>
    __device__ inline int get_cell_idx(
        size_t particle_idx, 
        const float3* positions,  
        float cell_size, 
        int x_dim, 
        int y_dim, 
        int z_dim,
        FloatType boxMinX,
        FloatType boxMinY,
        FloatType boxMinZ,
        FloatType boxMaxX,
        FloatType boxMaxY,
        FloatType boxMaxZ
    ) {
        int x_idx = clamp<int>(int(std::ceil((positions[particle_idx].x - boxMinX) / cell_size)), 0, x_dim - 1);
        int y_idx = clamp<int>(int(std::ceil((positions[particle_idx].y - boxMinY) / cell_size)), 0, y_dim - 1);
        int z_idx = clamp<int>(int(std::ceil((positions[particle_idx].z - boxMinZ) / cell_size)), 0, z_dim - 1);
        return x_idx + (y_idx * x_dim) + (z_idx * x_dim * y_dim); 
    }

    template<typename FloatType>
    __global__ void update_positions(
        float3* positions, 
        const float3* velocities, 
        float3* forces, 
        float3* oldForces, 
        int* tmp, 
        int* cell_offsets,
        int* starts, 
        const float3 globalForce, 
        const float deltaT, 
        const size_t numParticles, 
        float cell_size,
        int x_dim,
        int y_dim,
        int z_dim,
        FloatType boxMinX,
        FloatType boxMinY,
        FloatType boxMinZ,
        FloatType boxMaxX,
        FloatType boxMaxY,
        FloatType boxMaxZ
    ) {
        const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= numParticles) {
            return;
        }

        constexpr float mass = 1.0;
        const float3 force = forces[i];
        const float3 velocity = velocities[i];
        oldForces[i] = force;
        forces[i] = globalForce;

        const float3 velocityPart = {velocity.x * deltaT, velocity.y * deltaT, velocity.z * deltaT};
        const float tt2m = deltaT * deltaT / (2.0f * mass);
        const float3 forcePart = {force.x * tt2m, force.y * tt2m, force.z * tt2m};
        const float3 displacement = {velocityPart.x + forcePart.x, velocityPart.y + forcePart.y, velocityPart.z + forcePart.z};
        positions[i] = {positions[i].x + displacement.x, positions[i].y + displacement.y, positions[i].z + displacement.z};
        

        int idx = get_cell_idx(i, positions, cell_size, x_dim, y_dim, z_dim, boxMinX, boxMinY, boxMinZ, boxMaxX, boxMaxY, boxMaxZ);
        //printf("IDX: %lu\n", idx);
        int offset = atomicAdd(&starts[idx + 1], 1); //returns the value at starts[idx + 1] *before* adding 1.
        tmp[i] = idx;
        cell_offsets[i] = offset;
    }

    __global__ void update_velocities(float3* velocities, const float3* forces, const float3* oldForces, const float deltaT, const size_t numParticles) {
        const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= numParticles) {
            return;
        }

        constexpr float mass = 1.0;
        const float3 force = forces[i];
        const float3 oldForce = oldForces[i];
        const float3 velocity = velocities[i];

        const float3 forcePart = {force.x + oldForce.x, force.y + oldForce.y, force.z + oldForce.z};
        const float t2m =  deltaT / (2.0f * mass);
        const float3 velChange = {forcePart.x * t2m, forcePart.y * t2m, forcePart.z * t2m};
        velocities[i] = {velocity.x + velChange.x, velocity.y + velChange.y, velocity.z + velChange.z};
    }

    __global__ void printStartsCells(int* starts, int* cells, size_t numCells, size_t numParticles) {
        printf("starts:\n");
        for (size_t j = 0; j <= numCells; j++) {
            printf("%d, ", starts[j]);
        }
        printf("\ncells:");
        for (size_t j = 0; j < numParticles; j++) {
            printf("%d, ", cells[j]);
        }
        printf("\n");
    }

#ifdef PPB_ENABLE_DOMAIN_COLORING
    __global__ void compute_forces_colored(
        int color,
        const float3* __restrict__ positions,
        float3* __restrict__ forces,
        int* cells,
        int* starts, 
        const unsigned int numParticles,
        const int* offsets,
        const int* offsets_colored,
        float cell_size,
        float cutoff_radius,
        int x_dim,
        int y_dim,
        int z_dim
    ) {
        unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
        const size_t num_cells = x_dim * y_dim * z_dim;
        if (i >= util::ceilDiv<size_t>(num_cells, 8)) {
            return;
        }

        int x_thread = i % x_dim;
        int y_thread = (i / x_dim) % y_dim;
        int z_thread = (i / (x_dim * y_dim));
        int x_cell = 2 * x_thread + (color % 2);
        int y_cell = 2 * y_thread + (color % 4);
        int z_cell = 2 * z_thread + (color % 8);
        int idx = x_cell + (y_cell * x_dim) + (z_cell * x_dim * y_dim);
        int startBaseCell = starts[idx];
        int endBaseCell = starts[idx + 1];
        float3 fi = make_float3(0.f, 0.f, 0.f);
        //printf("color: %d, idx: %lu, startBaseCell: %lu, endBaseCell: %lu\n", color, idx, startBaseCell, endBaseCell);
        for (i = startBaseCell; i < endBaseCell; i++) {
        for (size_t o = 0; o < 8; o++) {
            size_t offset = offsets[offsets_colored[o]];
            if (!is_in_bounds(idx, offset, x_dim, y_dim, z_dim)) {
                continue;
            }
            idx += offset;
            int start = starts[idx];
            int end = starts[idx + 1];
            //printf("iterating through cell idx: %lu at offset: %lu, starts at: %lu, ends at: %lu\n", idx, offset, start, end);
            for (int k = start; k < end; k++) {
                int j = cells[k];

                //N3L via natural ordering of indicies (only necessary in same cell)
                if (offset == 0 && i >= j) {
                    continue;
                }
                const float3 dr = make_float3_sub(positions[i], positions[j]);
                const float dr2 = dot3(dr, dr); 

                // = here too because this way we never get into a race condition with another cell of the same color
                if (std::sqrt(dr2) >= cutoff_radius) { 
                    continue;
                }

                const float sigma = 1.0f;
                const float sigmaSquared = sigma * sigma;
                const float epsilon24 = 24.0f; // 1.0 * 24.0

                const float invdr2 = 1.0f / dr2;
                float lj6 = sigmaSquared * invdr2;
                lj6 = lj6 * lj6 * lj6;
                const float lj12 = lj6 * lj6;
                const float lj12m6 = lj12 - lj6;
                const float fac = epsilon24 * (lj12 + lj12m6) * invdr2;
                
                const float3 f = make_float3_scale(dr, fac);
                fi = make_float3_add(fi, f); 
                forces[j] = make_float3_sub(forces[j], f);
                printf("updated forces of i: %lu <-> j: %lu\n", i, j);
            }
            idx -= offset;
        }
        forces[i] = make_float3_add(forces[i], fi);
    }
    }
#else
    template<typename FloatType>
    __global__ void compute_forces(
        const float3* __restrict__ positions,
        float3* __restrict__ forces,
        const int* __restrict__ cells,
        const int* __restrict__ starts, 
        const unsigned int numParticles,
        const int* offsets,
        float cell_size,
        float cutoff_radius,
        int x_dim,
        int y_dim,
        int z_dim,
        FloatType boxMinX,
        FloatType boxMinY,
        FloatType boxMinZ,
        FloatType boxMaxX,
        FloatType boxMaxY,
        FloatType boxMaxZ
    ) {
        const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= numParticles) {
            return;
        }

        float3 fi = make_float3(0.f, 0.f, 0.f);
        size_t idx = get_cell_idx<FloatType>(i, positions, cell_size, x_dim, y_dim, z_dim, boxMinX, boxMinY, boxMinZ, boxMaxX, boxMaxY, boxMaxZ);
        //printf("IDX: %lu\n", idx);
        for (size_t offset = 0; offset < 27; offset++) {
            if (!is_in_bounds(idx, offsets[offset], x_dim, y_dim, z_dim)) continue; 
            //printf("In bounds for particle %u: offset %lu\n", i, offset);
            idx += offsets[offset];
        //printf("IDX + offset: %lu\n", idx);
            size_t start = starts[idx];
            size_t end = starts[idx + 1];
            for (size_t k = start; k < end; k++) {
               // printf("Thread %u TRYING access!\n", i);
                size_t j = cells[k];
               // printf("Thread %u access worked!\n", i);
                if (i >= j) continue; //N3L via natural ordering of indicies

                const float3 dr = make_float3_sub(positions[i], positions[j]);
                const float dr2 = dot3(dr, dr);
                if (std::sqrt(dr2) >= cutoff_radius) continue; // = here too because less atomics in domain coloring

                const float sigma = 1.0f;
                const float sigmaSquared = sigma * sigma;
                const float epsilon24 = 24.0f; // 1.0 * 24.0

                const float invdr2 = 1.0f / dr2;
                float lj6 = sigmaSquared * invdr2;
                lj6 = lj6 * lj6 * lj6;
                const float lj12 = lj6 * lj6;
                const float lj12m6 = lj12 - lj6;
                const float fac = epsilon24 * (lj12 + lj12m6) * invdr2;
                
                const float3 f = make_float3_scale(dr, fac);
                fi = make_float3_add(fi, f); 
                atomicAdd(&forces[j].x, f.x * -1.0f);
                atomicAdd(&forces[j].y, f.y * -1.0f);
                atomicAdd(&forces[j].z, f.z * -1.0f);
                //printf("Thread %u access DONE!\n", i);
            }
            idx -= offsets[offset];
            //printf("Thread %u offset %lu DONE\n", i, offset);
        }

        atomicAdd(&forces[i].x, fi.x);
        atomicAdd(&forces[i].y, fi.y);
        atomicAdd(&forces[i].z, fi.z);
    }
#endif

    template<typename FloatType>
    ImplCuda<FloatType>::ImplCuda(const ParticleSimulationConfig<FloatType> &config) 
        : _config{config}
    {
        //DO NOT PUT THESE LINES IN THE CONSTRUCTOR CAUSE IT BREAKS FOR SOME WEIRD REASON
        _globalForce.x = _config.globalForce[0];
        _globalForce.y = _config.globalForce[1];
        _globalForce.z = _config.globalForce[2];
        //-------------------------------------------------------------------------------

        const size_t size = _config.size;
        constexpr unsigned int WARP_SIZE = 32;
        constexpr unsigned int MAX_THREADS = 1024;

        printf("GLOBAL FORCE INIT: x: %f, y: %f, z: %f\n", _config.globalForce[0], _config.globalForce[1], _config.globalForce[2]);
        printf("GLOBAL FORCE INIT: x: %f, y: %f, z: %f\n", _globalForce.x, _globalForce.y, _globalForce.z);

        if (size <= MAX_THREADS) {
            _blockSize = size;
        } else {
            int minGridSize = 0;
            CHECK_CUDA_ERROR(cudaOccupancyMaxPotentialBlockSize(&minGridSize, &_blockSize, reinterpret_cast<void *>(update_positions<FloatType>), 0, size));
        }
        _gridSize = util::ceilDiv<unsigned int>(size, _blockSize);
        
        x_dim = (_config.boxMax[0] - _config.boxMin[0]) / _config.cell_size;
        y_dim = (_config.boxMax[1] - _config.boxMin[1]) / _config.cell_size;
        z_dim = (_config.boxMax[2] - _config.boxMin[2]) / _config.cell_size;
        

#ifdef PPB_ENABLE_DOMAIN_COLORING
        size_t number_of_cells_with_same_color = util::ceilDiv<size_t>(x_dim * y_dim * z_dim, 8);
        int offsets_coloredDeclared[8] = { 13, 14, 16, 17, 22, 23, 25, 26 };
        memcpy(offsets_colored, &offsets_coloredDeclared, 8 * sizeof(int));
        if (number_of_cells_with_same_color <= MAX_THREADS) {
            _blockSizeColored = number_of_cells_with_same_color;
        } else {
            CHECK_CUDA_ERROR(cudaOccupancyMaxPotentialBlockSize(&_gridSizeColored, &_blockSizeColored, reinterpret_cast<void *>(compute_forces_colored), 0, 0));
        }
        _gridSizeColored = util::ceilDiv<unsigned int>(number_of_cells_with_same_color, _blockSizeColored);
        std::cout<<"_gridSizeColored: "<<_gridSizeColored<<", _blockSizeColored: "<<_blockSizeColored<<std::endl;
#endif
        int offsetsDeclared[27] = {
            //front section
            -((x_dim + 1) * y_dim) - 1, -((x_dim + 1) * y_dim), -((x_dim + 1) * y_dim) + 1,
            -(x_dim * y_dim) - 1, -(x_dim * y_dim), (x_dim * y_dim) + 1,
            -((x_dim - 1) * y_dim) - 1, -((x_dim - 1) * y_dim), -((x_dim - 1) * y_dim) + 1,
            //mid section
            -x_dim - 1, -x_dim, -x_dim + 1,
            -1, 0, 1,
            x_dim - 1, x_dim, x_dim + 1,
            //back section
            ((x_dim - 1) * y_dim) - 1, ((x_dim - 1) * y_dim), ((x_dim - 1) * y_dim) + 1,
            (x_dim * y_dim) - 1, (x_dim * y_dim), (x_dim * y_dim) + 1,
            ((x_dim + 1) * y_dim) - 1, ((x_dim + 1) * y_dim), ((x_dim + 1) * y_dim) + 1
        };
        memcpy(offsets, &offsetsDeclared, 27 * sizeof(int));
        
        const size_t num_cells = x_dim * y_dim * z_dim;        
        CHECK_CUDA_ERROR(cudaMalloc(&cells, sizeof(int) * size));
        CHECK_CUDA_ERROR(cudaMalloc(&tmp, sizeof(int) * size));
        CHECK_CUDA_ERROR(cudaMalloc(&cell_offsets, sizeof(int) * size));
        CHECK_CUDA_ERROR(cudaMalloc(&starts, sizeof(int) * (num_cells + 1)));
        CHECK_CUDA_ERROR(cudaMemset(cells, 0, sizeof(int) * size));
        CHECK_CUDA_ERROR(cudaMemset(tmp, 0, sizeof(int) * size));
        CHECK_CUDA_ERROR(cudaMemset(cell_offsets, 0, sizeof(int) * size));
        CHECK_CUDA_ERROR(cudaMemset(starts, 0, sizeof(int) * (num_cells + 1)));
    }

    template<typename FloatType>
    ImplCuda<FloatType>::~ImplCuda() {
        CHECK_CUDA_ERROR(cudaFree(starts));
        CHECK_CUDA_ERROR(cudaFree(cells));
        CHECK_CUDA_ERROR(cudaFree(cell_offsets));
        CHECK_CUDA_ERROR(cudaFree(tmp));
    }

    __global__ void update_starts(int* starts, size_t num_cells) {
        thrust::inclusive_scan(thrust::device, starts, starts + (num_cells + 1), starts);
    }

    template<typename FloatType>
    void ImplCuda<FloatType>::updatePositionsAndResetForce() {
        const size_t size = _config.size;
        const float cell_size = _config.cell_size;
        const auto dt = static_cast<FloatType>(_config.deltaT);
        const auto &globalForce = _config.globalForce;
        const FloatType boxMinX = _config.boxMin[0];
        const FloatType boxMinY = _config.boxMin[1];
        const FloatType boxMinZ = _config.boxMin[2];
        const FloatType boxMaxX = _config.boxMax[0];
        const FloatType boxMaxY = _config.boxMax[1];
        const FloatType boxMaxZ = _config.boxMax[2];
        auto &force = _particles->forces;
        auto &oldForce = _particles->oldForces;
        auto &velocity = _particles->velocities;
        auto &position = _particles->positions;
        const size_t num_cells = x_dim * y_dim * z_dim;
        
        CHECK_CUDA_ERROR(cudaMemset(starts, 0.0, sizeof(int) * (num_cells + 1)));
        CHECK_CUDA_ERROR(cudaMemset(cells, 0.0, sizeof(int) * size));
        CHECK_CUDA_ERROR(cudaMemset(cell_offsets, 0.0, sizeof(int) * size));
        CHECK_CUDA_ERROR(cudaMemset(tmp, 0.0, sizeof(int) * size));

        float elapsedTime;
        cudaEvent_t start, stop;
        CHECK_CUDA_ERROR(cudaEventCreate(&start));
        CHECK_CUDA_ERROR(cudaEventCreate(&stop));

        CHECK_CUDA_ERROR(cudaEventRecord(start));
        update_positions<FloatType><<<_gridSize, _blockSize>>>(position, velocity, force, oldForce, tmp, cell_offsets, 
            starts, _globalForce, dt, size, cell_size, x_dim, y_dim, z_dim, boxMinX, boxMinY, boxMinZ, boxMaxX, boxMaxY, boxMaxZ);        
        //printf("After update_positions:\n");
        //printStartsCells<<<1,1>>>(starts, cells, num_cells, size); 
        //cudaDeviceSynchronize();

        update_starts<<<1,1>>>(starts, num_cells); //bit of a hacky workaround. maybe make this prettier 
        //printf("After update_starts:\n");
        //printStartsCells<<<1,1>>>(starts, cells, num_cells, size); 
        //cudaDeviceSynchronize();
      
        update_cells<<<_gridSize, _blockSize>>>(cells, tmp, cell_offsets, starts, size); 
        //printf("After update_cells:\n");
        //printStartsCells<<<1,1>>>(starts, cells, num_cells, size); 
        ///cudaDeviceSynchronize();
        
        CHECK_CUDA_ERROR(cudaEventRecord(stop));

        CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
        CHECK_CUDA_ERROR(cudaEventElapsedTime(&elapsedTime, start, stop));
        _timings.positionUpdateForceResetTime += (elapsedTime * 1e6);
    }

    template<typename FloatType>
    void ImplCuda<FloatType>::computeForces() {
        const size_t size = _config.size;
        const float cutoff_radius = _config.cutoff_radius;
        const float cell_size = _config.cell_size;
        const FloatType boxMinX = _config.boxMin[0];
        const FloatType boxMinY = _config.boxMin[1];
        const FloatType boxMinZ = _config.boxMin[2];
        const FloatType boxMaxX = _config.boxMax[0];
        const FloatType boxMaxY = _config.boxMax[1];
        const FloatType boxMaxZ = _config.boxMax[2];
        auto &force = _particles->forces;
        auto &position = _particles->positions;

        float elapsedTime;
        cudaEvent_t start, stop;
        CHECK_CUDA_ERROR(cudaEventCreate(&start));
        CHECK_CUDA_ERROR(cudaEventCreate(&stop));
        CHECK_CUDA_ERROR(cudaEventRecord(start));
#ifdef PPB_ENABLE_DOMAIN_COLORING
        for (size_t color = 0; color < 8; color++) {
            compute_forces_colored<<<_gridSizeColored, _blockSizeColored>>>(color, position, force, 
                cells, starts, size, offsets, offsets_colored, cell_size, cutoff_radius, x_dim, y_dim, z_dim);
        }
#else
        compute_forces<FloatType><<<_gridSize, _blockSize>>>(position, force, cells, starts, size, offsets, 
            cell_size, cutoff_radius, x_dim, y_dim, z_dim, boxMinX, boxMinY, boxMinZ, boxMaxX, boxMaxY, boxMaxZ);
#endif
        CHECK_CUDA_ERROR(cudaEventRecord(stop));

        CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
        CHECK_CUDA_ERROR(cudaEventElapsedTime(&elapsedTime, start, stop));
        _timings.forceUpdateTime += (elapsedTime * 16);
    }

    template<typename FloatType>
    void ImplCuda<FloatType>::updateVelocities() {
        const size_t size = _config.size;
        const auto dt = static_cast<FloatType>(_config.deltaT);
        auto &force = _particles->forces;
        auto &oldForce = _particles->oldForces;
        auto &velocity = _particles->velocities;

        float elapsedTime;
        cudaEvent_t start, stop;
        CHECK_CUDA_ERROR(cudaEventCreate(&start));
        CHECK_CUDA_ERROR(cudaEventCreate(&stop));

        CHECK_CUDA_ERROR(cudaEventRecord(start));
        update_velocities<<<_gridSize, _blockSize>>>(velocity, force, oldForce, dt, size);
        CHECK_CUDA_ERROR(cudaEventRecord(stop));
        
        CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
        CHECK_CUDA_ERROR(cudaEventElapsedTime(&elapsedTime, start, stop));

        _timings.velocityUpdateTime += (elapsedTime * 1e6);
    }

#ifdef PPB_ENABLE_VTK
    //this only works if FloatType is float. does not work for double for now.
    template<typename FloatType>
    void plotParticles(std::vector<Particle<FloatType>>& particles, const std::string& filename, int iteration) {
        // Initialize points
        auto points = vtkSmartPointer<vtkPoints>::New();

        // Create and configure data arrays
        vtkNew<vtkFloatArray> velocity_array;
        velocity_array->SetName("velocity");
        velocity_array->SetNumberOfComponents(3);

        vtkNew<vtkFloatArray> force_array;
        force_array->SetName("force");
        force_array->SetNumberOfComponents(3);

        for (auto& p : particles) {
            const float pos[3] = {p.getPosition()[0], p.getPosition()[1], p.getPosition()[2]};
            const float vel[3] = {p.getVelocity()[0], p.getVelocity()[1], p.getVelocity()[2]};
            const float force[3] = {p.getForce()[0], p.getForce()[1], p.getForce()[2]}; 
            points->InsertNextPoint(pos);
            velocity_array->InsertNextTuple(vel);
            force_array->InsertNextTuple(force);
        }

        // Set up the grid
        auto grid = vtkSmartPointer<vtkUnstructuredGrid>::New();
        grid->SetPoints(points);

        // Add arrays to the grid
        grid->GetPointData()->AddArray(velocity_array);
        grid->GetPointData()->AddArray(force_array);

        // Create filename with iteration number
        std::stringstream strstr;
        strstr << filename << "_" << std::setfill('0') << std::setw(4) << iteration << ".vtu";

        // Create writer and set data
        vtkNew<vtkXMLUnstructuredGridWriter> writer;
        writer->SetFileName(strstr.str().c_str());
        writer->SetInputData(grid);
        writer->SetDataModeToBinary();

        // Write the file
        writer->Write();
    }
#endif

    template<typename FloatType>
    std::pair<std::vector<Particle<FloatType>>, ParticleSimulationTimings> ImplCuda<FloatType>::simulate(const std::vector<Particle<FloatType>> &particles) {
        _timings.reset();
        _particles.emplace(particles);

        for (int i = 0; i < _config.numberTimeSteps; ++i) {
            updatePositionsAndResetForce();
            computeForces();
            updateVelocities();
#ifdef PPB_ENABLE_VTK
            std::vector<Particle<FloatType>> particles = _particles.value().toParticles();
            plotParticles(particles, "VTK", i);
#endif
        }
        return std::make_pair(_particles->toParticles(), _timings);
    }


    template class ImplCuda<float>;
  } // namespace ppb
