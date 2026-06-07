#include "Impl_Cuda.cuh"
#include <iostream>
#include <cuda_runtime.h>
#include <math.h>
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

#include <iostream>

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

        cudaMalloc(&positions, sizeof(float3) * size);
        cudaMalloc(&velocities, sizeof(float3) * size);
        cudaMalloc(&forces, sizeof(float3) * size);
        cudaMalloc(&oldForces, sizeof(float3) * size);

        cudaMemcpy(positions, positionsHost.data(), sizeof(float3) * size, cudaMemcpyHostToDevice);
        cudaMemcpy(velocities, velocitiesHost.data(), sizeof(float3) * size, cudaMemcpyHostToDevice);
        cudaMemcpy(forces, forcesHost.data(), sizeof(float3) * size, cudaMemcpyHostToDevice);
        cudaMemset(oldForces, 0.0, sizeof(float3) * size);
    }

    template <typename FloatType>
    CudaParticleSoA<FloatType>::~CudaParticleSoA() {
        cudaFree(positions);
        cudaFree(velocities);
        cudaFree(forces);
        cudaFree(oldForces);
    }

    template <typename FloatType>
    std::vector<Particle<FloatType>> CudaParticleSoA<FloatType>::toParticles() {
        std::vector<Particle<FloatType>> particles{_ref};
        cudaMemcpy(positionsHost.data(), positions, sizeof(float3) * _ref.size(), cudaMemcpyDeviceToHost);
        cudaMemcpy(velocitiesHost.data(), velocities, sizeof(float3) * _ref.size(), cudaMemcpyDeviceToHost);
        cudaMemcpy(forcesHost.data(), forces, sizeof(float3) * _ref.size(), cudaMemcpyDeviceToHost);
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

    __global__ void update_positions(
        float3* positions, const float3* velocities, 
        float3* forces, 
        float3* oldForces, 
        const float3 globalForce, 
        const float deltaT, 
        const size_t numParticles 
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

    __global__ void compute_forces(
        const float3* __restrict__ positions,
        float3* __restrict__ forces,
        const int* verletLists,
        const int* starts, 
        const size_t numParticles
    ) {
        const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= numParticles) {
            return;
        }

        float3 fi = make_float3(0.f, 0.f, 0.f);
        size_t start = starts[i];
        size_t end = starts[i + 1];
        for (size_t k = start; k < end; ++k) {
            size_t j = verletLists[k];
           
            if (i >= j) continue; //N3L via natural ordering of indicies

            const float sigma = 1.0f;
            const float sigmaSquared = sigma * sigma;
            const float epsilon24 = 24.0f; // 1.0 * 24.0

            const float3 dr = make_float3_sub(positions[i], positions[j]);
            const float dr2 = dot3(dr, dr);

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
        }
        atomicAdd(&forces[i].x, fi.x);
        atomicAdd(&forces[i].y, fi.y);
        atomicAdd(&forces[i].z, fi.z);
    }

    __global__ void get_number_of_neighbors(int* starts, float3* positions, float cutoff_radius, float verlet_skin, size_t numParticles) {
        const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= numParticles) {
            return;
        }

        int neighbors = 0;
        float3 pi = positions[i];
        for (size_t j = 0; j < numParticles; j++) {
            if (i == j) continue;
            const float3 dr = make_float3_sub(pi, positions[j]);
            const float dr2 = dot3(dr, dr);
            if (std::sqrt(dr2) <= cutoff_radius + verlet_skin) {
                neighbors++;
            }
        }

        starts[i + 1] = neighbors;
    }

    __global__ void update_starts(int* starts, size_t numParticles) {
        thrust::inclusive_scan(thrust::device, starts, starts + (numParticles + 1), starts);
    }

    __global__ void make_verlet_lists(int* verletLists, int* starts, float3* positions, size_t numParticles, float cutoff_radius, float verlet_skin) {
        const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= numParticles) {
            return;
        }

        float3 pi = positions[i];
        size_t base = starts[i];
        int offset = 0;
        for (size_t j = 0; j < numParticles; j++) {
            if (i == j) continue;
            const float3 dr = make_float3_sub(pi, positions[j]);
            const float dr2 = dot3(dr, dr);
            if (std::sqrt(dr2) <= cutoff_radius + verlet_skin) {
                verletLists[base + offset] = j;
                offset++;
            }
        }
    }

    template<typename FloatType>
    ImplCuda<FloatType>::ImplCuda(const ParticleSimulationConfig<FloatType> &config) 
        : _config{config}
        , _globalForce{_config.globalForce[0], _config.globalForce[1], _config.globalForce[2]}
    {
        const size_t size = _config.size;
        constexpr unsigned int WARP_SIZE = 32;
        constexpr unsigned int MAX_THREADS = 1024;

        if (size <= MAX_THREADS) {
            _blockSize = size;
        } else {
            int minGridSize = 0;
            cudaOccupancyMaxPotentialBlockSize(&minGridSize, &_blockSize, reinterpret_cast<void *>(update_positions), 0, size);
        }
        _gridSize = util::ceilDiv<unsigned int>(size, _blockSize);
        
        cudaMalloc(&starts, sizeof(int) * (size + 1)); // +1 so the last element is the total number of neighbors
        cudaMemset(starts, 0, sizeof(int) * (size + 1));
    }

    template<typename FloatType>
    ImplCuda<FloatType>::~ImplCuda() {
        cudaFree(starts);
        cudaFree(verletLists);
    }

    template<typename FloatType>
    void ImplCuda<FloatType>::updatePositionsAndResetForce() {
        const size_t size = _config.size;
        const auto dt = static_cast<FloatType>(_config.deltaT);
        const auto &globalForce = _config.globalForce;
        const float cutoff_radius = _config.cutoff_radius;
        const float verlet_skin = _config.verlet_skin;
        const size_t frequency = _config.frequency;
        auto &force = _particles->forces;
        auto &oldForce = _particles->oldForces;
        auto &velocity = _particles->velocities;
        auto &position = _particles->positions;

        float elapsedTime;
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start);
        update_positions<<<_gridSize, _blockSize>>>(position, velocity, force, oldForce, _globalForce, dt, size); 
        // every frequency iterations update verlet lists
        if (iteration % frequency == 0) {
            if (verletLists != nullptr) {
                cudaFree(verletLists);
            }
            get_number_of_neighbors<<<_gridSize, _blockSize>>>(starts, position, cutoff_radius, verlet_skin, size);
            update_starts<<<1,1>>>(starts, size);
            size_t num_neighbors = 0;
            cudaMemcpy(&num_neighbors, &starts[size], sizeof(int), cudaMemcpyDeviceToHost);
            if (num_neighbors != 0) {
                cudaMalloc(&verletLists, sizeof(int) * num_neighbors);
                make_verlet_lists<<<_gridSize, _blockSize>>>(verletLists, starts, position, size, cutoff_radius, verlet_skin);
            }
        }
        cudaEventRecord(stop);

        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsedTime, start, stop);
        _timings.positionUpdateForceResetTime += (elapsedTime * 1e6);
    }

    template<typename FloatType>
    void ImplCuda<FloatType>::computeForces() {
        const size_t size = _config.size;
        auto &force = _particles->forces;
        auto &position = _particles->positions;

        float elapsedTime;
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start);
        compute_forces<<<_gridSize, _blockSize>>>(position, force, verletLists, starts, size);
        cudaEventRecord(stop);

        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsedTime, start, stop);
        _timings.forceUpdateTime += (elapsedTime * 16);
    }

    template<typename FloatType>
    void ImplCuda<FloatType>::updateVelocities() {
        const size_t size = _config.size;
        constexpr size_t dim = 3;
        const auto dt = static_cast<FloatType>(_config.deltaT);
        auto &force = _particles->forces;
        auto &oldForce = _particles->oldForces;
        auto &velocity = _particles->velocities;

        float elapsedTime;
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start);
        update_velocities<<<_gridSize, _blockSize>>>(velocity, force, oldForce, dt, size);
        cudaEventRecord(stop);
        
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsedTime, start, stop);

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

        for (iteration = 0; iteration < _config.numberTimeSteps; ++iteration) {
            updatePositionsAndResetForce();
            computeForces();
            updateVelocities();
    
            std::vector<Particle<FloatType>> particles = _particles.value().toParticles();
#ifndef NDEBUG
            std::cout<<"Iteration: "<< iteration << std::endl;
            for (auto p : particles) {
                std::cout<<p<<std::endl;
            }
#endif
#ifdef PPB_ENABLE_VTK
            std::vector<Particle<FloatType>> particles = _particles.value().toParticles();
            plotParticles(particles, "VTK", i);
#endif
        }
        return std::make_pair(_particles->toParticles(), _timings);
    }


    template class ImplCuda<float>;
  } // namespace ppb
