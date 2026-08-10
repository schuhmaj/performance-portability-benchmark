#include "Impl_OpenCL.h"
#include "common/UtilityFloatArithmetic.h"

namespace ppb {

    template <typename FloatType>
    OpenCLParticleSoA<FloatType>::OpenCLParticleSoA(const std::vector<Particle<FloatType>> &ref, cl_context &context)
        : _ref{ref}
        , positions{}
        , velocities{}
        , forces{}
        , oldForces{}
        , positionsHost{ref.size()}
        , velocitiesHost{ref.size()}
        , forcesHost{ref.size()}
    {
        const size_t size = ref.size();
        const auto transformVec4 = [](auto getter) {
            return [getter](const auto& particle) {
                const auto arr = std::invoke(getter, particle);
                return cl_vec4{arr[0], arr[1], arr[2], static_cast<cl_scalar>(0)};
            };
        };
        std::transform(ref.begin(), ref.end(), positionsHost.begin(), transformVec4(&Particle<FloatType>::getPosition));
        std::transform(ref.begin(), ref.end(), velocitiesHost.begin(), transformVec4(&Particle<FloatType>::getVelocity));
        std::transform(ref.begin(), ref.end(), forcesHost.begin(), transformVec4(&Particle<FloatType>::getForce));

        cl_int err = 0;
        positions = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, size * sizeof(cl_vec4),
                                const_cast<cl_vec4*>(positionsHost.data()), &err);
        velocities = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, size * sizeof(cl_vec4),
                                const_cast<cl_vec4*>(velocitiesHost.data()), &err);
        forces = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, size * sizeof(cl_vec4),
                                const_cast<cl_vec4*>(forcesHost.data()), &err);
        oldForces = clCreateBuffer(context, CL_MEM_READ_WRITE, size * sizeof(cl_vec4), nullptr, nullptr);
    }

    template <typename FloatType>
    OpenCLParticleSoA<FloatType>::~OpenCLParticleSoA() {
        clReleaseMemObject(positions);
        clReleaseMemObject(velocities);
        clReleaseMemObject(forces);
        clReleaseMemObject(oldForces);
    }


    template <typename FloatType>
    std::vector<Particle<FloatType>> OpenCLParticleSoA<FloatType>::toParticles(cl_command_queue &queue) {
        int err;
        const size_t size = _ref.size();
        std::vector<Particle<FloatType>> particles{_ref};

        err = clEnqueueReadBuffer(queue, positions, CL_FALSE, 0,  size * sizeof(cl_vec4),positionsHost.data(), 0, nullptr, nullptr);
        err |= clEnqueueReadBuffer(queue, velocities, CL_FALSE, 0,  size * sizeof(cl_vec4),velocitiesHost.data(), 0, nullptr, nullptr);
        err |= clEnqueueReadBuffer(queue, forces, CL_FALSE, 0,  size * sizeof(cl_vec4),forcesHost.data(), 0, nullptr, nullptr);
        if (err != CL_SUCCESS) throw std::runtime_error("ReadBuffer failed");
        clFinish(queue);
        const auto toArray = [](const cl_vec4 &v) -> std::array<FloatType, 3> {
            return {v.x, v.y, v.z};
        };
        for (size_t i = 0; i < particles.size(); ++i) {
            particles[i].setPosition(toArray(positionsHost[i]));
            particles[i].setVelocity(toArray(velocitiesHost[i]));
            particles[i].setForce(toArray(forcesHost[i]));
        }
        return particles;
    }


    template <typename FloatType>
    ImplOpenCL<FloatType>::ImplOpenCL(const ParticleSimulationConfig<FloatType> &config)
        : _config{config}
        , _numParticles{static_cast<unsigned int>(config.size)}
        , _deltaT{config.deltaT}
        , _globalForce{config.globalForce[0], config.globalForce[1], config.globalForce[2], 0.0}
    {
        gpu = opencl_utility::getFirstGPU();
        cl_int err;

        context = clCreateContext(0, 1, &gpu, nullptr, nullptr, &err);
        queue = clCreateCommandQueue(context, gpu, CL_QUEUE_PROFILING_ENABLE, &err);

        std::string kernelSource;
        const char *kernelProg = KERNEL_SOURCE;
        program = clCreateProgramWithSource(context, 1, &kernelProg, nullptr, &err);
        // FloatType is injected via the build args (OPENCL_COMPILE_ARGS) per PPB_FloatType.
        err = clBuildProgram(program, 0, nullptr, OPENCL_COMPILE_ARGS, nullptr, nullptr);
        kernelPositionUpdate = clCreateKernel(program, "update_positions_reset_forces", &err);
        kerneVelocityUpdate = clCreateKernel(program, "update_velocities", &err);
        kernelForceUpdate = clCreateKernel(program, "compute_forces", &err);
        if (err != CL_SUCCESS) throw std::runtime_error("Kernel creation failed");

    }

    template <typename FloatType>
    ImplOpenCL<FloatType>::~ImplOpenCL() {
        clReleaseProgram(program);
        clReleaseKernel(kernelPositionUpdate);
        clReleaseKernel(kerneVelocityUpdate);
        clReleaseKernel(kernelForceUpdate);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        clReleaseDevice(gpu);
    }


    template <typename FloatType>
    void ImplOpenCL<FloatType>::init(const std::vector<Particle<FloatType>> &particles) {
        _timings.reset();
        _particles.emplace(particles, context);
        clSetKernelArg(kernelPositionUpdate, 0, sizeof(cl_mem), &_particles->positions);
        clSetKernelArg(kernelPositionUpdate, 1, sizeof(cl_mem), &_particles->velocities);
        clSetKernelArg(kernelPositionUpdate, 2, sizeof(cl_mem), &_particles->forces);
        clSetKernelArg(kernelPositionUpdate, 3, sizeof(cl_mem), &_particles->oldForces);
        clSetKernelArg(kernelPositionUpdate, 4, sizeof(cl_vec4), &_globalForce);
        clSetKernelArg(kernelPositionUpdate, 5, sizeof(cl_scalar), &_deltaT);
        clSetKernelArg(kernelPositionUpdate, 6, sizeof(cl_uint), &_numParticles);

        clSetKernelArg(kerneVelocityUpdate, 0, sizeof(cl_mem), &_particles->velocities);
        clSetKernelArg(kerneVelocityUpdate, 1, sizeof(cl_mem), &_particles->forces);
        clSetKernelArg(kerneVelocityUpdate, 2, sizeof(cl_mem), &_particles->oldForces);
        clSetKernelArg(kerneVelocityUpdate, 3, sizeof(cl_scalar), &_deltaT);
        clSetKernelArg(kerneVelocityUpdate, 4, sizeof(cl_uint), &_numParticles);

        clSetKernelArg(kernelForceUpdate, 0, sizeof(cl_mem), &_particles->positions);
        clSetKernelArg(kernelForceUpdate, 1, sizeof(cl_mem), &_particles->forces);
        clSetKernelArg(kernelForceUpdate, 2, sizeof(cl_uint), &_numParticles);
    }


    template <typename FloatType>
    std::pair<std::vector<Particle<FloatType>>, ParticleSimulationTimings> ImplOpenCL<FloatType>::simulate(const std::vector<Particle<FloatType>> &particles) {
        init(particles);
        for (int i = 0; i < _config.numberTimeSteps; ++i) {
            updatePositionsAndResetForce();
            computeForces();
            updateVelocities();
        }
        return std::make_pair(_particles->toParticles(queue), _timings);
    }

    template <typename FloatType>
    void ImplOpenCL<FloatType>::updatePositionsAndResetForce() {
        const size_t localSize = ParticleSimulationConfig<FloatType>::TILE_SIZE;
        const size_t globalSize = util::roundUp<size_t>(_numParticles, localSize);

        cl_event event;
        cl_ulong start, end;
        int err = clEnqueueNDRangeKernel(queue, kernelPositionUpdate, 1, nullptr, &globalSize, &localSize, 0, nullptr, &event);
        if (err != CL_SUCCESS) throw std::runtime_error("Position Update Kernel failed");

        // The profiling counters are only valid once the kernel has completed.
        clWaitForEvents(1, &event);
        clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(start), &start, nullptr);
        clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(end), &end, nullptr);
        const double elapsed_nanoseconds = end - start;
        _timings.positionUpdateForceResetTime += elapsed_nanoseconds;
        clReleaseEvent(event);
    }

    template <typename FloatType>
    void ImplOpenCL<FloatType>::updateVelocities() {
        const size_t localSize = ParticleSimulationConfig<FloatType>::TILE_SIZE;
        const size_t globalSize = util::roundUp<size_t>(_numParticles, localSize);

        cl_event event;
        cl_ulong start, end;
        int err = clEnqueueNDRangeKernel(queue, kerneVelocityUpdate, 1, nullptr, &globalSize, &localSize, 0, nullptr, &event);
        if (err != CL_SUCCESS) throw std::runtime_error("Velocity Kernel failed");

        // The profiling counters are only valid once the kernel has completed.
        clWaitForEvents(1, &event);
        clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(start), &start, nullptr);
        clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(end), &end, nullptr);
        const double elapsed_nanoseconds = end - start;
        _timings.velocityUpdateTime += elapsed_nanoseconds;
        clReleaseEvent(event);
    }

    template <typename FloatType>
    void ImplOpenCL<FloatType>::computeForces() {
        const size_t localSize = ParticleSimulationConfig<FloatType>::TILE_SIZE;
        const size_t globalSize = util::roundUp<size_t>(_numParticles, localSize);

        cl_event event;
        cl_ulong start, end;
        int err = clEnqueueNDRangeKernel(queue, kernelForceUpdate, 1, nullptr, &globalSize, &localSize, 0, nullptr, &event);
        if (err != CL_SUCCESS) throw std::runtime_error("Force Kernel failed");

        // The profiling counters are only valid once the kernel has completed.
        clWaitForEvents(1, &event);
        clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(start), &start, nullptr);
        clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(end), &end, nullptr);
        const double elapsed_nanoseconds = end - start;
        _timings.forceUpdateTime += elapsed_nanoseconds;
        clReleaseEvent(event);
    }

    template class ImplOpenCL<float>;
    template class OpenCLParticleSoA<float>;
    template class ImplOpenCL<double>;
    template class OpenCLParticleSoA<double>;

} // namespace ppb


