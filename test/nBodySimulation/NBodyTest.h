#pragma once

#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "GoogleTestMatcher.h"
#include "nBodySimulation/NBodySimulation.h"
#include "nBodySimulation/cpp/Impl_Cpp.h"

class NBodyTest : public ::testing::TestWithParam<int> {
protected:

    static constexpr double EPSILON = 0.05;

    static constexpr float TIME_STEP = 0.0005;
    static constexpr int ITERATIONS = 5;

    /*
    AutoPas Config File to replicate the values below.
    With the huge simulation box and cut-off radius, we effectively disable these features.
        container: [ DirectSum ]
        selector-strategy: Fastest-Absolute-Value
        data-layout: [ AoS, SoA ]
        traversal: [ ds_sequential ]
        functor: Lennard-Jones
        newton3: [ disabled, enabled ]
        cutoff: 10000
        box-min: [ -1000, -1000, -1000 ]
        box-max: [ 1000, 1000, 1000 ]
        deltaT: 0.0005
        iterations: 1000
        boundary-type: [ none,none,none ]
        globalForce: [ 0 ,0 ,0 ]
        Sites:
          0:
            epsilon: 1.
            sigma: 1.
            mass: 1.
        Objects:
          CubeUniform:
            0:
              numberOfParticles: 10
              box-length: [ 5, 5, 5 ]
              bottomLeftCorner: [ -5, -5, -5 ]
              velocity: [ 0, 0, 0 ]
              particle-type-id: 0
        vtk-filename: fallingDrop
        vtk-write-frequency: 1000
        vtk-output-folder: vtkOutputFolder
        no-end-config: true
        log-level: info
    */

    static inline std::vector<ppb::Particle<float>> start_state = {
        ppb::Particle<float>({-4.54697f, -1.90807f, -3.08769f}, {0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f}),
        ppb::Particle<float>({-0.0838456f, -2.66619f, -0.700298f}, {0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f}),
        ppb::Particle<float>({-1.74556f, -4.71794f, -1.39001f}, {0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f}),
        ppb::Particle<float>({-1.01729f, -4.08283f, -1.10155f}, {0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f}),
        ppb::Particle<float>({-4.76667f, -0.131222f, -3.83614f}, {0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f}),
        ppb::Particle<float>({-2.01575f, -2.77084f, -4.50013f}, {0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f}),
        ppb::Particle<float>({-0.307236f, -4.99611f, -0.0389422f}, {0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f}),
        ppb::Particle<float>({-2.70376f, -3.33146f, -4.28567f}, {0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f}),
        ppb::Particle<float>({-4.88469f, -2.37613f, -3.0007f}, {0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f}),
        ppb::Particle<float>({-1.91259f, -1.94173f, -4.96467f}, {0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f})};
    static inline std::vector<ppb::Particle<float>> end_state = {
        ppb::Particle<float>({10.1082f, 18.4454f, -6.88032f}, {29.3611f, 40.7827f, -7.60054f}, {-1.61701e-08f, -2.82026e-08f, 5.06539e-09f}),
        ppb::Particle<float>({-0.136977f, -2.75841f, -0.720557f}, {-0.276126f, -0.502396f, -0.101534f}, {-1.00656f, -2.07012f, -0.314161f}),
        ppb::Particle<float>({-1.93135f, -4.89487f, -1.45594f}, {-0.153927f, -0.19403f, -0.0295663f}, {0.589444f, 0.386863f, 0.303662f}),
        ppb::Particle<float>({-0.71208f, -3.9097f, -0.912773f}, {0.71042f, 0.216672f, 0.600289f}, {1.11028f, -0.108743f, 1.45302f}),
        ppb::Particle<float>({-4.74386f, -0.145481f, -3.82485f}, {0.049964f, -0.0305189f, 0.02346f}, {0.00427584f, -0.0027542f, -0.00185222f}),
        ppb::Particle<float>({-1.18162f, -2.89843f, -4.29062f}, {1.64198f, -0.239156f, 0.407105f}, {-0.0399926f, 0.0504114f, -0.0285169f}),
        ppb::Particle<float>({-0.374242f, -4.89932f, -0.143446f}, {-0.283253f, 0.482698f, -0.477017f}, {-0.699009f, 1.79725f, -1.45887f}),
        ppb::Particle<float>({-3.63295f, -4.08344f, -3.99767f}, {-1.85655f, -1.51388f, 0.583714f}, {0.0242466f, 0.00990631f, 0.00421503f}),
        ppb::Particle<float>({-19.561f, -22.7167f, 0.779729f}, {-29.4066f, -40.756f, 7.57473f}, {1.36785e-08f, 1.4923e-08f, -2.49879e-09f}),
        ppb::Particle<float>({-1.81841f, -1.06152f, -5.45934f}, {0.212958f, 1.75392f, -0.98065f}, {0.0173082f, -0.0628128f, 0.042503f})};

    float get_total_energy(std::vector<ppb::Particle<float>> system) {
        //assuming all particles have the same mass
        float result = 0.f;
        for (auto& p : system) {
            result += p.getVelocity()[0] * p.getVelocity()[0] 
                    + p.getVelocity()[1] * p.getVelocity()[1]
                    + p.getVelocity()[2] * p.getVelocity()[2];
        }
        return result / 2;
    }

    template <typename Implementation>
    void runTest(const int size, const double epsilon = EPSILON) {
        using namespace testing;
        using namespace ppb;

        if (size == 10) {
            //Positions
            ParticleSimulationConfig<float> config{static_cast<size_t>(size), ITERATIONS, TIME_STEP};
            Implementation nBodySim{config};
            const auto [actualResult, timings] = nBodySim.simulate(start_state);
            ASSERT_THAT(actualResult, ParticlesEq(end_state, epsilon));
            
            //Energy conservation
            auto state = start_state;
            for (size_t i = 0; i < ITERATIONS; i++) {
                printf("ITERATION %lu\n", i);
                float total_energy_before = get_total_energy(state);
                ParticleSimulationConfig<float> config{static_cast<size_t>(size), 1, TIME_STEP};
                Implementation nBodySim{config};
                const auto [new_state, timings] = nBodySim.simulate(state);
                state = new_state;
                float total_energy_after = get_total_energy(state);
                printf("BEFORE: %f, AFTER: %f\n", total_energy_before, total_energy_after);
            }

            return;
        }

            //Positions
            ParticleSimulationConfig<float> config{static_cast<size_t>(size), ITERATIONS, 0.0005};
            NBodySimulation<ImplCpp<float>> cppNBodySim{config};
            NBodySimulation<Implementation> otherNBodySim{config};
            const auto [expectedResult, timings1] = cppNBodySim();
            const auto [actualResult, timings2] = otherNBodySim();
/*             std::cout<<"Expected:"<<std::endl;
            for (auto& p : expectedResult) {
                std::cout<<p<<std::endl;
            }
            std::cout<<"Actual:"<<std::endl;
            for (auto& p : actualResult) {
                std::cout<<p<<std::endl;
            } */
            ASSERT_THAT(actualResult, ParticlesEq(expectedResult, epsilon));
            printf("MEOW!\n");
/*             //Energy conservation
            ParticleSimulationConfig<float> config2{static_cast<size_t>(size), 1, TIME_STEP};
            auto state = Particle<float>::generateUniform(config2.boxMin, config2.boxMax, config2.size, config2.seed)};
            for (size_t i = 0; i < ITERATIONS; i++) {
                float total_energy_before = get_total_energy(state);
                config2{static_cast<size_t>(size), 1, TIME_STEP};
                Implementation nBodySim{config};
                const auto [new_state, timings] = nBodySim.simulate(state);
                state = new_state;
                float total_energy_after = get_total_energy(state);
                printf("BEFORE: %f, AFTER: %f\n", total_energy_before, total_energy_after);
            } */
        }
    };
