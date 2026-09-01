#pragma once

#include <cmath>
#include <random>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "GoogleTestMatcher.h"
#include "nBodySimulation/NBodySimulation.h"
#include "nBodySimulation/cpp/Impl_Cpp.h"

class NBodyTest : public ::testing::TestWithParam<int> {
protected:

    static constexpr double EPSILON = 1e-2;

    static constexpr double TIME_STEP = 0.0005;
    static constexpr int ITERATIONS = 1000;

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
        ppb::Particle<float>({10.1082f, 18.4454f, -6.88032f}, {29.3611f, 40.7827f, -7.60054f},
                                 {-1.61701e-08f, -2.82026e-08f, 5.06539e-09f}),
        ppb::Particle<float>({-0.136977f, -2.75841f, -0.720557f}, {-0.276126f, -0.502396f, -0.101534f},
                                 {-1.00656f, -2.07012f, -0.314161f}),
        ppb::Particle<float>({-1.93135f, -4.89487f, -1.45594f}, {-0.153927f, -0.19403f, -0.0295663f},
                                 {0.589444f, 0.386863f, 0.303662f}),
        ppb::Particle<float>({-0.71208f, -3.9097f, -0.912773f}, {0.71042f, 0.216672f, 0.600289f},
                                 {1.11028f, -0.108743f, 1.45302f}),
        ppb::Particle<float>({-4.74386f, -0.145481f, -3.82485f}, {0.049964f, -0.0305189f, 0.02346f},
                                 {0.00427584f, -0.0027542f, -0.00185222f}),
        ppb::Particle<float>({-1.18162f, -2.89843f, -4.29062f}, {1.64198f, -0.239156f, 0.407105f},
                                 {-0.0399926f, 0.0504114f, -0.0285169f}),
        ppb::Particle<float>({-0.374242f, -4.89932f, -0.143446f}, {-0.283253f, 0.482698f, -0.477017f},
                                 {-0.699009f, 1.79725f, -1.45887f}),
        ppb::Particle<float>({-3.63295f, -4.08344f, -3.99767f}, {-1.85655f, -1.51388f, 0.583714f},
                                 {0.0242466f, 0.00990631f, 0.00421503f}),
        ppb::Particle<float>({-19.561f, -22.7167f, 0.779729f}, {-29.4066f, -40.756f, 7.57473f},
                                 {1.36785e-08f, 1.4923e-08f, -2.49879e-09f}),
        ppb::Particle<float>({-1.81841f, -1.06152f, -5.45934f}, {0.212958f, 1.75392f, -0.98065f},
                                 {0.0173082f, -0.0628128f, 0.042503f})};


    template <typename Implementation>
    void runTest(const int size, const double epsilon = EPSILON) {
        using namespace testing;
        using namespace ppb;

        if (size == 10) {
            ParticleSimulationConfig<float> config{static_cast<size_t>(size), ITERATIONS, TIME_STEP};
            Implementation nBodySim{config};
            const auto [actualResult, timings] = nBodySim.simulate(start_state);
            ASSERT_THAT(actualResult, ParticlesEq(end_state, epsilon));
            return;
        }

        ParticleSimulationConfig<float> config{static_cast<size_t>(size), ITERATIONS, 1e-10};
        NBodySimulation<ImplCpp<float>> cppNBodySim{config};
        NBodySimulation<Implementation> otherNBodySim{config};
        const auto [expectedResult, timings1] = cppNBodySim();
        const auto [actualResult, timings2]  = otherNBodySim();

        ASSERT_THAT(actualResult, ParticlesEq(expectedResult, epsilon));
    }
};

/**
 * Fixture for the physics-invariant test case of the n-body scenario.
 *
 * In contrast to NBodyTest, which pins the trajectory against a reference produced by AutoPas, this fixture does not
 * compare against any pre-computed state. It instead checks a property that every correct Lennard-Jones implementation
 * has to fulfil, no matter how it distributes or orders its work:
 *
 * 1. Conservation of the total energy E = E_kin + E_pot.
 *    The simulated system is isolated (no boundaries, no thermostat, no global force), i.e. it is a Hamiltonian
 *    system, and Velocity-Stoermer-Verlet is a symplectic integrator. Hence the total energy must not drift; it may
 *    only oscillate with an amplitude of O(deltaT^2). This directly tests that the implemented force is really the
 *    negative gradient of the Lennard-Jones potential and that the integrator consumes it consistently.
 *
 * 2. Conservation of the total linear momentum P = sum_i m_i * v_i.
 *    Follows from Newton's third law. This is essentially free to check and catches an asymmetric accumulation of
 *    the pairwise forces (e.g. a broken Newton-3 optimisation or a race condition in a parallel reduction), which
 *    the energy criterion alone can mask.
 *
 * The initial state is a simple-cubic lattice with a spacing of 2^(1/6) * sigma (the minimum of the Lennard-Jones
 * potential) plus small random velocities with zero net momentum. A lattice is used instead of NBodySimulation's
 * uniform random placement, because uniformly drawn positions inevitably produce overlapping pairs whose
 * r^-12 repulsion blows up any explicit integrator, which would make an energy criterion meaningless.
 * The cluster is not in equilibrium (its surface particles get pulled inwards), so the run has real dynamics:
 * the kinetic energy grows by roughly an order of magnitude while the total energy stays put.
 */
class NBodyEnergyTest : public ::testing::Test {
protected:

    /**
     * Lattice dimensions. Their product is the particle count of the scenario (5 * 5 * 4 = 100).
     */
    static constexpr int LATTICE_SIZE_X = 5;
    static constexpr int LATTICE_SIZE_Y = 5;
    static constexpr int LATTICE_SIZE_Z = 4;

    /**
     * Lattice spacing 2^(1/6) * sigma, i.e. the minimum of the Lennard-Jones potential for sigma = 1.
     * Close enough that every particle interacts strongly with its neighbours, far enough that nothing overlaps.
     */
    static constexpr double LATTICE_SPACING = 1.1224620483093730;

    /**
     * Bound of the uniformly drawn initial velocity components. Chosen so that the particles travel a noticeable
     * fraction of sigma during the run without producing violent close encounters.
     */
    static constexpr double MAX_INITIAL_VELOCITY = 0.25;

    /**
     * Seed of the initial velocity distribution. Fixed, so the scenario is bit-wise reproducible.
     */
    static constexpr unsigned int VELOCITY_SEED = 42u;

    /**
     * Integration parameters of the scenario. 1000 steps of 0.0005 cover a simulated time of 0.5 in Lennard-Jones
     * units, long enough for the lattice to visibly relax and short enough to keep the run cheap.
     */
    static constexpr double ENERGY_TIME_STEP = 0.0005;
    static constexpr int ENERGY_ITERATIONS = 1000;

    /**
     * Simulation domain. Only relevant for the cell-list based implementations, which derive their cell grid from it.
     * It is generously larger than the lattice, so no particle can leave it during the run, and wide enough that
     * the cell-list implementations get at least three cells per dimension (box width >= 3 * ParticleSimulationConfig::h),
     * which their 3x3x3 neighbour stencil requires. The lattice sits in the centre of the domain.
     */
    static constexpr double BOX_HALF_WIDTH = 15.0;

    /**
     * Allowed relative drift |E(T) - E(0)| / |E(0)| of the total energy.
     * The reference C++ implementation stays below 2e-6 in single precision, so this leaves roughly two orders
     * of magnitude of head room for implementations that reorder their floating point operations.
     */
    static constexpr double ENERGY_TOLERANCE = 1e-4;

    /**
     * Allowed drift of the total linear momentum, relative to the sum of the absolute particle momenta.
     */
    static constexpr double MOMENTUM_TOLERANCE = 1e-4;

    /**
     * Builds the initial state: a simple-cubic lattice centred around the origin, carrying uniformly drawn
     * velocities from which the mean is subtracted, so the total momentum starts at zero.
     *
     * @return the initial particles of the energy conservation scenario
     */
    static std::vector<ppb::Particle<float>> generateLatticeState() {
        constexpr size_t numberOfParticles = LATTICE_SIZE_X * LATTICE_SIZE_Y * LATTICE_SIZE_Z;

        std::mt19937 generator{VELOCITY_SEED};
        std::uniform_real_distribution<double> distribution{-MAX_INITIAL_VELOCITY, MAX_INITIAL_VELOCITY};
        std::vector<std::array<double, 3>> velocities(numberOfParticles);
        std::array<double, 3> meanVelocity{0.0, 0.0, 0.0};
        for (auto &velocity : velocities) {
            velocity = {distribution(generator), distribution(generator), distribution(generator)};
            for (int dim = 0; dim < 3; ++dim) {
                meanVelocity[dim] += velocity[dim] / static_cast<double>(numberOfParticles);
            }
        }

        std::vector<ppb::Particle<float>> particles;
        particles.reserve(numberOfParticles);
        size_t idx = 0;
        for (int x = 0; x < LATTICE_SIZE_X; ++x) {
            for (int y = 0; y < LATTICE_SIZE_Y; ++y) {
                for (int z = 0; z < LATTICE_SIZE_Z; ++z) {
                    const std::array<float, 3> position{
                        static_cast<float>((x - (LATTICE_SIZE_X - 1) * 0.5) * LATTICE_SPACING),
                        static_cast<float>((y - (LATTICE_SIZE_Y - 1) * 0.5) * LATTICE_SPACING),
                        static_cast<float>((z - (LATTICE_SIZE_Z - 1) * 0.5) * LATTICE_SPACING)};
                    const std::array<float, 3> velocity{
                        static_cast<float>(velocities[idx][0] - meanVelocity[0]),
                        static_cast<float>(velocities[idx][1] - meanVelocity[1]),
                        static_cast<float>(velocities[idx][2] - meanVelocity[2])};
                    particles.emplace_back(position, velocity);
                    ++idx;
                }
            }
        }
        return particles;
    }

    /**
     * @return the total kinetic energy sum_i 0.5 * m_i * |v_i|^2, accumulated in double precision
     */
    static double kineticEnergy(const std::vector<ppb::Particle<float>> &particles) {
        double energy = 0.0;
        for (const auto &particle : particles) {
            const auto velocity = particle.getVelocity();
            const double velocitySquared = static_cast<double>(velocity[0]) * velocity[0] +
                                           static_cast<double>(velocity[1]) * velocity[1] +
                                           static_cast<double>(velocity[2]) * velocity[2];
            energy += 0.5 * static_cast<double>(particle.getMass()) * velocitySquared;
        }
        return energy;
    }

    /**
     * @return the total Lennard-Jones potential energy sum_{i<j} 4 * eps * ((sigma/r)^12 - (sigma/r)^6),
     *         accumulated in double precision. The full (untruncated) potential is evaluated.
     */
    static double potentialEnergy(const std::vector<ppb::Particle<float>> &particles) {
        double energy = 0.0;
        for (size_t i = 0; i < particles.size(); ++i) {
            for (size_t j = 0; j < i; ++j) {
                const auto positionI = particles[i].getPosition();
                const auto positionJ = particles[j].getPosition();
                const double dx = static_cast<double>(positionI[0]) - positionJ[0];
                const double dy = static_cast<double>(positionI[1]) - positionJ[1];
                const double dz = static_cast<double>(positionI[2]) - positionJ[2];

                const double sigma = 0.5 * (static_cast<double>(particles[i].getSigma()) + particles[j].getSigma());
                const double epsilon = std::sqrt(static_cast<double>(particles[i].getEpsilon()) * particles[j].getEpsilon());

                const double invR2 = (sigma * sigma) / (dx * dx + dy * dy + dz * dz);
                const double lj6 = invR2 * invR2 * invR2;
                energy += 4.0 * epsilon * (lj6 * lj6 - lj6);
            }
        }
        return energy;
    }

    /**
     * @return the total linear momentum sum_i m_i * v_i, accumulated in double precision
     */
    static std::array<double, 3> totalMomentum(const std::vector<ppb::Particle<float>> &particles) {
        std::array<double, 3> momentum{0.0, 0.0, 0.0};
        for (const auto &particle : particles) {
            const auto velocity = particle.getVelocity();
            for (int dim = 0; dim < 3; ++dim) {
                momentum[dim] += static_cast<double>(particle.getMass()) * velocity[dim];
            }
        }
        return momentum;
    }

    /**
     * @return sum_i |m_i * v_i|, the scale against which the momentum drift is judged
     */
    static double totalAbsoluteMomentum(const std::vector<ppb::Particle<float>> &particles) {
        double scale = 0.0;
        for (const auto &particle : particles) {
            const auto velocity = particle.getVelocity();
            const double velocitySquared = static_cast<double>(velocity[0]) * velocity[0] +
                                           static_cast<double>(velocity[1]) * velocity[1] +
                                           static_cast<double>(velocity[2]) * velocity[2];
            scale += static_cast<double>(particle.getMass()) * std::sqrt(velocitySquared);
        }
        return scale;
    }

    /**
     * Runs the 100 particle lattice scenario with the given implementation and asserts that neither the total
     * energy nor the total linear momentum of the isolated system changed.
     *
     * @tparam Implementation the simulation implementation under test
     * @param energyTolerance allowed relative drift of the total energy
     * @param momentumTolerance allowed relative drift of the total linear momentum
     */
    template <typename Implementation>
    void runEnergyConservationTest(const double energyTolerance = ENERGY_TOLERANCE,
                                   const double momentumTolerance = MOMENTUM_TOLERANCE) {
        using namespace ppb;

        const auto initialParticles = generateLatticeState();

        ParticleSimulationConfig<float> config{initialParticles.size(), ENERGY_ITERATIONS,
                                               static_cast<float>(ENERGY_TIME_STEP)};
        config.boxMin = {static_cast<float>(-BOX_HALF_WIDTH), static_cast<float>(-BOX_HALF_WIDTH),
                         static_cast<float>(-BOX_HALF_WIDTH)};
        config.boxMax = {static_cast<float>(BOX_HALF_WIDTH), static_cast<float>(BOX_HALF_WIDTH),
                         static_cast<float>(BOX_HALF_WIDTH)};

        Implementation nBodySim{config};
        const auto [finalParticles, timings] = nBodySim.simulate(initialParticles);

        ASSERT_EQ(finalParticles.size(), initialParticles.size());

        const double initialEnergy = kineticEnergy(initialParticles) + potentialEnergy(initialParticles);
        const double finalEnergy = kineticEnergy(finalParticles) + potentialEnergy(finalParticles);
        const double energyDrift = std::abs(finalEnergy - initialEnergy) / std::abs(initialEnergy);

        EXPECT_LE(energyDrift, energyTolerance)
            << "The total energy of the isolated system is not conserved.\n"
            << "  E(0) = " << initialEnergy << " (kin " << kineticEnergy(initialParticles)
            << ", pot " << potentialEnergy(initialParticles) << ")\n"
            << "  E(T) = " << finalEnergy << " (kin " << kineticEnergy(finalParticles)
            << ", pot " << potentialEnergy(finalParticles) << ")\n"
            << "  relative drift = " << energyDrift;

        const auto initialMomentum = totalMomentum(initialParticles);
        const auto finalMomentum = totalMomentum(finalParticles);
        const double momentumScale = totalAbsoluteMomentum(initialParticles);
        double momentumDrift = 0.0;
        for (int dim = 0; dim < 3; ++dim) {
            momentumDrift = std::max(momentumDrift, std::abs(finalMomentum[dim] - initialMomentum[dim]));
        }
        momentumDrift /= momentumScale;

        EXPECT_LE(momentumDrift, momentumTolerance)
            << "The total linear momentum of the isolated system is not conserved.\n"
            << "  P(0) = [" << initialMomentum[0] << ", " << initialMomentum[1] << ", " << initialMomentum[2] << "]\n"
            << "  P(T) = [" << finalMomentum[0] << ", " << finalMomentum[1] << ", " << finalMomentum[2] << "]\n"
            << "  relative drift = " << momentumDrift;
    }
};
