#include "gtest/gtest.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include "polyhedralGravity/PolyhedralGravityDefinitions.h"

namespace {

    /** A computation point together with the analytic potential and acceleration of the cube there */
    struct CubeReference {
        Array3 point;
        double potential;
        std::array<double, 3> acceleration;
    };

    /** The analytic solution of the cube for the density stated in the first line of its file */
    struct CubeSolution {
        double density = 0.0;
        std::vector<CubeReference> references;
    };

    CubeSolution readCubeSolution(const std::string &filename) {
        CubeSolution solution{};
        std::ifstream infile(filename);
        std::string line;

        // The first line contains only one value: The density
        if (std::getline(infile, line)) {
            std::istringstream(line) >> solution.density;
        }

        // The other lines contain tuples of point, potential and acceleration
        while (std::getline(infile, line)) {
            std::istringstream linestream(line);
            FloatType p1, p2, p3;
            double potential, acc1, acc2, acc3;
            if (!(linestream >> p1 >> p2 >> p3 >> potential >> acc1 >> acc2 >> acc3)) {
                break;
            }
            solution.references.push_back({Array3{p1, p2, p3}, potential, {acc1, acc2, acc3}});
        }
        return solution;
    }

    /** |actual - expected| relative to |expected|, or to the scale where the expected value is close to zero */
    template<typename Actual, std::size_t N>
    double relativeDeviation(const Actual &actual, const std::array<double, N> &expected, double scale) {
        double difference = 0.0;
        double magnitude = 0.0;
        for (std::size_t i = 0; i < N; ++i) {
            difference += (actual[i] - expected[i]) * (actual[i] - expected[i]);
            magnitude += expected[i] * expected[i];
        }
        return std::sqrt(difference) / std::max(std::sqrt(magnitude), scale);
    }

    /** Whether a coordinate of the point lies in the plane of one of the cube's faces */
    bool inFacePlane(const Array3 &point) {
        return std::abs(point[0]) == 1 || std::abs(point[1]) == 1 || std::abs(point[2]) == 1;
    }

}// namespace

/**
 * Compares the gravity model of a cube with its analytic solution.
 *
 * Every test evaluates all of its points with a single evaluable, so a result which leaks from one evaluation into
 * the next fails just as a wrong kernel does.
 */
class GravityModelTest : public ::testing::TestWithParam<std::string> {

protected:
#if FLOAT_BITS == 32
    /** How far potential and acceleration may deviate from the analytic solution, relative to it */
    static constexpr double LOCAL_TEST_EPSILON = 1e-4;
    /** The step of the central differences the tensor is checked with */
    static constexpr FloatType DIFFERENCE_STEP = 1e-2;
    /** How far the tensor may deviate from those differences, relative to them */
    static constexpr double TENSOR_TEST_EPSILON = 1e-2;
#else
    static constexpr double LOCAL_TEST_EPSILON = 1e-10;
    static constexpr FloatType DIFFERENCE_STEP = 1e-4;
    static constexpr double TENSOR_TEST_EPSILON = 1e-5;
#endif

    /** Only every n-th point is checked for the tensor, since each one takes seven evaluations */
    static constexpr std::size_t TENSOR_POINT_STRIDE = 5;

    /** The deviations reported individually before a test only counts them */
    static constexpr std::size_t REPORTED_DEVIATIONS = 10;

    std::vector<Array3> _vertices{{-1.0, -1.0, -1.0}, {1.0, -1.0, -1.0}, {1.0, 1.0, -1.0}, {-1.0, 1.0, -1.0},
                                  {-1.0, -1.0, 1.0}, {1.0, -1.0, 1.0}, {1.0, 1.0, 1.0}, {-1.0, 1.0, 1.0}};

    std::vector<IndexArray3> _faces{{1, 3, 2}, {0, 3, 1}, {0, 1, 5}, {0, 5, 4}, {0, 7, 3}, {0, 4, 7},
                                    {1, 2, 6}, {1, 6, 5}, {2, 3, 6}, {3, 7, 6}, {4, 5, 6}, {4, 6, 7}};

    CubeSolution _solution;

    GravityEvaluablePtr _evaluable;

    void SetUp() override {
        _solution = readCubeSolution(GetParam());
        ASSERT_FALSE(_solution.references.empty()) << "No reference points in " << GetParam();
        _evaluable = create_gravity_evaluable(_vertices, _faces, _solution.density);
    }

    /** The magnitude of acceleration and tensor of the cube, i.e. G * rho times powers of its unit side length */
    [[nodiscard]] double fieldScale() const {
        return GRAVITATIONAL_CONSTANT * _solution.density;
    }
};

TEST_P(GravityModelTest, PotentialAndAcceleration) {
    std::size_t deviations = 0;
    double maximumPotentialDeviation = 0.0;
    double maximumAccelerationDeviation = 0.0;
    for (const CubeReference &reference: _solution.references) {
        const auto [potential, acceleration, tensor] = _evaluable->evaluate(reference.point);

        const double potentialDeviation =
                relativeDeviation(std::array<double, 1>{potential}, std::array<double, 1>{reference.potential}, 0.0);
        const double accelerationDeviation = relativeDeviation(acceleration, reference.acceleration, fieldScale());
        maximumPotentialDeviation = std::max(maximumPotentialDeviation, potentialDeviation);
        maximumAccelerationDeviation = std::max(maximumAccelerationDeviation, accelerationDeviation);
        if (potentialDeviation <= LOCAL_TEST_EPSILON && accelerationDeviation <= LOCAL_TEST_EPSILON) {
            continue;
        }
        if (++deviations <= REPORTED_DEVIATIONS) {
            ADD_FAILURE() << "At (" << reference.point[0] << ", " << reference.point[1] << ", " << reference.point[2]
                          << "): potential " << potential << " instead of " << reference.potential
                          << " (relative deviation " << potentialDeviation << "), acceleration ("
                          << acceleration[0] << ", " << acceleration[1] << ", " << acceleration[2] << ") instead of ("
                          << reference.acceleration[0] << ", " << reference.acceleration[1] << ", "
                          << reference.acceleration[2] << ") (relative deviation " << accelerationDeviation << ")";
        }
    }
    // The largest deviations show how much headroom the tolerance leaves, e.g. with --gtest_output=xml
    RecordProperty("MaximumPotentialDeviation", std::to_string(maximumPotentialDeviation));
    RecordProperty("MaximumAccelerationDeviation", std::to_string(maximumAccelerationDeviation));
    EXPECT_EQ(deviations, 0u) << "of " << _solution.references.size() << " points deviate";
}

TEST_P(GravityModelTest, GradiometricTensor) {
    // The tensor is the derivative of the acceleration, so it is checked against central differences of it. Points in
    // the plane of a face are skipped: the acceleration is not differentiable across the cube's surface, and on the
    // lines of the edges the tensor is not defined.
    std::size_t deviations = 0;
    std::size_t checked = 0;
    double maximumDeviation = 0.0;
    for (std::size_t index = 0; index < _solution.references.size(); index += TENSOR_POINT_STRIDE) {
        const Array3 &point = _solution.references[index].point;
        if (inFacePlane(point)) {
            continue;
        }
        ++checked;

        const auto [potential, acceleration, tensor] = _evaluable->evaluate(point);

        // derivatives[j][i] is the derivative of the i-th acceleration component along the j-th axis
        std::array<std::array<double, 3>, 3> derivatives{};
        for (std::size_t axis = 0; axis < 3; ++axis) {
            Array3 forward = point;
            Array3 backward = point;
            forward[axis] += DIFFERENCE_STEP;
            backward[axis] -= DIFFERENCE_STEP;
            const Array3 forwardAcceleration = _evaluable->evaluate(forward).acceleration;
            const Array3 backwardAcceleration = _evaluable->evaluate(backward).acceleration;
            const double step = static_cast<double>(forward[axis]) - static_cast<double>(backward[axis]);
            for (std::size_t component = 0; component < 3; ++component) {
                derivatives[axis][component] =
                        (static_cast<double>(forwardAcceleration[component]) - backwardAcceleration[component]) / step;
            }
        }

        // The tensor is ordered xx, yy, zz, xy, xz, yz; the mixed derivatives are symmetric and taken in both orders
        const std::array<double, 6> expected{
                derivatives[0][0],
                derivatives[1][1],
                derivatives[2][2],
                (derivatives[1][0] + derivatives[0][1]) / 2,
                (derivatives[2][0] + derivatives[0][2]) / 2,
                (derivatives[2][1] + derivatives[1][2]) / 2};
        const double deviation = relativeDeviation(tensor, expected, fieldScale());
        maximumDeviation = std::max(maximumDeviation, deviation);
        if (deviation <= TENSOR_TEST_EPSILON) {
            continue;
        }
        if (++deviations <= REPORTED_DEVIATIONS) {
            ADD_FAILURE() << "At (" << point[0] << ", " << point[1] << ", " << point[2] << "): tensor (" << tensor[0]
                          << ", " << tensor[1] << ", " << tensor[2] << ", " << tensor[3] << ", " << tensor[4] << ", "
                          << tensor[5] << ") instead of (" << expected[0] << ", " << expected[1] << ", "
                          << expected[2] << ", " << expected[3] << ", " << expected[4] << ", " << expected[5]
                          << ") (relative deviation " << deviation << ")";
        }
    }
    ASSERT_GT(checked, 0u);
    RecordProperty("MaximumTensorDeviation", std::to_string(maximumDeviation));
    EXPECT_EQ(deviations, 0u) << "of " << checked << " points deviate";
}

INSTANTIATE_TEST_SUITE_P(
        CubeGravityModelTest, GravityModelTest,
        ::testing::Values("resources/polyhedral_analytic_cube_solution_density1.txt",
                          "resources/polyhedral_analytic_cube_solution_density42.txt"),
        [](const ::testing::TestParamInfo<std::string> &info) {
            // resources/polyhedral_analytic_cube_solution_density<N>.txt --> Density<N>
            const std::size_t begin = info.param.rfind("density") + std::string("density").size();
            return "Density" + info.param.substr(begin, info.param.rfind('.') - begin);
        });
