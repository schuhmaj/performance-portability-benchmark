#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include <array>
#include <fstream>
#include <sstream>
#include <string>
#include <tuple>
#include <utility>
#include <vector>
#include "polyhedralGravity/PolyhedralGravityDefinitions.h"


/**
 * Contains Tests how the calculation handles a cubic polyhedron
 */
class GravityModelCubeTest
    : public ::testing::TestWithParam<std::tuple<Array3, FloatType, FloatType, Array3>> {

protected:

#if FLOAT_BITS == 32
    static constexpr FloatType LOCAL_TEST_EPSILON = 1e-5;
#else
    static constexpr FloatType LOCAL_TEST_EPSILON = 1e-10;
#endif



    std::vector<Array3> _vertices{{-1.0, -1.0, -1.0}, {1.0, -1.0, -1.0}, {1.0, 1.0, -1.0}, {-1.0, 1.0, -1.0},
                                  {-1.0, -1.0, 1.0},  {1.0, -1.0, 1.0},  {1.0, 1.0, 1.0},  {-1.0, 1.0, 1.0}};

    std::vector<IndexArray3> _faces{{1, 3, 2}, {0, 3, 1}, {0, 1, 5}, {0, 5, 4}, {0, 7, 3}, {0, 4, 7},
                                    {1, 2, 6}, {1, 6, 5}, {2, 3, 6}, {3, 7, 6}, {4, 5, 6}, {4, 6, 7}};
    FloatType _density = 1.0;

public:
    [[nodiscard]] static std::vector<std::tuple<Array3, FloatType, FloatType, Array3>>
    readCubePoints(const std::string &filename) {
        std::vector<std::tuple<Array3, FloatType, FloatType, Array3>> result{};
        std::ifstream infile(filename);
        std::string line;

        // The first line contains only one value: The density
        std::getline(infile, line);
        std::istringstream firstLineStream(line);
        FloatType density{1.0};
        firstLineStream >> density;

        // The other lines contain tuples of point, potential and acceleration
        while (std::getline(infile, line)) {
            std::istringstream linestream(line);
            FloatType p1, p2, p3, potential, acc1, acc2, acc3;
            if (!(linestream >> p1 >> p2 >> p3 >> potential >> acc1 >> acc2 >> acc3)) {
                break;
            }
            result.emplace_back(Array3{p1, p2, p3}, density, potential,
                                Array3{acc1, acc2, acc3});
        }
        return result;
    }
};

TEST_P(GravityModelCubeTest, CubePoints) {
    using namespace testing;

    const std::tuple<Array3, FloatType, FloatType, Array3> testData = GetParam();
    const Array3 computationPoint = std::get<0>(testData);
    const FloatType density = std::get<1>(testData);

    const FloatType expectedPotential = std::get<2>(testData);
    const Array3 expectedAcceleration = std::get<3>(testData);

    const auto evaluable = create_gravity_evaluable(_vertices, _faces, density);

    const auto [actualPotential, actualAcceleration, x] = evaluable->evaluate(computationPoint);

    ASSERT_NEAR(actualPotential, expectedPotential, LOCAL_TEST_EPSILON);
}

INSTANTIATE_TEST_SUITE_P(
    CubeGravityModelTest01, GravityModelCubeTest,
    ::testing::ValuesIn(GravityModelCubeTest::readCubePoints("resources/polyhedral_analytic_cube_solution_density1.txt")));

INSTANTIATE_TEST_SUITE_P(
    CubeGravityModelTest42, GravityModelCubeTest,
    ::testing::ValuesIn(GravityModelCubeTest::readCubePoints("resources/polyhedral_analytic_cube_solution_density42.txt")));
