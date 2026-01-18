#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "NBodyTest.h"
#include "nBodySimulation/acpp/Impl_AdaptiveCpp.h"

static constexpr double_t cutoff = 3.0;
static constexpr size_t problem_size = 1000;

/**
 * These sets of tests are only useful for low ParticleSimulationConfig::boxMin and ParticleSimulationConfig::boxMax
 * Since ImplCpp is far slower than ImplAdaptiveCpp, testing is limited by the problem size which can be simulated by ImplCpp.
 * Because particles are distributed uniformly in the entire domain and we want to test the correctness algorithm with the intended cutoff radius,
 * we require adequate particle density. Where problem_size and the cutoff radius are fixed, particle density is based solely on domain size.
 * For large domains, accuracy is likely perfect, because there are no interactions inside the cutoff radius.
 * For domains too small, algorithms might still execute all interactions.
 * Example domain sizes are 20x20x20
 */

TEST_P(NBodyTest, ImplAcpp_UnsortedNaive_Precision) {
    const double epsilon = 1.0 / GetParam();
    this->runTest<ppb::ImplAdaptiveCpp<float, ppb::Naive<ppb::SorterKinds::None, cutoff>>, cutoff>(problem_size, epsilon);
}

TEST_P(NBodyTest, ImplAcpp_SortedNaive_Precision) {
    const double epsilon = 1.0 / GetParam();
    this->runTest<ppb::ImplAdaptiveCpp<float, ppb::Naive<ppb::SorterKinds::MergeProjection, cutoff>>, cutoff>(problem_size, epsilon);
}

TEST_P(NBodyTest, ImplAcpp_UnsortedCellList_Precision) {
    const double epsilon = 1.0 / GetParam();
    this->runTest<ppb::ImplAdaptiveCpp<float, ppb::CellList<ppb::SorterKinds::None, cutoff>>, cutoff>(problem_size, epsilon);
}

TEST_P(NBodyTest, ImplAcpp_SortedCellList_Precision) {
    const double epsilon = 1.0 / GetParam();
    this->runTest<ppb::ImplAdaptiveCpp<float, ppb::CellList<ppb::SorterKinds::MergeCellID, cutoff>>, cutoff>(problem_size, epsilon);
}

INSTANTIATE_TEST_SUITE_P(BySize, NBodyTest, ::testing::Values(1, 10, 100, 1000, 10000, 100000));