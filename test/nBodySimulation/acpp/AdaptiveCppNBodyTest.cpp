#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "NBodyTest.h"
#include "nBodySimulation/acpp/Impl_AdaptiveCpp.h"

static constexpr double_t cutoff = std::numeric_limits<double_t>::infinity();

/**
 * These tests are for functionality, not correctness. No interactions are skipped for any algorithm
 */

TEST_P(NBodyTest, ImplAcpp_UnsortedNaive_Implementation) {
    const int size = GetParam();
    this->runTest<ppb::ImplAdaptiveCpp<float, ppb::Naive<ppb::SorterKinds::None, cutoff>>, cutoff>(size);
}

TEST_P(NBodyTest, ImplAcpp_SortedNaive_Implementation) {
    const int size = GetParam();
    this->runTest<ppb::ImplAdaptiveCpp<float, ppb::Naive<ppb::SorterKinds::MergeProjection, cutoff>>, cutoff>(size);
}

TEST_P(NBodyTest, ImplAcpp_UnsortedCellList_Implementation) {
    const int size = GetParam();
    this->runTest<ppb::ImplAdaptiveCpp<float, ppb::CellList<ppb::SorterKinds::None, cutoff>>, cutoff>(size);
}

TEST_P(NBodyTest, ImplAcpp_SortedCellList_Implementation) {
    const int size = GetParam();
    this->runTest<ppb::ImplAdaptiveCpp<float, ppb::CellList<ppb::SorterKinds::MergeCellID, cutoff>>, cutoff>(size);
}

INSTANTIATE_TEST_SUITE_P(BySize, NBodyTest, ::testing::Values(10, 100, 1000));