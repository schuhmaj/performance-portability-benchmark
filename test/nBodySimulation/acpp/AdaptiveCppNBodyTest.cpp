#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "NBodyTest.h"
#include "nBodySimulation/acpp/Impl_AdaptiveCpp.h"

TEST_P(NBodyTest, ImplAcpp_UnsortedNaive_Implementation) {
    const int size = GetParam();
    this->runTest<ppb::ImplAdaptiveCpp<float, ppb::Naive<ppb::SorterKinds::None>>>(size);
}

TEST_P(NBodyTest, ImplAcpp_SortedNaive_Implementation) {
    const int size = GetParam();
    this->runTest<ppb::ImplAdaptiveCpp<float, ppb::Naive<ppb::SorterKinds::MergeProjection>>>(size);
}

TEST_P(NBodyTest, ImplAcpp_UnsortedCellList_Implementation) {
    const int size = GetParam();
    this->runTest<ppb::ImplAdaptiveCpp<float, ppb::CellList<ppb::SorterKinds::None>>>(size);
}

TEST_P(NBodyTest, ImplAcpp_SortedCellList_Implementation) {
    const int size = GetParam();
    this->runTest<ppb::ImplAdaptiveCpp<float, ppb::CellList<ppb::SorterKinds::MergeCellID>>>(size);
}

INSTANTIATE_TEST_SUITE_P(BySize, NBodyTest, ::testing::Values(10, 100, 1000, 10'000));