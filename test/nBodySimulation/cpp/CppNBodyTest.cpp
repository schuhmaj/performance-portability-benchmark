#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "NBodyTest.h"
#include "nBodySimulation/cpp/Impl_Cpp.h"

TEST_P(NBodyTest, ImplCpp_Implementation) {
    const int size = GetParam();
    this->runTest<ppb::ImplCpp<float>>(size);
}

INSTANTIATE_TEST_SUITE_P(BySize, NBodyTest, ::testing::Values(10));