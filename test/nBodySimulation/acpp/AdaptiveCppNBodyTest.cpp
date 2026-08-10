#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "NBodyTest.h"
#include "nBodySimulation/acpp/Impl_AdaptiveCpp.h"

TEST_P(NBodyTest, ImplAdaptiveCpp_Implementation) {
    const int size = GetParam();
    this->runTest<ppb::ImplAdaptiveCpp<float>>(size);
}

INSTANTIATE_TEST_SUITE_P(BySize, NBodyTest, ::testing::Values(10, 100));