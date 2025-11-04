#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "NBodyTest.h"
#include "nBodySimulation/alpaka/Impl_Alpaka.h"

TEST_P(NBodyTest, ImplAlpaka_Implementation) {
    const int size = GetParam();
    this->runTest<ppb::ImplAlpaka<float>>(size);
}

INSTANTIATE_TEST_SUITE_P(BySize, NBodyTest, ::testing::Values(10, 100, 1000));