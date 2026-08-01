#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "NBodyTest.h"
#include "nBodySimulation/slang/naive/Impl_Slang_Cuda.cuh"

TEST_P(NBodyTest, ImplSlangCuda_Implementation) {
    const int size = GetParam();
    this->runTest<ppb::ImplSlangCuda<float>>(size);
}

INSTANTIATE_TEST_SUITE_P(BySize, NBodyTest, ::testing::Values(10, 100));