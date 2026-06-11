#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "NBodyTest.h"
#include "nBodySimulation/cuda/linked_cells/Impl_Cuda.cuh"

TEST_P(NBodyTest, ImplCuda_Implementation) {
    const int size = GetParam();
    this->runTest<ppb::ImplCuda<float>>(size);
}

INSTANTIATE_TEST_SUITE_P(BySize, NBodyTest, ::testing::Values(10, 100));