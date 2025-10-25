#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "NBodyTest.h"
#include "nBodySimulation/openacc/Impl_OpenACC.h"

TEST_P(NBodyTest, ImplOpenACC_Implementation) {
    const int size = GetParam();
    this->runTest<ppb::ImplOpenACC<float>>(size);
}

INSTANTIATE_TEST_SUITE_P(BySize, NBodyTest, ::testing::Values(10, 100));