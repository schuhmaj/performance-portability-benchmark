#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "MatrixMultiplicationTest.h"
#include "matrixMultiplication/boost/Impl_Boost.h"


TEST_P(MatrixMultiplicationTest, ImplBoost_Implementation) {
    const int size = GetParam();
    this->runTest<ppb::ImplBoost<float>>(size);
}

INSTANTIATE_TEST_SUITE_P(BySize,MatrixMultiplicationTest, ::testing::Values(2, 10, 32, 50, 64, 512));