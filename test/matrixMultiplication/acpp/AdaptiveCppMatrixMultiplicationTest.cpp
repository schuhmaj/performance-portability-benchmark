#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "MatrixMultiplicationTest.h"
#include "matrixMultiplication/acpp/Impl_AdaptiveCpp.h"
#include "matrixMultiplication/acpp/Impl_AdaptiveCppShr.h"

TEST_P(MatrixMultiplicationTest, ImplAdaptiveCpp_Implementation) {
    const int size = GetParam();
    this->runTest<ppb::ImplAdaptiveCpp<float>>(size);
}

TEST_P(MatrixMultiplicationTest, ImplAdaptiveCppShr_Implementation) {
    const int size = GetParam();
    this->runTest<ppb::ImplAdaptiveCppShr<float>>(size);
}

INSTANTIATE_TEST_SUITE_P(BySize,MatrixMultiplicationTest, ::testing::Values(2, 10, 32, 50, 64, 512));