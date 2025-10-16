#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "MatrixMultiplicationTest.h"
#include "matrixMultiplication/cpp/Impl_Cpp.h"


TEST_P(MatrixMultiplicationTest, ImplCpp_Implementation) {
    const int size = GetParam();
    this->runTest<ppb::ImplCpp<float>>(size);
}

INSTANTIATE_TEST_SUITE_P(BySize,MatrixMultiplicationTest, ::testing::Values(2));