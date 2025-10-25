#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "MatrixMultiplicationTest.h"
#include "matrixMultiplication/openacc/Impl_OpenACC.h"

TEST_P(MatrixMultiplicationTest, OpenACC_Implementation) {
    const int size = GetParam();
    this->runTest<ppb::ImplOpenACC<float>>(size);
}


INSTANTIATE_TEST_SUITE_P(BySize,MatrixMultiplicationTest, ::testing::Values(2, 10, 32, 50, 64, 512));