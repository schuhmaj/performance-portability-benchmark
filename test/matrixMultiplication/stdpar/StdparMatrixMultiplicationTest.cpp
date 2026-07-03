#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "MatrixMultiplicationTest.h"
#include "matrixMultiplication/stdpar/Impl_Stdpar.h"

TEST_P(MatrixMultiplicationTest, Stdpar_Implementation) {
    const int size = GetParam();
    this->runTest<ppb::ImplStdpar<float>>(size);
}

INSTANTIATE_TEST_SUITE_P(BySize, MatrixMultiplicationTest, ::testing::Values(2, 10, 32, 50, 64, 512));
