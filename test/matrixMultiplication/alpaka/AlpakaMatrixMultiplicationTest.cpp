#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "MatrixMultiplicationTest.h"
#include "matrixMultiplication/alpaka/Impl_Alpaka.h"

TEST_P(MatrixMultiplicationTest, ImplAlpaka_Implementation) {
    const int size = GetParam();
    this->runTest<ppb::ImplAlpaka<float>>(size);
}

INSTANTIATE_TEST_SUITE_P(BySize,MatrixMultiplicationTest, ::testing::Values(2, 10, 32, 50, 64, 512));