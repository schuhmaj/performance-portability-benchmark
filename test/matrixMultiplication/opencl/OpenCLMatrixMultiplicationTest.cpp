#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "MatrixMultiplicationTest.h"
#include "matrixMultiplication/opencl/Impl_OpenCL.h"

TEST_P(MatrixMultiplicationTest, ImplOpenCL_Implementation) {
    const int size = GetParam();
    this->runTest<ppb::ImplOpenCL<float>>(size);
}

INSTANTIATE_TEST_SUITE_P(BySize,MatrixMultiplicationTest, ::testing::Values(2, 10, 32, 50, 64, 512));