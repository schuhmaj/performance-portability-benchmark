#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "MatrixMultiplicationTest.h"
#include "matrixMultiplication/openmp/Impl_OpenMP.h"
#include "matrixMultiplication/openmp/Impl_OpenMPDevice.h"

TEST_P(MatrixMultiplicationTest, OpenMP_CPU_Implementation) {
    const int size = GetParam();
    this->runTest<ppb::ImplOpenMP<float>>(size);
}

TEST_P(MatrixMultiplicationTest, OpenMP_Device_Implementation) {
    const int size = GetParam();
    this->runTest<ppb::ImplOpenMPDevice<float>>(size);
}


INSTANTIATE_TEST_SUITE_P(BySize,MatrixMultiplicationTest, ::testing::Values(2, 10, 32, 50, 64, 512));