#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "MatrixMultiplicationTest.h"
#include "matrixMultiplication/cuda/Impl_Cuda.cuh"
#include "matrixMultiplication/cuda/Impl_CudaTensor.cuh"
#include "matrixMultiplication/cuda/Impl_CudaBuffer.cuh"
#include "matrixMultiplication/cuda/Impl_Cublas.cuh"
#include "matrixMultiplication/cuda/Impl_CudaNaive.cuh"

TEST_P(MatrixMultiplicationTest, CudaImplementation_AllSizes) {
    const int size = GetParam();
    this->runTest<ppb::ImplCuda<float>>(size);
}

TEST_P(MatrixMultiplicationTest, CudaNaiveImplementation_AllSizes) {
    const int size = GetParam();
    this->runTest<ppb::ImplCudaNaive<float>>(size);
}

TEST_P(MatrixMultiplicationTest, CudaTensorImplementation_AllSizes) {
    const int size = GetParam();
    if (size == 2 || size == 10 || size == 50) {
        GTEST_SKIP() << "Skipping test for size " << size;
        return;
    }
    this->runTest<ppb::ImplCudaTensor<float>>(size, HALF_EPSILON);
}

TEST_P(MatrixMultiplicationTest, CudaBufferImplementation_AllSizes) {
    const int size = GetParam();
    this->runTest<ppb::ImplCudaBuffer<float>>(size);
}

TEST_P(MatrixMultiplicationTest, CublasImplementation_AllSizes) {
    const int size = GetParam();
    this->runTest<ppb::ImplCublas<float>>(size);
}

INSTANTIATE_TEST_SUITE_P(BySize,MatrixMultiplicationTest, ::testing::Values(2, 10, 32, 50, 64, 512));