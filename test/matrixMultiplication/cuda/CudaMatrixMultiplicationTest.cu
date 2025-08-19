#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "MatrixMultiplication.h"
#include "cpp/Impl_Cpp.h"
#include "cuda/Impl_Cuda.cuh"
#include "cuda/Impl_CudaTensor.cuh"
#include "cuda/Impl_CudaBuffer.cuh"
#include "cuda/Impl_Cublas.cuh"


class MatrixMultiplicationTest : public ::testing::TestWithParam<size_t> {
protected:

    static constexpr double EPSILON = 1e-4;

    const std::vector<float> matrixA = {1, 3, 2, 4};
    const std::vector<float> matrixB = {5, 7, 6, 8};
    const std::vector<float> matrixC = {19, 43, 22, 50};

    const std::vector<float> matrixA_rowMajor = {1, 2, 3, 4};
    const std::vector<float> matrixB_rowMajor = {5, 6, 7, 8};
    const std::vector<float> matrixC_rowMajor = {19, 22, 43, 50};


    template<typename Implementation>
    void runTest(const size_t size) {
        using namespace testing;
        using namespace ppb;

        if (size == 2) {
            Implementation cudaMatMul{};
            ImplCpp<float> cppMatMul{};
            static_assert(Implementation::row_major::value == ImplCpp<float>::row_major::value, "Memory Layout must be the same for both implementations");
            if constexpr (Implementation::row_major::value) {
                const auto expectedResult = cppMatMul(matrixA_rowMajor, matrixB_rowMajor, {size, size, size});
                const auto actualResult = cudaMatMul(matrixA_rowMajor, matrixB_rowMajor, {size, size, size});
                ASSERT_THAT(matrixC_rowMajor, Pointwise(FloatEq(), actualResult));
                ASSERT_THAT(matrixC_rowMajor, Pointwise(FloatEq(), expectedResult));
            } else {
                const auto expectedResult = cppMatMul(matrixA, matrixB, {size, size, size});
                const auto actualResult = cudaMatMul(matrixA, matrixB, {size, size, size});
                ASSERT_THAT(matrixC, Pointwise(FloatEq(), actualResult));
                ASSERT_THAT(matrixC, Pointwise(FloatEq(), expectedResult));
            }
            return;
        }

        MatrixMultiplication<ImplCpp<float>> cppMatMul{size};
        MatrixMultiplication<Implementation> cudaMatMul{size};
        const auto expectedResult = cppMatMul();
        const auto actualResult = cudaMatMul();

        ASSERT_THAT(actualResult, Pointwise(FloatNear(EPSILON), expectedResult));
    }

};

TEST_P(MatrixMultiplicationTest, CudaImplementation_AllSizes) {
    const int size = GetParam();
    this->runTest<ppb::ImplCuda<float>>(size);
}

TEST_P(MatrixMultiplicationTest, CudaTensorImplementation_AllSizes) {
    const int size = GetParam();
    this->runTest<ppb::ImplCudaTensor<float>>(size);
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