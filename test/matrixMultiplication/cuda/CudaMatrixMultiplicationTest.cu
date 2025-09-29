#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "matrixMultiplication/MatrixMultiplication.h"
#include "matrixMultiplication/cpp/Impl_Cpp.h"
#include "matrixMultiplication/cuda/Impl_Cuda.cuh"
#include "matrixMultiplication/cuda/Impl_CudaTensor.cuh"
#include "matrixMultiplication/cuda/Impl_CudaBuffer.cuh"
#include "matrixMultiplication/cuda/Impl_Cublas.cuh"
#include "matrixMultiplication/cuda/Impl_CudaNaive.cuh"


class MatrixMultiplicationTest : public ::testing::TestWithParam<int> {
protected:

    static constexpr double EPSILON = 1e-4;
    static constexpr double HALF_EPSILON = 1e-2;

    const std::vector<float> matrixA = {1, 3, 2, 4};
    const std::vector<float> matrixB = {5, 7, 6, 8};
    const std::vector<float> matrixC = {19, 43, 22, 50};

    const std::vector<float> matrixA_rowMajor = {1, 2, 3, 4};
    const std::vector<float> matrixB_rowMajor = {5, 6, 7, 8};
    const std::vector<float> matrixC_rowMajor = {19, 22, 43, 50};


    template<typename Implementation>
    void runTest(const int size) {
        using namespace testing;
        using namespace ppb;

        if (size == 2) {
            Implementation cudaMatMul{};
            ImplCpp<float> cppMatMul{};
            static_assert(Implementation::row_major::value == ImplCpp<float>::row_major::value, "Memory Layout must be the same for both implementations");
            if constexpr (Implementation::row_major::value) {
                const auto [expectedResult, t1] = cppMatMul(matrixA_rowMajor, matrixB_rowMajor, {size, size, size});
                const auto [actualResult, t2] = cudaMatMul(matrixA_rowMajor, matrixB_rowMajor, {size, size, size});
                ASSERT_THAT(matrixC_rowMajor, Pointwise(FloatEq(), actualResult));
                ASSERT_THAT(matrixC_rowMajor, Pointwise(FloatEq(), expectedResult));
            } else {
                const auto [expectedResult, t1] = cppMatMul(matrixA, matrixB, {size, size, size});
                const auto [actualResult, t2] = cudaMatMul(matrixA, matrixB, {size, size, size});
                ASSERT_THAT(matrixC, Pointwise(FloatEq(), actualResult));
                ASSERT_THAT(matrixC, Pointwise(FloatEq(), expectedResult));
            }
            return;
        }
        if (std::is_same<Implementation, ImplCudaTensor<float>>::value && (size == 2 || size == 10 || size == 50)) {
            GTEST_SKIP() << "Skipping test for size " << size;
            return;
        }

        MatrixMultiplication<ImplCpp<float>> cppMatMul{size};
        MatrixMultiplication<Implementation> cudaMatMul{size};
        const auto [expectedResult, t1] = cppMatMul();
        const auto [actualResult, t2] = cudaMatMul();

        if (std::is_same<Implementation, ImplCudaTensor<float>>::value) {
            ASSERT_THAT(actualResult, Pointwise(FloatNear(HALF_EPSILON), expectedResult));
        } else {
            ASSERT_THAT(actualResult, Pointwise(FloatNear(EPSILON), expectedResult));
        }
    }

};

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