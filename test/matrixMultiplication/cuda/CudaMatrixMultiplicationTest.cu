#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "MatrixMultiplication.h"
#include "cpp/Impl_Cpp.h"
#include "cuda/Impl_Cuda.cuh"


class MatrixMultiplicationTest : public ::testing::Test {
protected:

    static constexpr double EPSILON = 1e-4;

    const std::vector<float> matrixA = {1, 3, 2, 4};
    const std::vector<float> matrixB = {5, 7, 6, 8};
    const std::vector<float> matrixC = {19, 43, 22, 50};

};

TEST_F(MatrixMultiplicationTest, CudaImplementation_Size2x2) {
    using namespace testing;
    using namespace ppb;

    constexpr size_t SIZE = 2;
    ImplCpp<float> matMul{};
    auto actualResult = matMul(matrixA, matrixB, {SIZE, SIZE, SIZE});

    ASSERT_THAT(actualResult, Pointwise(FloatEq(), matrixC));
}

TEST_F(MatrixMultiplicationTest, CudaImplementation_Size10x10) {
    using namespace testing;
    using namespace ppb;

    constexpr size_t SIZE = 10;
    MatrixMultiplication<ImplCpp<float>> cppMatMul{SIZE};
    MatrixMultiplication<ImplCuda<float>> cudaMatMul{SIZE};
    const auto expectedResult = cppMatMul();
    const auto actualResult = cudaMatMul();

    ASSERT_THAT(actualResult, Pointwise(FloatNear(EPSILON), expectedResult));
}

TEST_F(MatrixMultiplicationTest, CudaImplementation_Size32x32) {
    using namespace testing;
    using namespace ppb;

    constexpr size_t SIZE = 32;
    MatrixMultiplication<ImplCpp<float>> cppMatMul{SIZE};
    MatrixMultiplication<ImplCuda<float>> cudaMatMul{SIZE};
    const auto expectedResult = cppMatMul();
    const auto actualResult = cudaMatMul();

    ASSERT_THAT(actualResult, Pointwise(FloatNear(EPSILON), expectedResult));
}

TEST_F(MatrixMultiplicationTest, CudaImplementation_Size64x64) {
    using namespace testing;
    using namespace ppb;

    constexpr size_t SIZE = 64;
    MatrixMultiplication<ImplCpp<float>> cppMatMul{SIZE};
    MatrixMultiplication<ImplCuda<float>> cudaMatMul{SIZE};
    const auto expectedResult = cppMatMul();
    const auto actualResult = cudaMatMul();

    ASSERT_THAT(actualResult, Pointwise(FloatNear(EPSILON), expectedResult));
}

TEST_F(MatrixMultiplicationTest, CudaImplementation_Size512x512) {
    using namespace testing;
    using namespace ppb;

    constexpr size_t SIZE = 512;
    MatrixMultiplication<ImplCpp<float>> cppMatMul{SIZE};
    MatrixMultiplication<ImplCuda<float>> cudaMatMul{SIZE};
    const auto expectedResult = cppMatMul();
    const auto actualResult = cudaMatMul();

    ASSERT_THAT(actualResult, Pointwise(FloatNear(EPSILON), expectedResult));
}