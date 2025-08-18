#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "MatrixMultiplication.h"
#include "cpp/Impl_Cpp.h"
#include "cuda/Impl_Cuda.cuh"


class MatrixMultiplicationTest : public ::testing::TestWithParam<size_t> {
protected:

    static constexpr double EPSILON = 1e-4;

    const std::vector<float> matrixA = {1, 3, 2, 4};
    const std::vector<float> matrixB = {5, 7, 6, 8};
    const std::vector<float> matrixC = {19, 43, 22, 50};

};

TEST_P(MatrixMultiplicationTest, CudaImplementation_AllSizes) {
    using namespace testing;
    using namespace ppb;

    const int size = GetParam();

    if (size == 2) {
        ImplCpp<float> matMul{};
        const auto actualResult = matMul(matrixA, matrixB, {size, size, size});
        ASSERT_THAT(actualResult, Pointwise(FloatEq(), matrixC));
        return;
    }

    MatrixMultiplication<ImplCpp<float>> cppMatMul{size};
    MatrixMultiplication<ImplCuda<float>> cudaMatMul{size};
    const auto expectedResult = cppMatMul();
    const auto actualResult = cudaMatMul();

    ASSERT_THAT(actualResult, Pointwise(FloatNear(EPSILON), expectedResult));
}

INSTANTIATE_TEST_SUITE_P(BySize,MatrixMultiplicationTest, ::testing::Values(2, 10, 32, 50, 64, 512));