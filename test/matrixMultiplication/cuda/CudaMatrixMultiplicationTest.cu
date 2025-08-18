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

    const std::vector<float> matrixA_rowMajor = {1, 2, 3, 4};
    const std::vector<float> matrixB_rowMajor = {5, 6, 7, 8};
    const std::vector<float> matrixC_rowMajor = {19, 22, 43, 50};

};

TEST_P(MatrixMultiplicationTest, CudaImplementation_AllSizes) {
    using namespace testing;
    using namespace ppb;

    const int size = GetParam();

    if (size == 2) {
        ImplCuda<float> cudaMatMul{};
        ImplCpp<float> cppMatMul{};
        static_assert(ImplCuda<float>::row_major::value == ImplCpp<float>::row_major::value, "Memory Layout must be the same for both implementations");
        if constexpr (ImplCuda<float>::row_major::value) {
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
    MatrixMultiplication<ImplCuda<float>> cudaMatMul{size};
    const auto expectedResult = cppMatMul();
    const auto actualResult = cudaMatMul();

    ASSERT_THAT(actualResult, Pointwise(FloatNear(EPSILON), expectedResult));
}

INSTANTIATE_TEST_SUITE_P(BySize,MatrixMultiplicationTest, ::testing::Values(2, 10, 32, 50, 64, 512));