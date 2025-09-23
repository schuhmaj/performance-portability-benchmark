#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "matrixMultiplication/cpp/Impl_Cpp.h"


class MatrixMultiplicationTest : public ::testing::Test {
protected:

    const std::vector<float> matrixA = {1, 3, 2, 4};
    const std::vector<float> matrixB = {5, 7, 6, 8};
    const std::vector<float> matrixC = {19, 43, 22, 50};

};

TEST_F(MatrixMultiplicationTest, CppImplementation_Size2x2) {
    using namespace testing;
    using namespace ppb;

    constexpr size_t SIZE = 10;
    ImplCpp<float> matMul{};
    const auto [actualResult, timeMeasurement] = matMul(matrixA, matrixB, {2, 2, 2});

    ASSERT_THAT(actualResult, Pointwise(FloatEq(), matrixC));
}