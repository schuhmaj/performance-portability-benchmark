#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "MatrixMultiplicationTest.h"
#include "matrixMultiplication/kokkos/Impl_Kokkos.h"


TEST_P(MatrixMultiplicationTest, ImplKokkos_Implementation) {
    const int size = GetParam();
    this->runTest<ppb::ImplKokkos<float>>(size);
}

INSTANTIATE_TEST_SUITE_P(BySize,MatrixMultiplicationTest, ::testing::Values(2, 10, 32, 50, 64, 512));