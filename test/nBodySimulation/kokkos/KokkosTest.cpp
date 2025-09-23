#include "GoogleTestMatcher.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "nBodySimulation/NBodySimulation.h"
#include "nBodySimulation/cpp/Impl_Cpp.h"
#include "nBodySimulation/kokkos/Impl_Kokkos.h"


class NBodyKokkosTest : public ::testing::TestWithParam<size_t> {

};

TEST_P(NBodyKokkosTest, KokkosImplementation) {
    using namespace testing;
    using namespace ppb;

    const size_t size = GetParam();
    NBodySimulation<ImplCpp<float>> nBodyCpp{size};
    NBodySimulation<ImplKokkos<float>> nBodyKokkos{size};

    const auto expectedResult = nBodyCpp();
    const auto actualResult = nBodyKokkos();


    ASSERT_THAT(actualResult, ParticleContainter1D(expectedResult));
}

INSTANTIATE_TEST_SUITE_P(BySize,NBodyKokkosTest, ::testing::Values(10, 100, 1000));