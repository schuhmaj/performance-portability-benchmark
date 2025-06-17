#include "../GoogleTestMatcher.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "NBodySimulation.h"
#include "cpp/Impl_Cpp.h"
#include "kokkos/Impl_Kokkos.h"


class NBodyKokkosTest : public ::testing::Test {

};

TEST_F(NBodyKokkosTest, KokkosImplementation_Size10) {
    using namespace testing;
    using namespace ppb;

    constexpr size_t SIZE = 10;
    NBodySimulation<ImplCpp<float>> nBodyCpp{SIZE};
    NBodySimulation<ImplKokkos<float>> nBodyKokkos{SIZE};

    const auto expectedResult = nBodyCpp();
    const auto actualResult = nBodyKokkos();


    ASSERT_THAT(actualResult, ParticleContainter1D(expectedResult));
}

TEST_F(NBodyKokkosTest, KokkosImplementation_Size100) {
    using namespace testing;
    using namespace ppb;

    constexpr size_t SIZE = 100;
    NBodySimulation<ImplCpp<float>> nBodyCpp{SIZE};
    NBodySimulation<ImplKokkos<float>> nBodyKokkos{SIZE};

    const auto expectedResult = nBodyCpp();
    const auto actualResult = nBodyKokkos();


    ASSERT_THAT(actualResult, ParticleContainter1D(expectedResult));
}