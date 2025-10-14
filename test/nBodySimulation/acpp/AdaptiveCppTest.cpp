#include "GoogleTestMatcher.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "NBodySimulation.h"
#include "cpp/Impl_Cpp.h"
#include "acpp/Impl_AdaptiveCpp.h"


class NBodyAcppTest : public ::testing::TestWithParam<size_t> {

};

TEST_P(NBodyAcppTest, AcppImplementation) {
    using namespace testing;
    using namespace ppb;

    const size_t size = GetParam();
    NBodySimulation<ImplCpp<float>> nBodyCpp{size};
    NBodySimulation<ImplAdaptiveCpp<float>> nBodyAcpp{size};

    const auto expectedResult = nBodyCpp();
    const auto actualResult = nBodyAcpp();


    ASSERT_THAT(actualResult, ParticleContainter1D(expectedResult));
}

INSTANTIATE_TEST_SUITE_P(BySize,NBodyAcppTest, ::testing::Values(10, 100, 1000));