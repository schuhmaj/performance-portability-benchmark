#include "gtest/gtest.h"
#include "polyhedralGravity/PolyhedralGravityDefinitions.h"

int main(int argc, char *argv[]) {
    testing::InitGoogleTest(&argc, argv);
    GlobalResources resources{argc, argv};
    int result = RUN_ALL_TESTS();
    return result;
}