message(STATUS "Setting up Metal CPP")

# Include FetchContent module
include(FetchContent)

# Declare and fetch the metal-cpp project
FetchContent_Declare(
        MetalCpp
        URL "${PROJECT_SOURCE_DIR}/libs/metal-cpp.zip"
        URL_MD5 "4f6df7e5255c1ae335c3c1ceab9b6888"
)

# Fetch the content
FetchContent_MakeAvailable(MetalCpp)
# Create an alias for the METAL_CPP target
add_library(MetalCpp::metal-cpp ALIAS METAL_CPP)
target_compile_features(METAL_CPP PUBLIC cxx_std_17)