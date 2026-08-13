message(STATUS "Setting up glm (OpenGL Mathematics) Library")
set(glm_VERSION 1.0.3)

find_package(glm ${glm_VERSION} QUIET)

if (TARGET glm::glm)
    message(STATUS "Found existing glm (OpenGL Mathematics) Library: ${glm_DIR}")
else ()
    message(STATUS "Using glm (OpenGL Mathematics) from GitHub Release ${glm_VERSION}")
    include(FetchContent)

    FetchContent_Declare(
            glm
            URL https://github.com/g-truc/glm/archive/refs/tags/${glm_VERSION}.tar.gz
            URL_HASH SHA256=6775e47231a446fd086d660ecc18bcd076531cfedd912fbd66e576b118607001
    )
    FetchContent_MakeAvailable(glm)
endif ()
