find_path(LIKWID_INCLUDE_DIR likwid.h)
find_library(LIKWID_LIBRARY likwid)

add_executable(vec_cuda main.cpp Impl_Cuda.cu)
target_link_libraries(vec_cuda PRIVATE vec_lib)
set_target_properties(vec_cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(vec_cuda
    PRIVATE
        vec_lib
        CUDA::cublas
)

target_compile_definitions(vec_cuda PRIVATE LIKWID_PERFMON)
target_include_directories(vec_cuda PRIVATE ${LIKWID_INCLUDE_DIR})
target_link_libraries(vec_cuda
        PRIVATE
        ${LIKWID_LIBRARY}
)