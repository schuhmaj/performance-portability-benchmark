# fix_generated_cuda.cmake - Script executed at build time
# Do not call by itself
# Used to modify the generated *.cuh file in terms of main kernel name and CUDA compliance
if (NOT DEFINED INPUT_FILE)
    message(FATAL_ERROR "INPUT_FILE is required")
endif ()

cmake_path(GET INPUT_FILE FILENAME input_name)
cmake_path(GET INPUT_FILE PARENT_PATH input_dir)

string(REGEX REPLACE "^temp_" "" output_name "${input_name}")
set(output_file "${input_dir}/${output_name}")

cmake_path(GET output_file STEM output_stem)

file(READ "${INPUT_FILE}" generated_cuda)

string(REPLACE
        "extern \"C\" __constant__ GlobalParams_0"
        "__device__ GlobalParams_0"
        generated_cuda
        "${generated_cuda}"
)

string(REPLACE
        "void computeMain()"
        "void run_${output_stem}()"
        generated_cuda
        "${generated_cuda}"
)

file(WRITE "${output_file}" "${generated_cuda}")
