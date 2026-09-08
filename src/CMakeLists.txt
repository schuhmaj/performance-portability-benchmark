set(PPB_SRC_SOURCE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

# Both LIKWID marker flavours are compiled in; src/common/Marker.h decides per
# region whether it reads the CPU or the Nvidia GPU counters.
if (${PPB_ENABLE_LIKWID})
    add_compile_definitions(PPB_ENABLE_LIKWID LIKWID_PERFMON LIKWID_NVMON)
    include_directories(${LIKWID_INCLUDE_DIRS})
    link_libraries(likwid::likwid)
endif ()

# NVTX is header-only and dlopens its implementation, so the header path plus
# libdl is the whole dependency.
if (${PPB_ENABLE_NVTX})
    add_compile_definitions(PPB_ENABLE_NVTX)
    include_directories(${PPB_NVTX_INCLUDE_DIR})
    link_libraries(${CMAKE_DL_LIBS})
endif ()

# Symbol and line information for the profiling build.
#
# A profiler can only report what the binary tells it. Without line tables it
# still names kernels - those names come from the ELF symbol table, which is
# present in any non-stripped build - but it cannot map a hot SASS instruction
# back to a source line, and a sampling profiler (HPCToolkit, TAU) cannot
# unwind through an optimized frame. The flags below add exactly that
# information and nothing else, because everything here has to keep generating
# the same machine code: a roofline measured from a differently optimized
# binary describes a different program.
#
#   -lineinfo (nvcc)          SASS-to-source line table, no effect on codegen
#   -g1 (gcc, clang, hipcc)   line tables and backtrace info, no local variables.
#                             Spelled -g1 rather than clang's -gline-tables-only
#                             because Kokkos compiles through a launcher that
#                             hands the flag to the system c++, which may be gcc
#                             no matter what CMAKE_CXX_COMPILER_ID says
#   -gopt (nvc++)             debug info while keeping the optimizer on; plain
#                             -g would fall back to -O0
#   -fno-omit-frame-pointer   lets a sampler walk the host call stack
#
# Deliberately absent is nvcc's -G (--device-debug), which is what
# PPB_PROFILING_DEVICE_DEBUG adds back for someone who wants to step through a
# kernel in cuda-gdb. -G disables every device optimization, spills registers
# to local memory and changes the instruction mix outright, so the measured
# arithmetic intensity and achieved bandwidth would no longer be those of the
# benchmark. Full -g on the host is left out for a smaller reason: it is not
# free on this code base (Kokkos, RAJA and Alpaka templates) and the profilers
# in use here need line tables, not local variable locations.
if (${PPB_PROFILING})
    add_compile_options(
            $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:GNU,Clang,AppleClang,IntelLLVM>>:-g1>
            $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:GNU,Clang,AppleClang,IntelLLVM>>:-fno-omit-frame-pointer>
            $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:NVHPC>>:-gopt>
            $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CUDA_COMPILER_ID:NVIDIA>>:-lineinfo>
            $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CUDA_COMPILER_ID:Clang>>:-g1>
            $<$<COMPILE_LANGUAGE:HIP>:-g1>
    )
    if (${PPB_PROFILING_DEVICE_DEBUG})
        add_compile_options($<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CUDA_COMPILER_ID:NVIDIA>>:-G>)
    endif ()
endif ()
add_subdirectory(common)
if (${PPB_ENABLE_VectorAddition})
    add_subdirectory(vectorAdditon)
endif ()
if (${PPB_ENABLE_MatrixMultiplication})
    add_subdirectory(matrixMultiplication)
endif ()
if (${PPB_ENABLE_NBodySimulation})
    add_subdirectory(nBodySimulation)
endif ()
if (${PPB_ENABLE_PolyhedralGravity})
    add_subdirectory(polyhedralGravity)
endif ()