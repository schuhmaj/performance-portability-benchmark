# Patches RAJA v2025.12.2 for the CUDA Toolkit >= 13, whose cudaMemAdvise signature takes a
# cudaMemLocation struct instead of a plain device id (executed as FetchContent PATCH_COMMAND
# with the RAJA source directory as working directory).
# The patch is idempotent - re-running it on an already patched source is a no-op.

set(MEM_UTILS_FILE "include/RAJA/policy/cuda/MemUtils_CUDA.hpp")

file(READ ${MEM_UTILS_FILE} FILE_CONTENT)

set(ORIGINAL_SNIPPET
"    CAMP_CUDA_API_INVOKE_AND_CHECK(cudaMemAdvise, ptr, nbytes,
                                   cudaMemAdviseSetPreferredLocation, device);
    CAMP_CUDA_API_INVOKE_AND_CHECK(cudaMemAdvise, ptr, nbytes,
                                   cudaMemAdviseSetAccessedBy, cudaCpuDeviceId);")

set(PATCHED_SNIPPET
"#if CUDART_VERSION >= 13000
    // cudaMemLocation cannot be routed through CAMP_CUDA_API_INVOKE_AND_CHECK
    // (camp streams the arguments on failure and the struct has no operator<<)
    cudaMemLocation preferredLocation;
    preferredLocation.type = cudaMemLocationTypeDevice;
    preferredLocation.id = device;
    cudaMemLocation accessedByLocation;
    accessedByLocation.type = cudaMemLocationTypeHost;
    accessedByLocation.id = 0;
    cudaError_t adviseError = cudaMemAdvise(ptr, nbytes,
                                            cudaMemAdviseSetPreferredLocation,
                                            preferredLocation);
    if (adviseError == cudaSuccess) {
      adviseError = cudaMemAdvise(ptr, nbytes, cudaMemAdviseSetAccessedBy,
                                  accessedByLocation);
    }
    if (adviseError != cudaSuccess) {
      throw std::runtime_error(std::string(\"cudaMemAdvise failed: \") +
                               cudaGetErrorString(adviseError));
    }
#else
${ORIGINAL_SNIPPET}
#endif")

if (FILE_CONTENT MATCHES "CUDART_VERSION >= 13000")
    message(STATUS "RAJA CUDA 13 patch already applied")
else ()
    string(REPLACE "${ORIGINAL_SNIPPET}" "${PATCHED_SNIPPET}" FILE_CONTENT "${FILE_CONTENT}")
    if (NOT FILE_CONTENT MATCHES "CUDART_VERSION >= 13000")
        message(FATAL_ERROR "Failed to apply the RAJA CUDA 13 patch to ${MEM_UTILS_FILE}")
    endif ()
    file(WRITE ${MEM_UTILS_FILE} "${FILE_CONTENT}")
    message(STATUS "Applied the RAJA CUDA 13 patch to ${MEM_UTILS_FILE}")
endif ()
