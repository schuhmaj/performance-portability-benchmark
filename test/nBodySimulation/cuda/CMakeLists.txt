if (${PPB_ENABLE_Cuda_DirectSum})
    add_subdirectory(direct_sum)
endif ()

if (${PPB_ENABLE_Cuda_LinkedCells})
    add_subdirectory(linked_cells)
endif ()

if (${PPB_ENABLE_Cuda_VerletLists})
    add_subdirectory(verlet_lists)
endif ()