import subprocess
import os
import json
import pandas as pd

import benchmarking_nBody



incl = ["vulkan_naive", "vulkan_cells", "vulkan_verlet", "slang->vulkan_naive", "slang->vulkan_cells", "slang->vulkan_verlet", "slang->ptx_naive", "slang->ptx_cells", "slang->ptx_verlet", "cpp_naive"]

executables = {
    "vulkan_naive"         : "../build/src/nBodySimulation/vulkan/naive/nbody_vulkan_naive",
    "vulkan_cells"         : "../build/src/nBodySimulation/vulkan/cell_lists/nbody_vulkan_cells",
    "vulkan_verlet"        : "../build/src/nBodySimulation/vulkan/verlet_lists/nbody_vulkan_verlet",
    "slang->vulkan_naive"  : "../build/src/nBodySimulation/slang/naive/nbody_slang_vulkan_naive", 
    "slang->vulkan_cells"  : "../build/src/nBodySimulation/slang/cell_lists/nbody_slang_vulkan_cells", 
    "slang->vulkan_verlet" : "../build/src/nBodySimulation/slang/verlet_lists/nbody_slang_vulkan_verlet", 
    "slang->ptx_naive"     : "../build/src/nBodySimulation/slang/naive/nbody_slang_cuda_naive", 
    "slang->ptx_cells"     : "../build/src/nBodySimulation/slang/cell_lists/nbody_slang_cuda_cells", 
    "slang->ptx_verlet"    : "../build/src/nBodySimulation/slang/verlet_lists/nbody_slang_cuda_verlet", 
    "cpp_naive"            : "../build/src/nBodySimulation/cpp/nbody_cpp"
}

tests = {
    "vulkan_naive"         : "../build/test/nBodySimulation/vulkan/naive/nbody_vulkan_naive_test",
    "vulkan_cells"         : "../build/test/nBodySimulation/vulkan/cell_lists/nbody_vulkan_cells_test",
    "vulkan_verlet"        : "../build/test/nBodySimulation/vulkan/verlet_lists/nbody_vulkan_verlet_test",
    "slang->vulkan_naive"  : "../build/test/nBodySimulation/slang/vulkan/naive/nbody_slang_vulkan_naive_test", 
    "slang->vulkan_cells"  : "../build/test/nBodySimulation/slang/vulkan/cell_lists/nbody_slang_vulkan_cells_test", 
    "slang->vulkan_verlet" : "../build/test/nBodySimulation/slang/vulkan/verlet_lists/nbody_slang_vulkan_verlet_test", 
    "slang->ptx_naive"     : "../build/test/nBodySimulation/slang/cuda/naive/nbody_slang_cuda_naive_test", 
    "slang->ptx_cells"     : "../build/test/nBodySimulation/slang/cuda/cell_lists/nbody_slang_cuda_cells_test", 
    "slang->ptx_verlet"    : "../build/test/nBodySimulation/slang/cuda/verlet_lists/nbody_slang_cuda_verlet_test", 
    "cpp_naive"            : "../build/test/nBodySimulation/cpp/nbody_cpp_test"

}

results = {
    "vulkan_naive"         : "../results/nBody_vulkan_naive.json",
    "vulkan_cells"         : "../results/nBody_vulkan_cells.json",
    "vulkan_verlet"        : "../results/nBody_vulkan_verlet.json",
    "slang->vulkan_naive"  : "../results/nBody_slang_vulkan_naive.json", 
    "slang->vulkan_cells"  : "../results/nBody_slang_vulkan_cells.json", 
    "slang->vulkan_verlet" : "../results/nBody_slang_vulkan_verlet.json", 
    "slang->ptx_naive"     : "../results/nBody_slang_cuda_naive.json", 
    "slang->ptx_cells"     : "../results/nBody_slang_cuda_cells.json", 
    "slang->ptx_verlet"    : "../results/nBody_slang_cuda_verlet.json", 
    "cpp_naive"            : "../results/nBody_cpp_naive.json"
}




git_hash = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
if __name__ == "__main__":
    print("searching hyper parameters...")
    try:
        print("Searching hyper parameter TILE_SIZE...")
        incl_search_TILE_SIZE = ["vulkan_verlet", "slang->vulkan_verlet", "vulkan_cells", "slang->vulkan_cells", "slang->ptx_cells", "slang->ptx_verlet"]
        search_range_TILE_SIZE = [2**i for i in range(4, 11)]
        dataframes_TILE_SIZE = benchmarking_nBody.search_TILE_SIZE(executables, incl_search_TILE_SIZE, search_range_TILE_SIZE)
        benchmarking_nBody.plot_summary(dataframes_TILE_SIZE, groupby="TILE_SIZE", xlabel="TILE_SIZE", title="TILE_SIZE vs real time", show=False)

        print("Searching hyper parameter h...")
        incl_search_h = ["vulkan_cells", "slang->vulkan_cells", "slang->ptx_cells"]
        search_range_h = [9.0 * i for i in range(1, 17)]
        dataframes_h = benchmarking_nBody.search_h(executables, incl_search_h, search_range_h)
        benchmarking_nBody.plot_summary(dataframes_h, groupby="h", xlabel="h", title="h vs real time", show=False)

        print("Searching hyper parameter influenceRadius...")
        incl_search_inflRad = ["vulkan_verlet", "slang->vulkan_verlet", "slang->ptx_verlet"]
        search_range_inflRad = [4.0 * i for i in range(1, 17)]
        dataframes_inflRad = benchmarking_nBody.search_influenceRadius(executables, incl_search_inflRad, search_range_inflRad)
        benchmarking_nBody.plot_summary(dataframes_inflRad, groupby="Influence Radius", xlabel="Influence Radius", title="Influence Radius vs real time", show=False)

        # print("Searching hyper parameter interval_neighbor_search...")
        # incl_search_interval = ["vulkan_verlet", "slang->vulkan_verlet", "vulkan_cells", "slang->vulkan_cells"]
        # search_range_interval = [5 * i for i in range(1, 10)]
        # highest_pass = benchmarking_nBody.search_interval_neighbor_search(executables=tests, incl=incl_search_interval, search_range=search_range_interval)
        # print("Highest passing interval_neighbor_search:", highest_pass)
    except:
        raise

    finally:
        # reset constants
        old_tile_size = int(benchmarking_nBody.find_const_in_file("TILE_SIZE"))
        new_tile_size = 256
        benchmarking_nBody.update_tile_size(old_tile_size=old_tile_size, new_tile_size=new_tile_size)

        old_h = float(benchmarking_nBody.find_const_in_file(search_string="FloatType h"))
        h = 9.0
        new_conf_line_h = f"FloatType h{{{h}}};"
        old_conf_line_h = f"FloatType h{{{old_h}}};"
        benchmarking_nBody.repl(old=old_conf_line_h, new=new_conf_line_h)

        old_inflRad = float(benchmarking_nBody.find_const_in_file(search_string="FloatType influenceRadius"))
        inflRad = 4.0
        new_conf_line_inflRad = f"FloatType influenceRadius{{{inflRad}}};"
        old_conf_line_inflRad = f"FloatType influenceRadius{{{old_inflRad}}};"
        benchmarking_nBody.repl(old=old_conf_line_inflRad, new=new_conf_line_inflRad)

        old_interval = int(benchmarking_nBody.find_const_in_file(search_string="static constexpr uint interval_neighbor_search"))
        interval = 10
        new_conf_line_interval = f"static constexpr uint interval_neighbor_search{{{interval}}};"
        old_conf_line_interval = f"static constexpr uint interval_neighbor_search{{{old_interval}}};"
        benchmarking_nBody.repl(old=old_conf_line_interval, new=new_conf_line_interval)
