import subprocess
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import re

shaders = {
    "naive"        : ["KernelForce", "KernelPosition", "KernelVelocity"],
    "cell_lists"   : ["KernelBlellochScan", "KernelBlockSum", "KernelForce", "KernelHistogram", "KernelIdCells", "KernelPosition", "KernelResetCells", "KernelVelocity"],
    "verlet_lists" : ["KernelBlellochScan", "KernelBlockSum", "KernelCountNeighbors", "KernelForce", "KernelPosition", "KernelVelocity", "KernelVerlet"]
}

vulkan_filetype     = ".comp"
slang_filetype      = ".slang"
vulkan_path         = "../src/nBodySimulation/vulkan/"
slang_path          = "../src/nBodySimulation/slang/"
config_path         = "../src/nBodySimulation/NBodySimulation.h"
temp_result_path = "../results/tmp_result.json"


def repl(old, new, path=config_path):
    with open(path, "r") as f:
        content = f.read()
        if old not in content:
            raise ValueError(f"content does not include {old}.")
        content = content.replace(old, new)
    with open(path, "w") as f:
        f.write(content)


def find_const_in_file(search_string, path=config_path):
    with open(path, "r") as f:
        content = f.read()
        for line in content.splitlines():
            if search_string in line:
                numbers = re.findall(r'\d+', line)
                const = numbers[0]
    if not const:
        raise ValueError("No const was found.")
    return const


def update_tile_size(old_tile_size, new_tile_size):
    # updating config
    new_conf_line = f"static constexpr uint TILE_SIZE{{{new_tile_size}}};"
    old_conf_line = f"static constexpr uint TILE_SIZE{{{old_tile_size}}};"
    #print(f"updating TILE_SIZE in {config_path}...")
    repl(old=old_conf_line, new=new_conf_line, path=config_path)

    # updating slang shaders
    new_slang_line     = f"[numthreads({new_tile_size},1,1)]"
    old_slang_line     = f"[numthreads({old_tile_size},1,1)]"
    new_slang_blelloch = f"groupshared uint tile[{new_tile_size}];"
    old_slang_blelloch = f"groupshared uint tile[{old_tile_size}];"

    for key in shaders.keys():
        for file in shaders[key]:
            path = slang_path + key + "/shaders/" + file + slang_filetype
            #print(f"updating TILE_SIZE in {path}...")
            repl(old=old_slang_line, new=new_slang_line, path=path)
            if file == "KernelBlellochScan":
                repl(old=old_slang_blelloch, new=new_slang_blelloch, path=path)
    
    # updating vulkan shaders
    new_vulkan_line     = f"layout(local_size_x = {new_tile_size}, local_size_y = 1, local_size_z = 1) in;"
    old_vulkan_line     = f"layout(local_size_x = {old_tile_size}, local_size_y = 1, local_size_z = 1) in;"
    new_vulkan_blelloch = f"shared uint tile[{new_tile_size}];"
    old_vulkan_blelloch = f"shared uint tile[{old_tile_size}];"

    for key in shaders.keys():
        for file in shaders[key]:
            path = vulkan_path + key + "/" + file + vulkan_filetype
            #print(f"updating TILE_SIZE in {path}...")
            repl(old=old_vulkan_line, new=new_vulkan_line, path=path)
            if file == "KernelBlellochScan":
                repl(old=old_vulkan_blelloch, new=new_vulkan_blelloch, path=path)


def file_exists(filepath):
    return os.path.exists(filepath)


def search_TILE_SIZE(executables, incl, search_range):
    dataframes = {}
    for method in incl:
        if not file_exists(executables[method]):
            raise FileNotFoundError(f"The executable for {method} was not found in the build directory. Reconfigure the project to build the correct files!")
        dataframes[method] = []
    
    old_tile_size = int(find_const_in_file("TILE_SIZE", config_path))
    for TILE_SIZE in search_range:
        update_tile_size(old_tile_size=old_tile_size, new_tile_size=TILE_SIZE)
        run_cmake()
        for method in incl:
            run_benchmark(executables=executables, method=method)
            dataframes[method].append(read_benchmark(append=TILE_SIZE, name_append="TILE_SIZE"))
        old_tile_size = TILE_SIZE
    
    return dataframes


def search_h(executables, incl, search_range):
    dataframes = {}
    for method in incl:
        if not file_exists(executables[method]):
            raise FileNotFoundError(f"The executable for {method} was not found in the build directory. Reconfigure the project to build the correct files!")
        dataframes[method] = []
    
    old_h = float(find_const_in_file(search_string="FloatType h", path=config_path))
    for h in search_range:
        new_conf_line = f"FloatType h{{{h}}};"
        old_conf_line = f"FloatType h{{{old_h}}};"
        repl(old=old_conf_line, new=new_conf_line, path=config_path)
        run_cmake()
        for method in incl:
            run_benchmark(executables=executables, method=method)
            dataframes[method].append(read_benchmark(append=h, name_append="h"))
        old_h = h
    
    return dataframes


def search_influenceRadius(executables, incl, search_range):
    dataframes = {}
    for method in incl:
        if not file_exists(executables[method]):
            raise FileNotFoundError(f"The executable for {method} was not found in the build directory. Reconfigure the project to build the correct files!")
        dataframes[method] = []
    
    old_inflRad = float(find_const_in_file(search_string="FloatType influenceRadius", path=config_path))
    for inflRad in search_range:
        new_conf_line = f"FloatType influenceRadius{{{inflRad}}};"
        old_conf_line = f"FloatType influenceRadius{{{old_inflRad}}};"
        repl(old=old_conf_line, new=new_conf_line, path=config_path)
        run_cmake()
        for method in incl:
            run_benchmark(executables=executables, method=method)
            dataframes[method].append(read_benchmark(append=inflRad, name_append="Influence Radius"))
        old_inflRad = inflRad
    
    return dataframes


def search_interval_neighbor_search(executables, incl, search_range):
    for method in incl:
        if not file_exists(executables[method]):
            raise FileNotFoundError(f"The executable for {method} was not found in the build directory. Reconfigure the project to build the correct files!")

    old_interval = uint(find_const_in_file(search_string="interval_neighbor_search", path=config_path))
    highest_pass = -1
    for interval_neighbor_search in search_range:
        new_conf_line = f"static constexpr uint interval_neighbor_search{{{interval_neighbor_search}}};"
        old_conf_line = f"static constexpr uint interval_neighbor_search{{{old_interval}}};"
        repl(old=old_conf_line, new=new_conf_line, path=config_path)
        run_cmake()
        passing = True
        for method in incl:
            failures = run_test(executables=executables, method=method)
            passing = passing and failures == 0
        old_interval = interval_neighbor_search
        if passing:
            highest_pass = interval_neighbor_search
        else:
            if highest_pass == -1:
                raise ValueError(f"At least one test failed with the lowest interval of {interval_neighbor_search}")
            return highest_pass
    
    return highest_pass


def plot_summary(dataframes, groupby, xlabel, title):
    for key, dfs in dataframes.items():
        df = pd.concat(dfs, ignore_index=True)
        # small N are dominated by overhead and not representative
        large_n = df[df['N'] >= 1000]
        summary_large = (
            large_n.groupby(groupby)['real_time_s']
                .mean()
        )
        plt.plot(
            summary_large.index,
            summary_large.values,
            marker='o',
            label=key
        )
    plt.title(title)
    plt.xlabel(xlabel=xlabel)
    plt.ylabel("Mean real_time_s (N ≥ 1000)")
    plt.xscale("log", base=2)
    plt.yscale("log")
    plt.legend()
    plt.grid(True, which="both", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.show()


def run_benchmark(executables, method):
    print("Now running " + method + " benchmark...")
    executable_path = executables[method]
    command = [executable_path, f"--benchmark_out={temp_result_path}", "--benchmark_out_format=json"]
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    print(result.stdout, "\n")


def run_test(executables, method):
    print("Now running " + method + " test...")
    test_path = executables[method]
    command = [test_path, f"--gtest_output=json:{temp_result_path}"]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    print(result.stdout, "\n")
    
    with open(temp_result_path, 'r') as f:
        data = json.load(f)
    failures = data["failures"]
    return failures


def read_benchmark(append=0, name_append=""):
    with open(temp_result_path, 'r') as f:
        data = json.load(f)
    benchmarks = data["benchmarks"]
    
    df = pd.DataFrame(benchmarks)
    df = df[df["name"].str.endswith("_mean")]
    df = df.drop(columns=["family_index", "per_family_instance_index", "run_name", 
        "run_type", "repetitions", "repetition_index", 
        "threads", "iterations", "cpu_coefficient", 
        "real_coefficient", "big_o", "rms", 
        "aggregate_name", "aggregate_unit", "force_update", 
        "neighbor_search", "position_update_reset", "velocity_update"])
    df["real_time_s"] = df["real_time"] / 1e9
    df["cpu_time_s"]  = df["cpu_time"]  / 1e9
    df

    df = df[df["name"].str.contains(r"_mean$", regex=True)]
    extracted = df["name"].str.extract(r"/(\d+)(?:/|_)")[0]

    if extracted.isna().any():
        bad = df.loc[extracted.isna(), "name"].unique()
        raise ValueError(f"Could not extract N from:\n{bad}")

    df["N"] = extracted.astype(int)
    if name_append != "":
        df[name_append] = append
    return df


def run_cmake():
    try:
        build_cmd = r'''
        if command -v module >/dev/null 2>&1; then
            module load VulkanSDK
        else
            export LD_LIBRARY_PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/25.9/cuda/13.0/targets/x86_64-linux/lib:$LD_LIBRARY_PATH
        fi

        cmake --build ../build
        '''

        cmake = subprocess.run(["bash", "-lc", build_cmd], capture_output=True, text=True, check=True)
        print(cmake.stdout, "\n")
    except subprocess.CalledProcessError as e:
        print(e.stdout)
        print(e.stderr)
        raise
