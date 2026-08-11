# Performance Portability Benchmarking

This repository contains four benchmark problems of increasing complexity, each implemented with
**14 GPU programming paradigms** plus a sequential C++ baseline. Every implementation solves the
same problem with the same algorithm and the same features, which makes the paradigms directly
comparable in terms of both **runtime** and **code complexity**.

It is the *benchmark* half of a performance-portability study; the *tooling* half — running the
benchmarks, computing application efficiency $e_A$ and performance portability $\Phi$, measuring
code complexity and rendering the plots — lives in
[**ppbcc**](https://github.com/schuhmaj/performance-portability-code-complexity).
The software setup helper resides in [**software-bootstrapper**](https://github.com/schuhmaj/software-bootstrapper).

## The Workflow

<p align="center">
  <img src="docs/workflow.png" width="100%">
  <br>
  <em>
    From a bare machine to performance portability and code complexity.<br>
  </em>
</p>

Three repositories play together:

| Repository | Role |
|---|---|
| **performance-portability-benchmark** (this one) | The implementations, the CMake build system, the correctness tests, and the recorded raw measurements under [`results/`](results) |
| [**performance-portability-code-complexity**](https://github.com/schuhmaj/performance-portability-code-complexity) (`ppbcc`) | Runs the benchmarks, consolidates their reports, computes $e_A$, $\Phi$ and the code-complexity metrics, and draws every chart. Its code-complexity analysis is stand-alone and works on any C++/GPU codebase |
| [**software-bootstrapper**](https://github.com/schuhmaj/software-bootstrapper) | Provisions the toolchains and libraries (LLVM with OpenMP offload, Vulkan SDK, CMake/Ninja, Kokkos, Boost, …) into a prefix of your choice, optionally with Lmod/Tcl module files |

## Benchmark Problems

| Problem | CMake option | Description |
|---|---|---|
| Vector Addition | `PPB_ENABLE_VectorAddition` | Element-wise $c = a + b$; the memory-bound baseline |
| Matrix Multiplication | `PPB_ENABLE_MatrixMultiplication` | Dense $C = A \times B$; the compute-bound baseline |
| N-Body Simulation | `PPB_ENABLE_NBodySimulation` | Pairwise Lennard-Jones forces, in a naive $\mathcal{O}(N^2)$, a linked-cells, and a Verlet-lists variant |
| Polyhedral Gravity Model | `PPB_ENABLE_PolyhedralGravity` | Full gravitational tensor of a homogeneous polyhedron following Tsoulis' line-integral approach; irregular meshes, reductions, and transcendental operations |

The latter two are not synthetic: they are motivated by two CPU-only scientific codebases that are
to be migrated to GPUs.

- The N-body benchmark stands in for the node-level particle-simulation library
  [**AutoPas**](https://github.com/AutoPas/AutoPas), which dynamically selects the fastest
  algorithm and parameters for a particle system. The N-body implementations here are additionally
  verified against results produced by AutoPas.
- The polyhedral gravity benchmark is derived from ESA's
  [**polyhedral-gravity-model**](https://github.com/esa/polyhedral-gravity-model). An earlier
  parallel GPU exploration of it is available at
  [rho2/polyhedral-gravity-parallel](https://github.com/rho2/polyhedral-gravity-parallel).
  The meshes it evaluates are real small-body shape models; fetch them once with
  [`data/download_models.sh`](data/download_models.sh).

## Paradigms

CUDA · HIP · SYCL (AdaptiveCpp / DPC++) · Kokkos · RAJA · Alpaka · OpenMP (target offload) ·
OpenACC · Stdpar (ISO C++ parallel algorithms) · OpenCL · Boost.Compute · Vulkan (Kompute) ·
Slang-Vulkan · Slang-CUDA

Vector addition additionally has a Metal implementation for macOS, which is outside the study's
six-platform matrix. Not every paradigm is available on every platform — that is precisely what the
study measures.

## Requirements

You need the **vendor toolchain of your target GPU**, a host C/C++ compiler, and Python $\geq$ 3.12.
Everything else can be provisioned by the
[software-bootstrapper](https://github.com/schuhmaj/software-bootstrapper) or is fetched
automatically by CMake.

### Vendor toolchains

| Vendor | Install | Notes |
|---|---|---|
| NVIDIA | [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) | Required for CUDA, Thrust and Slang-CUDA |
| NVIDIA | [HPC SDK (NVHPC)](https://developer.nvidia.com/hpc-sdk) | The only compiler supporting **OpenACC**; also provides `nvc++ --stdpar=gpu` |
| AMD | [ROCm](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/index.html) | Provides HIP and `amdclang++`, which drives Stdpar on AMD |
| INTEL | [oneAPI Base Toolkit](https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html) | Provides `icx`/`icpx` with SYCL. Source it via `source /opt/intel/oneapi/setvars.sh` |
| INTEL | SYCL plugins for [NVIDIA](https://developer.codeplay.com/products/oneapi/nvidia/download/) or [AMD](https://developer.codeplay.com/products/oneapi/amd/home/) GPUs | Optional; install the CUDA Toolkit *before* the NVIDIA plugin |

### Toolchains and libraries via the software-bootstrapper

```bash
git clone https://github.com/schuhmaj/software-bootstrapper.git && cd software-bootstrapper
pip install -r requirements.txt

# $MOD_DIR is optional - pass it to also get Lmod/Tcl module files
./installer_llvm.py   $INSTALL_PREFIX $MOD_DIR --llvm-targets "NVPTX;host"  # LLVM + OpenMP offload
./installer_cmake.py  $INSTALL_PREFIX $MOD_DIR                              # CMake + Ninja
./installer_vulkan.py $INSTALL_PREFIX $MOD_DIR                              # Vulkan SDK (bundles Slang)
./installer.py        $INSTALL_PREFIX $MOD_DIR                              # Kokkos, Boost, googletest, ...
```

Afterwards either `module use $MOD_DIR && module load llvm VulkanSDK cmake ninja`, or
`export CMAKE_PREFIX_PATH=$INSTALL_PREFIX` and add the binaries to your `PATH`.

> [!TIP]
> `installer.py` is optional. Kokkos, RAJA, Alpaka, AdaptiveCpp, Boost, GoogleTest and Google
> Benchmark are all fetched by CMake's `FetchContent` if they are not found. Installing the large
> ones (AdaptiveCpp, Boost) once is still much faster than rebuilding them per build
> directory.

### Optional

- **OpenCL** usually ships with the vendor driver. If you need a standalone runtime, use the
  [Portable Computing Language (PoCL)](https://github.com/pocl/pocl).
- **AdaptiveCpp** can be built by CMake, but because of its compilation time it is worth installing
  it into your userspace with `installer.py`.

## Build

### Using the CMake presets — the recommended path

[`CMakePresets.json`](CMakePresets.json) is the fastest way to a complete build: each preset picks
the compilers for a platform and switches on exactly the paradigms that platform supports, so a
full set of executables is one command away.

```bash
cmake --preset cuda-llvm     # configure into build-cuda-llvm/
cd build-cuda-llvm
cmake --build .              # build every enabled target
ctest                        # verify every implementation
```

| Preset | Compilers | Enables / disables |
|---|---|---|
| `cpu` | default | Only the sequential C++ baselines |
| `abstract-base` | default | Base preset with **all** paradigms and problems on; inherit from it for a platform we do not cover |
| `cuda-llvm` | `clang`/`clang++`, `nvcc` | Everything except OpenACC and Stdpar |
| `cuda-nvhpc` | `nvc`/`nvc++`, `nvcc` | Adds OpenACC and Stdpar; drops OpenMP offload, Vulkan and Slang-Vulkan |
| `cuda-gcc` | `gcc`/`g++`, `nvcc` | Drops OpenMP offload, OpenACC and Stdpar |
| `rocm-amdclang-cdna` | `amdclang`/`amdclang++` | Compute GPUs (MI210, …). Drops OpenACC, Slang-CUDA, Vulkan and Slang-Vulkan. Run the Stdpar binaries with `HSA_XNACK=1` |
| `rocm-llvm-cdna` | `clang`/`clang++` | Like the above, but additionally drops Stdpar |
| `rocm-amdclang-rdna` | `amdclang`/`amdclang++` | Consumer GPUs. Drops OpenACC and Slang-CUDA; Stdpar uses allocation interposition since consumer cards lack HMM/XNACK |
| `rocm-llvm-rdna` | `clang`/`clang++` | Like the above, but drops Stdpar (needs an upstream LLVM matching the ROCm headers, LLVM $\geq$ 21 for ROCm 7.x) |
| `intel` | `icx`/`icpx` | Drops OpenACC, Slang and Vulkan |
| `macos` | AppleClang | Enables Metal; drops OpenACC, Slang-CUDA and Stdpar |

Each preset configures into its own `build-<preset>/` directory, so several toolchains can coexist
in one checkout. CUDA and HIP are enabled automatically through CMake's language detection —
`PPB_ENABLE_CUDA` and `PPB_ENABLE_HIP` never need to be set by hand.

To deviate from a preset, override single options on the command line, e.g.
`cmake --preset cuda-llvm -DPPB_ENABLE_Alpaka=OFF`.

### Manual configuration

```bash
mkdir build && cd build
# Set up your environment first: module load <compiler>, source setvars.sh, conda activate ...
cmake .. -G Ninja
ccmake ..                             # select the options interactively
cmake --build .                       # build all configured targets
cmake --build . --target vec_acpp     # ... or a single one
```

### Build Options

By default **all** paradigm targets are disabled — one has to enable a technology explicitly (or
use a preset, which does it for you).

**General**

| Option | Default | Description |
|---|---|---|
| `PPB_LOGGING_LEVEL` | `INFO` | `TRACE`, `DEBUG`, `INFO`, `WARN`, `ERROR`, `CRITICAL` or `OFF` |
| `PPB_FloatType` | `32` | Floating-point precision, `32` or `64` bit |
| `PPB_ENABLE_OnlyKernelRuntime` | `OFF` | Measure only the kernel runtime instead of the wall-clock time including transfers |
| `PPB_ENABLE_Testing` | `ON` | Build the GoogleTest suite, run it with `ctest` |

**Paradigms**

| Option | Default | Toolchain                                                                       |
|---|---|---------------------------------------------------------------------------------|
| `PPB_ENABLE_CUDA` | auto | CUDA; auto-detected via `check_language(CUDA)`                                  |
| `PPB_ENABLE_HIP` | auto | HIP; auto-detected via `check_language(HIP)`                                    |
| `PPB_ENABLE_Kokkos` | `OFF` | LLVM, NVHPC, amdclang, icpx                                                     |
| `PPB_ENABLE_Raja` | `OFF` | LLVM, NVHPC, amdclang, icpx                                                     |
| `PPB_ENABLE_Alpaka` | `OFF` | LLVM, NVHPC, amdclang, icpx                                                     |
| `PPB_ENABLE_AdaptiveCpp` | `OFF` | SYCL. Uses AdaptiveCpp, except with `icpx`, which compiles the sources natively |
| `PPB_ENABLE_OpenMP` | `OFF` | OpenMP target offload; LLVM or `icpx`                                           |
| `PPB_ENABLE_OpenACC` | `OFF` | **NVHPC only** — configuring with any other compiler is a hard error            |
| `PPB_ENABLE_Stdpar` | `OFF` | ISO C++ parallel algorithms; `nvc++`, `amdclang++`, `icpx`, or `acpp`           |
| `PPB_ENABLE_OpenCL` | `OFF` | LLVM, NVHPC; the runtime usually comes with the vendor driver                   |
| `PPB_ENABLE_Boost` | `OFF` | Boost.Compute — requires `PPB_ENABLE_OpenCL`                                    |
| `PPB_ENABLE_Vulkan` | `OFF` | Vulkan-Kompute; needs the Vulkan SDK                                            |
| `PPB_ENABLE_Slang_Vulkan` | `OFF` | Slang shaders on Vulkan; needs `slangc` from the Vulkan SDK                     |
| `PPB_ENABLE_Slang_Cuda` | `OFF` | Slang shaders on CUDA; needs `slangc` and the CUDA Toolkit                      |
| `PPB_ENABLE_Metal` | `OFF` | Metal, AppleClang on macOS only  (experimental, not maintained)                 |

**Benchmark problems**

| Option | Default | Description |
|---|---|---|
| `PPB_ENABLE_VectorAddition` | `ON` | Vector addition |
| `PPB_ENABLE_MatrixMultiplication` | `ON` | Matrix multiplication |
| `PPB_ENABLE_NBodySimulation` | `OFF` | N-body simulation; gates the three variants below, each of which is enabled separately |
| ↳ `PPB_ENABLE_NBodySimulation_naive` | `OFF` | Naive $\mathcal{O}(N^2)$ all-pairs |
| ↳ `PPB_ENABLE_NBodySimulation_cells` | `OFF` | Linked cells |
| ↳ `PPB_ENABLE_NBodySimulation_verlet` | `OFF` | Verlet lists |
| `PPB_ENABLE_PolyhedralGravity` | `OFF` | Polyhedral gravity model; needs the meshes from `data/download_models.sh` |

**Stdpar fine-tuning** (see [`cmake/stdpar_offload.cmake`](cmake/stdpar_offload.cmake))

| Option | Default | Description |
|---|---|---|
| `PPB_Stdpar_Offload_Arch` | *(empty)* | GPU architecture, e.g. `cc80` (NVHPC) or `gfx90a` (AMD); empty uses the toolchain default |
| `PPB_Stdpar_Hip_InterposeAlloc` | `OFF` | Adds `--hipstdpar-interpose-alloc` for AMD systems without HMM/XNACK |
| `PPB_Stdpar_UseAcpp` | `OFF` | Offload via AdaptiveCpp instead of the vendor toolchain; requires `CMAKE_CXX_COMPILER=acpp` |
| `PPB_Stdpar_Intel_OffloadTarget` | `gpu` | `-fsycl-pstl-offload` target; set to `cpu` to isolate kernel bugs from driver bugs |

## Execution

Every target is a Google Benchmark executable and can be run on its own:

```bash
./src/vectorAdditon/acpp/vec_acpp --help
```

Running all of them and collecting their JSON reports into one tidy CSV is what `ppbcc` is for:

```bash
git clone https://github.com/schuhmaj/performance-portability-code-complexity.git
pip install ./performance-portability-code-complexity

cd build-cuda-llvm
ppbcc benchmark -p src -H "NVIDIA GH200" \
  -r "vec_.*" "matMul_.*" "nbody_.*" "polyhedral_.*" -x ".*_cpp" --dry-run
```

Drop `--dry-run` to actually run them, and add `-o "Results_NVIDIA_GH200"` to name the CSV.
`ctest` runs the correctness suite in the same build directory.

## Results and Reproduction

The recorded raw measurements of the six GPUs of the study (NVIDIA RTX 3080/4060/5080 and GH200,
AMD Instinct MI210, INTEL Data Center GPU Max 1550) are archived under [`results/`](results),
alongside CPU runs. [`results/README.md`](results/README.md) documents the
exact commands that turn them into the published CSVs, code-complexity tables and plots.

For the analysis options themselves, see the
[ppbcc documentation](https://schuhmaj.github.io/performance-portability-code-complexity/) and its
[example gallery](https://schuhmaj.github.io/performance-portability-code-complexity/usage/plots.html).

## Cluster Notes

[`tools/`](tools) collects node-specific helpers: a `Dockerfile` and, under `tools/lrz/`, the Slurm
batch scripts and setup scripts for the LRZ systems. The `software-bootstrapper` repository carries
the matching
[BEAST-specific notes and configs](https://github.com/schuhmaj/software-bootstrapper/tree/main/tools/lrz-beast),
including the `libstdc++`/`icpx` how-tos/ hacks.

## Credits

The performance-portability metrics and the Cascade/Navchart layouts implemented in `ppbcc` are
inspired by the [P3 Analysis Library](https://github.com/P3HPC/p3-analysis-library) by Pennycook et al.
