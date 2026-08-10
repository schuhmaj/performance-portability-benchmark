# Benchmarks to Re-Run

This file lists the benchmarks that have to be re-measured after the
"comparable implementations" pass (workgroup/tile-size equalization, index-order
fixes and timing-bug fixes). Only the executables listed here changed —
everything else can keep its existing `results/**/*.json`.

## 1. What changed, and therefore what must be re-run

| Problem | Executables to re-run | Reason |
| --- | --- | --- |
| MatrixMultiplication | `matMul_acpp`, `matMul_alpaka`, `matMul_vulkan`, `matMul_slang_vulkan`, `matMul_slang_cuda` | tile size 32×32 → 16×16; AdaptiveCpp/Alpaka row index moved onto the fastest-varying dimension |
| NBody (direct sum) | `nbody_cuda`, `nbody_hip`, `nbody_ocl`, `nbody_boost`, `nbody_acpp`, `nbody_alpaka` | CUDA/HIP force-timing scale bug (`* 16` → `* 1e6`); OpenCL read profiling counters without waiting on the event; local size 32/16 → 256; OpenCL and Alpaka force kernels now accumulate in registers |
| PolyhedralGravity | `polyhedral_cuda`, `polyhedral_hip`, `polyhedral_ocl`, `polyhedral_boost`, `polyhedral_vulkan`, `polyhedral_slang_vulkan`, `polyhedral_slang_cuda` | eval workgroup 64/32/16 → 256 (plus Vulkan/Slang reduction restructure and a Vulkan reduction indexing fix); CUDA/HIP dead per-call host allocations removed |
| VectorAddition | *(nothing)* | only an added `<iostream>` include in `Impl_SlangVulkan.cpp`; no behavioural change |

`nbody_vulkan_*` / `nbody_slang_*` (naive, cells, verlet) were **not** touched and
do not need re-running. `matMul_kokkos` keeps its explicit `{16, 16}` MDRange tile
and is unchanged.

### Effect measured locally (RTX 5080, `cuda-llvm`)

Spot-check against the stored `results/nvidia-rtx5080/*.json`. `matMul_ocl` and
`nbody_kokkos` were re-measured unchanged as controls.

| Benchmark | metric | before | after | change |
| --- | --- | ---: | ---: | ---: |
| `matMul_acpp` @16384 | kernel | 18.95 s | 3.30 s | **5.75x faster** |
| `matMul_alpaka` @16384 | kernel | 61.33 s | 18.64 s | **3.29x faster** |
| `matMul_slang_cuda` @16384 | kernel | 15.48 s | 9.83 s | 1.58x faster |
| `matMul_slang_vulkan` @16384 | kernel | 2.840 s | 2.608 s | 1.09x faster |
| `matMul_vulkan` @16384 | kernel | 2.824 s | 2.624 s | 1.08x faster |
| `matMul_ocl` @16384 *(control)* | kernel | 2.536 s | 2.545 s | 1.00x |
| `nbody_alpaka` @100000 | wall | 190.4 s | 34.8 s | **5.47x faster** |
| `nbody_boost` @100000 | wall | 75.9 s | 14.0 s | **5.43x faster** |
| `nbody_ocl` @100000 | wall | 75.9 s | 14.1 s | **5.37x faster** |
| `nbody_acpp` @100000 | wall | 24.67 s | 18.89 s | 1.31x faster |
| `nbody_kokkos` @100000 *(control)* | wall | 25.30 s | 25.69 s | 0.98x |
| `nbody_cuda` @100000 | **force** | 0.00038 s | 24.19 s | timing bug fixed (was 62500x too small) |
| `nbody_hip` @100000 | **force** | 0.00051 s | 32.52 s | timing bug fixed |
| `nbody_ocl` @100000 | **force** | 1.8e13 s (garbage) | 14.08 s | timing bug fixed |

Note which column each problem's plots use: **MatrixMultiplication and VecAdd are
scored on `Kernel Time`, NBody and PolyhedralGravity on `Wall Clock Time`.** The
CUDA/HIP/OpenCL force-timing bugs therefore corrupted the per-phase
*Force Update Time* column only — they did not distort the published
application-efficiency plots. The block-size and index-order fixes above *do*
change the plotted numbers.

## 2. Environment

Both the toolchain modules **and** the conda env are needed. Activating only the
conda env makes the binaries fail with exit code 127 (missing shared libraries),
because the module `LD_LIBRARY_PATH` is not set.

```bash
cd ~/Programming/performance-portability-benchmark

# toolchain (pick the one matching the preset)
source ~/load_all_llvm.sh      # clang/LLVM   -> cuda-llvm, rocm-llvm-*
# source ~/load_all_nvhpc.sh   # NVHPC        -> cuda-nvhpc (OpenACC/Stdpar)

# ppbcc
source ~/miniforge3/etc/profile.d/conda.sh
conda activate standard-math
```

## 3. Build

```bash
# NVIDIA + LLVM (the preset used for CUDA/Vulkan/Slang/AdaptiveCpp/Kokkos/RAJA/Alpaka)
cmake -U CMAKE_COLOR_DIAGNOSTICS --preset cuda-llvm
cmake --build build-cuda-llvm -j $(nproc)
```

The `-U CMAKE_COLOR_DIAGNOSTICS` is required whenever the build directory was
last configured from CLion: the stale cache entry injects `-fcolor-diagnostics`
into every CXX target and the Kokkos `nvcc_wrapper` (g++ host compiler) rejects it.

Other platforms use the corresponding preset from `CMakePresets.json`:

```bash
cmake --preset cuda-nvhpc          && cmake --build build-cuda-nvhpc          -j $(nproc)  # OpenACC/Stdpar host
cmake --preset rocm-amdclang-cdna  && cmake --build build-rocm-amdclang-cdna  -j $(nproc)  # AMD MI210
cmake --preset intel               && cmake --build build-intel               -j $(nproc)  # Intel Max 1550
```

## 4. Re-run the benchmarks

`ppbcc benchmark` searches `--build-dir` recursively for executables matching
`--regex`, runs them, and writes one Google-Benchmark JSON per binary plus a
consolidated CSV. Run it from the repository root and point `-b` at the build
folder so the search does not pick up binaries from a second build directory.

### NVIDIA (RTX 5080 / RTX 3080 / RTX 4060 / GH200)

```bash
BUILD=build-cuda-llvm
HW="NVIDIA RTX5080"        # adjust per machine

ppbcc benchmark -b $BUILD -H "$HW" -o "Redo_MatMul" \
  -r ".*matMul_(acpp|alpaka|vulkan|slang_vulkan|slang_cuda)$"

ppbcc benchmark -b $BUILD -H "$HW" -o "Redo_NBody" \
  -r ".*nbody_(cuda|hip|ocl|boost|acpp|alpaka)$"

ppbcc benchmark -b $BUILD -H "$HW" -o "Redo_Polyhedral" \
  -r ".*polyhedral_(cuda|hip|ocl|boost|vulkan|slang_vulkan|slang_cuda)$"
```

Or everything in one go:

```bash
ppbcc benchmark -b build-cuda-llvm -H "NVIDIA RTX5080" -o "Redo_All" \
  -r ".*matMul_(acpp|alpaka|vulkan|slang_vulkan|slang_cuda)$" \
     ".*nbody_(cuda|hip|ocl|boost|acpp|alpaka)$" \
     ".*polyhedral_(cuda|hip|ocl|boost|vulkan|slang_vulkan|slang_cuda)$"
```

`ppbcc benchmark` skips a binary when its report already exists; add `-f` to
force a re-measurement.

### AMD MI210 (no Vulkan / Slang / CUDA in this preset)

```bash
ppbcc benchmark -b build-rocm-amdclang-cdna -H "AMD Instinct MI210" -o "Redo_All" -f \
  -r ".*matMul_(acpp|alpaka)$" \
     ".*nbody_(hip|ocl|boost|acpp|alpaka)$" \
     ".*polyhedral_(hip|ocl|boost)$"
```

### Intel Data Center GPU Max 1550 (no Vulkan / Slang / CUDA / HIP)

```bash
ppbcc benchmark -b build-intel -H "INTEL Data Center GPU Max 1550" -o "Redo_All" -f \
  -r ".*matMul_(acpp|alpaka)$" \
     ".*nbody_(ocl|boost|acpp|alpaka)$" \
     ".*polyhedral_(ocl|boost)$"
```

## 5. Copy the new reports into `results/`

`ppbcc benchmark` writes the JSON reports into the working directory it was
started from. Copy only the regenerated ones over the stored results:

```bash
# adjust the destination to the machine that was measured
DEST=results/nvidia-rtx5080     # results/nvidia-rtx3080 | results/nvidia-rtx4060
                                # results/nvidia-gh200   | results/amd-mi210
                                # results/intel-data_center_gpu_max_1550

for f in matMul_acpp matMul_alpaka matMul_vulkan \
         matMul_slang_vulkan matMul_slang_cuda \
         nbody_cuda nbody_hip nbody_ocl nbody_boost nbody_acpp nbody_alpaka \
         polyhedral_cuda polyhedral_hip polyhedral_ocl polyhedral_boost \
         polyhedral_vulkan polyhedral_slang_vulkan polyhedral_slang_cuda; do
    [ -f "$f.json" ] && cp -v "$f.json" "$DEST/$f.json"
done
```

## 6. Regenerate the aggregated CSVs, complexity data and plots

The per-implementation source changed, so the code-complexity aggregate has to be
rebuilt as well (SLOC and Halstead operand/operator counts shift slightly).

```bash
# 6a. code complexity
python scripts/generate_code_complexity.py

# 6b. consolidated per-hardware CSVs (from results/)
cd results
../scripts/benchmark.py -p ./nvidia-rtx3080 -r ".*\.json" --skip-benchmark -H "NVIDIA RTX3080" -o "Results_NVIDIA_RTX3080"
../scripts/benchmark.py -p ./nvidia-rtx4060 -r ".*\.json" --skip-benchmark -H "NVIDIA RTX4060" -o "Results_NVIDIA_RTX4060"
../scripts/benchmark.py -p ./nvidia-rtx5080 -r ".*\.json" --skip-benchmark -H "NVIDIA RTX5080" -o "Results_NVIDIA_RTX5080"
../scripts/benchmark.py -p ./nvidia-gh200   -r ".*\.json" --skip-benchmark -H "NVIDIA GH200"   -o "Results_NVIDIA_GH200"
../scripts/benchmark.py -p ./amd-mi210      -r ".*\.json" --skip-benchmark -H "AMD Instinct MI210" -o "Results_AMD_Instinct_MI210"
../scripts/benchmark.py -p ./intel-data_center_gpu_max_1550 -r ".*\.json" --skip-benchmark \
    -H "INTEL Data Center GPU Max 1550" -o "Results_INTEL_Data_Center_GPU_Max_1550"
```

Then regenerate the plots with the `ppbcc p3analysis` commands in
[`results/README.md`](results/README.md).

## 7. Sanity check before benchmarking

The correctness suites must stay green (they were green after these changes on
the RTX 5080 / `cuda-llvm` build):

```bash
ctest --test-dir build-cuda-llvm -R "MatrixMultiplicationTest" --output-on-failure
ctest --test-dir build-cuda-llvm -R "NBodyTest"                --output-on-failure
ctest --test-dir build-cuda-llvm -R "CubeGravityModelTest"     --output-on-failure
```

The polyhedral suites are slow for some paradigms (OpenCL/Boost ≈ 30 s each,
the rest well under 5 s); run a single binary directly with a timeout if needed:

```bash
timeout 900 build-cuda-llvm/test/polyhedralGravity/polyhedral_vulkan_test
```

Known pre-existing failures, unrelated to these changes:
`BySize/NBodyTest.ImplSlangCuda_Implementation` for the **cell_lists** and
**verlet_lists** variants (tiny numeric deviations vs. the reference). The
**naive** Slang-CUDA nbody passes.
