# Roofline models for every implementation

One roofline per problem covering **all** paradigms, including the four that reach the GPU without a
CUDA context. This file records what produced `matmul_roofline_NVIDIA_RTX5080.pdf` and
`polyhedral_roofline_NVIDIA_RTX5080.pdf`; the general `ppbcc profile` options and the LIKWID notes
are in [`README.md`](README.md) step 5.

RTX 5080 (GB203, sm_120), driver 610.57.04, Nsight Compute 2026.2.1, Nsight Systems 2026.1.3,
Nsight Graphics 2026.3.1, NVHPC 26.5, 2026-09-15.

| Problem | Profiling input (`PPB_PROFILING=ON`) | Precision |
|---|---|---|
| Matrix multiplication | 4096² | FP32 |
| Polyhedral gravity | `SHAPE_SFM_3M_v20180804.obj`, 3 145 728 faces | FP32 |

## Methodology

### The roofline

Every point is one implementation and one *region* — the benchmark names its work with NVTX ranges
from [`src/common/Marker.h`](../src/common/Marker.h): `matmul` (`matmul-cublas`/`matmul-naive` in
`matMul_cuda`), `init` (face normals and segment vectors) and `evaluate` (one gravity evaluation
including its reduction). For each point three quantities are determined, summed over every kernel
launch *k* inside the region:

| Symbol | Meaning | Unit |
|---|---|---|
| W = Σ W<sub>k</sub> | floating-point work | FLOP |
| Q = Σ Q<sub>k</sub> | data traffic between the GPU and its DRAM | byte |
| T = Σ T<sub>k</sub> | kernel time | s |

From these the two plotted coordinates follow, identically for every tool:

- **Arithmetic intensity** I = W / Q  [FLOP/byte]
- **Attained performance** P = W / T  [FLOP/s]

The roof is P<sub>max</sub>(I) = min(π, β · I), with the compute ceiling π [FLOP/s] and the DRAM
bandwidth ceiling β [byte/s]; they meet at the ridge point I = π / β. A point on the slanted part is
bandwidth-bound, a point far below both parts is latency-bound. The last column of the tables below
is P / (β · I), the fraction of the memory roof a point reaches.

What differs between the tools is only how W, Q, T and the ceilings are obtained.

### Nsight Compute (`ncu`): counted

`ncu` attaches to CUDA-context paradigms and reports per kernel launch:

- W<sub>k</sub> = N<sub>add</sub> + N<sub>mul</sub> + 2 · N<sub>fma</sub>, the executed SASS
  instructions of the build precision (`sm__sass_thread_inst_executed_op_f{add,mul,fma}_pred_on.sum`
  for FP32); a fused multiply-add is two FLOP.
- Q<sub>k</sub> = `dram__bytes.sum`, T<sub>k</sub> = `gpu__time_duration.sum`.
- π = 2 · `…ffma_pred_on.sum.peak_sustained` · f<sub>SM</sub> and
  β = `dram__bytes.sum.peak_sustained` · f<sub>DRAM</sub>: every `peak_sustained` metric is an
  achievable rate *per cycle*, scaled with the clock f (`*_cycles_elapsed.avg.per_second`) that unit
  actually ran at. The plot uses the largest value over all plotted rows: π = 57.4 TFLOP/s, β = 959–961 GB/s.

### Nsight Systems (`nsys`): sampled time and traffic, analytic work

OpenCL and Vulkan build their own contexts, which CUPTI (and therefore `ncu`) cannot see.
`nsys --gpu-metrics-devices` samples the same performance monitors device-wide at 100 kHz, as
percentages of peak: compute occupancy a(t) (`Compute Warps in Flight`) and DRAM read/write
bandwidth r(t), w(t).

- **Windows.** Inside each NVTX region the samples with a(t) > 1 % are grouped into windows; gaps
  below 0.5 ms are merged, windows shorter than 0.1 ms are dropped, and so is every window whose mean
  occupancy is below 0.7 × that of the busiest window in the region (Kompute moves its buffers with
  a compute shader at ~49 % occupancy, the dispatch runs at ~96 %).
- **Time and traffic.** T<sub>run</sub> = Σ window lengths and
  Q<sub>run</sub> = Σ<sub>windows</sub> ∫ (r(t) + w(t)) / 100 · β dt. β converts the percentages
  into bytes, so it must be supplied.
- **Two-point measurement.** Each binary runs with 20 iterations and with 1. With e the number of
  times the region was entered, T = (T<sub>20</sub> − T<sub>1</sub>) / (e<sub>20</sub> − e<sub>1</sub>)
  and Q likewise, which cancels lazy setup (pipeline compilation, buffer staging). A region entered
  equally often in both runs (`init`) is reported as measured.
- **Work.** The sampler has no instruction counts, so W is analytic:
  - matrix multiplication: W = 2 · M · N · K = 2 · 4096³ = 137 438 953 472 (one add and one
    multiply per inner-loop step); `ncu` counts exactly this for every CUDA-backed implementation
    except cuBLAS (+0.085 %).
  - polyhedral gravity: no closed form (branches per segment), so W is the **median of the counted
    `ncu` `evaluate` rows**: 1120.9 FLOP/face · 3 145 728 = 3.52607e9 FLOP. The ten counted
    implementations spread from 1000 (AdaptiveCpp, OpenMP) to 1360 FLOP/face (Slang-Cuda), a factor
    of 1.36, which the sampled points inherit as uncertainty.
  - `init` gets no work rule and is not plotted.
- **Ceilings.** π and β are the values `ncu` measured (5.74e13 FLOP/s, 9.592e11 byte/s).

The sampled P is therefore a measured *time* placed on the FLOP axis with borrowed work, not an
independent measurement of the arithmetic.

### Nsight Graphics (`ngfx`): traced cross-check

GPU Trace reads the monitors as counter sums over one bounded queue submission (submission 1, the
steady-state dispatch; a headless compute app has no frames). T = trace frame time,
Q = `dram__sectors.sum` (exported in bytes), W analytic as above, no region. It only cross-checks
the Vulkan rows and is not plotted.

### Accuracy

- **Kernel times** agree with independent timers to a few percent (table below), so the *ranking by
  time* is solid.
- **Byte counts** are good to about a factor of two: at 4096² a matrix (67 MB) exceeds the 64 MB L2,
  so whether a kernel re-reads DRAM depends on where the driver places the allocation, and
  `matMul_kokkos` measured 2.4 GB in one batch and 4.4 GB in others at identical time. Read every
  arithmetic intensity accordingly.

## Reproduce

```bash
# 0. Environment and profiling builds (one input, one iteration)
source ~/load_all_llvm.sh && conda activate standard-math
cmake --preset cuda-llvm-profiling && cmake --build build-cuda-llvm-profiling -j
source ~/load_all_nvhpc.sh && rm -rf build-cuda-nvhpc-profiling && cmake --preset cuda-nvhpc-profiling
cmake --build build-cuda-nvhpc-profiling -j --target matMul_acc matMul_stdpar polyhedral_acc polyhedral_stdpar
module purge && source ~/load_all_llvm.sh

# 1. Nsight Compute: the paradigms with a CUDA context
ppbcc profile -b build-cuda-llvm-profiling -p src -r "matMul_.*" "polyhedral_.*" -x ".*_cpp$" \
  -d ../profiling-rtx5080 -H "NVIDIA RTX5080" --no-csv --timeout 420 --force
ppbcc profile -b build-cuda-nvhpc-profiling -p src \
  -r "matMul_acc$" "matMul_stdpar$" "polyhedral_acc$" "polyhedral_stdpar$" \
  -d ../profiling-rtx5080 -H "NVIDIA RTX5080" --no-csv --timeout 420 --force
ppbcc profile -b . -d profiling-rtx5080 -r ".*" --skip-profile -H "NVIDIA RTX5080" \
  -o results/Profiling_NCU_NVIDIA_RTX5080

# 2. Analytic polyhedral work: median counted FLOP of the evaluate region
FLOP_POLY=$(python -c "import pandas as p; d=p.read_csv('results/Profiling_NCU_NVIDIA_RTX5080.csv'); d=d[d.Executable.str.startswith('polyhedral_') & (d.Region=='evaluate')]; print(d.groupby('Executable').FLOP.sum().median())")

# 3. Nsight Systems: the paradigms without a CUDA context
ppbcc profile --profiler nsys -b build-cuda-llvm-profiling -p src \
  -r "matMul_ocl$" "matMul_boost$" "matMul_vulkan$" "matMul_slang_vulkan$" \
     "polyhedral_ocl$" "polyhedral_boost$" "polyhedral_vulkan$" "polyhedral_slang_vulkan$" \
  -d ../profiling-nsys-rtx5080 -H "NVIDIA RTX5080" --timeout 900 -O iterations=20 --force \
  --peak-performance 5.74e13 --peak-bandwidth 9.592e11 \
  --analytic-flop "matMul_.*=137438953472" --analytic-flop "polyhedral_.*\[evaluate\]=$FLOP_POLY" \
  -o "$PWD/results/Profiling_NSYS_NVIDIA_RTX5080"

# 4. Nsight Graphics: cross-check on Vulkan
ppbcc profile --profiler ngfx -b build-cuda-llvm-profiling -p src \
  -r "matMul_vulkan$" "matMul_slang_vulkan$" "polyhedral_vulkan$" "polyhedral_slang_vulkan$" \
  -d ../profiling-ngfx-rtx5080 -H "NVIDIA RTX5080" --timeout 600 -O submit=1 --force \
  --peak-performance 5.74e13 --peak-bandwidth 9.592e11 \
  --analytic-flop "matMul_.*=137438953472" --analytic-flop "polyhedral_.*=$FLOP_POLY" \
  -o "$PWD/results/Profiling_NGFX_NVIDIA_RTX5080"

# 5. Plots: merge the counted and the sampled table
ppbcc profile --from-csv results/Profiling_NCU_NVIDIA_RTX5080.csv results/Profiling_NSYS_NVIDIA_RTX5080.csv \
  -r "matMul_" -H "NVIDIA RTX5080" --no-csv --label-points --region "^matmul" \
  --roofline results/matmul_roofline_NVIDIA_RTX5080.pdf
ppbcc profile --from-csv results/Profiling_NCU_NVIDIA_RTX5080.csv results/Profiling_NSYS_NVIDIA_RTX5080.csv \
  -r "polyhedral_" -H "NVIDIA RTX5080" --no-csv --label-points --region "^evaluate$" \
  --roofline results/polyhedral_roofline_NVIDIA_RTX5080.pdf
```

What the commands do:

- `source ~/load_all_llvm.sh && conda activate standard-math`: loads the toolchains and the Python
  environment with `ppbcc`. Both are needed, `ppbcc` launches the binaries.
- `cmake --preset cuda-llvm-profiling …`: builds every paradigm with `PPB_PROFILING=ON`, i.e. one input
  and one iteration. It also switches on the NVTX regions the tools attribute work to.
- `rm -rf build-cuda-nvhpc-profiling && cmake --preset cuda-nvhpc-profiling`: configures the NVHPC build
  in an empty directory. On an existing cache CMake keeps `/usr/local/cuda` as CUDA compiler, and
  `nvc++ -stdpar=gpu` then fails with "CUDA compiler and CUDA toolkit headers are incompatible".
- `cmake --build build-cuda-nvhpc-profiling … --target …`: builds only OpenACC and Stdpar, which need
  `nvc++`. Kokkos does not compile with it.
- `module purge && source ~/load_all_llvm.sh`: switches back to the LLVM toolchain for the profilers.
- `ppbcc profile -b build-cuda-llvm-profiling …`: runs every LLVM-built binary under `ncu` and writes
  `profiling-rtx5080/<exe>.ncu-rep`. OpenCL/Vulkan binaries produce no report, and `--force`
  replaces existing reports.
- `ppbcc profile -b build-cuda-nvhpc-profiling …`: does the same for the four NVHPC binaries into the
  same directory. `polyhedral_stdpar` exits with SIGABRT under `ncu` after its report is written;
  without a profiler it exits cleanly.
- `ppbcc profile -b . -d profiling-rtx5080 --skip-profile …`: re-reads all reports without running
  anything. It writes `Profiling_NCU_NVIDIA_RTX5080.csv`, one row per kernel launch with region.
- `FLOP_POLY=$(python -c …)`: sums the counted FLOP of each polyhedral implementation's `evaluate`
  region and takes the median. That is the analytic work of the sampled polyhedral points (3.52607e9).
- `ppbcc profile --profiler nsys …`: samples each OpenCL/Vulkan binary twice, at 20 and 1 iterations,
  and applies the window detection and two-point subtraction above. The ceilings are the `ncu`
  values, and `$PWD` keeps the CSV out of the build directory.
- `ppbcc profile --profiler ngfx …`: traces queue submission 1 of each Vulkan binary with GPU Trace.
  `polyhedral_slang_vulkan` yields no compute submission and has no row.
- `ppbcc profile --from-csv … --region "^matmul"`: merges the `ncu` and `nsys` tables and plots one
  point per implementation and region. `--region` keeps only the matrix-multiplication regions.
- `ppbcc profile --from-csv … --region "^evaluate$"`: the same for the polyhedral kernel. `init` stays
  in the CSVs but is not plotted.

## Results

### Matrix multiplication, 4096², FP32 — 137.44 GFLOP per implementation

| Executable | Region | Tool | ms | DRAM GB | AI [FLOP/byte] | TFLOP/s |
|---|---|---|---:|---:|---:|---:|
| `matMul_cuda` | matmul-cublas | ncu | 3.60 | 0.53 | 260.6 | 38.26 |
| `matMul_slang_vulkan` | matmul | nsys | 37.07 | 4.23 | 32.5 | 3.71 |
| `matMul_vulkan` | matmul | nsys | 37.07 | 4.22 | 32.6 | 3.71 |
| `matMul_boost` | matmul | nsys | 37.08 | 2.77 | 49.6 | 3.71 |
| `matMul_ocl` | matmul | nsys | 37.16 | 5.27 | 26.1 | 3.70 |
| `matMul_raja` | matmul | ncu | 39.36 | 2.41 | 57.1 | 3.49 |
| `matMul_kokkos` | matmul | ncu | 39.81 | 2.44 | 56.3 | 3.45 |
| `matMul_acpp` | matmul | ncu | 39.94 | 1.92 | 71.5 | 3.44 |
| `matMul_hip` | matmul | ncu | 42.96 | 1.84 | 74.9 | 3.20 |
| `matMul_cuda` | matmul-naive | ncu | 43.07 | 2.01 | 68.4 | 3.19 |
| `matMul_stdpar` | matmul | ncu | 58.62 | 5.49 | 25.0 | 2.35 |
| `matMul_acc` | matmul | ncu | 58.73 | 2.44 | 56.3 | 2.34 |
| `matMul_alpaka` | matmul | ncu | 59.27 | 1.12 | 122.4 | 2.32 |
| `matMul_omp` | matmul | ncu | 62.47 | 2.04 | 67.5 | 2.20 |
| `matMul_slang_cuda` | matmul | ncu | 83.16 | 1.98 | 69.3 | 1.65 |

Every hand-written kernel is **latency-bound**: 1.7–3.7 TFLOP/s against a 57.4 TFLOP/s compute roof
and a memory roof of ≥ 24 TFLOP/s at their intensities (≤ 16 % of it). The fourteen spread by 2.2× in
time. cuBLAS is the only tuned SGEMM — 12× faster than the binary's own naive kernel, at 67 % of the
compute roof.

### Polyhedral gravity, SHAPE_SFM_3M, FP32, `evaluate` — 3.526 GFLOP (counted rows: measured)

| Executable | Paradigm | Tool | ms | DRAM GB | AI [FLOP/byte] | TFLOP/s | memory roof |
|---|---|---|---:|---:|---:|---:|---:|
| `polyhedral_acc` | OpenACC | ncu | 0.53 | 0.46 | 7.6 | 6.66 | 91 % |
| `polyhedral_kokkos` | Kokkos | ncu | 0.53 | 0.36 | 9.7 | 6.60 | 71 % |
| `polyhedral_alpaka` | Alpaka | ncu | 0.61 | 0.47 | 7.4 | 5.69 | 80 % |
| `polyhedral_ocl` | OpenCL | nsys | 0.62 | 0.43 | 8.2 | 5.72 | 73 % |
| `polyhedral_boost` | Boost | nsys | 0.62 | 0.43 | 8.2 | 5.69 | 72 % |
| `polyhedral_acpp` | AdaptiveCpp | ncu | 0.63 | 0.38 | 8.3 | 5.02 | 63 % |
| `polyhedral_stdpar` | Stdpar | ncu | 0.67 | 0.58 | 6.1 | 5.31 | 90 % |
| `polyhedral_hip` | HIP | ncu | 0.87 | 0.71 | 5.0 | 4.06 | 85 % |
| `polyhedral_cuda` | CUDA | ncu | 0.88 | 0.71 | 4.9 | 4.03 | 85 % |
| `polyhedral_slang_cuda` | Slang-Cuda | ncu | 1.18 | 0.85 | 5.1 | 3.63 | 75 % |
| `polyhedral_raja` | RAJA | ncu | 1.25 | 0.39 | 9.4 | 2.94 | 32 % |
| `polyhedral_omp` | OpenMP | ncu | 12.31 | 0.66 | 4.7 | 0.26 | 6 % |
| `polyhedral_slang_vulkan` | Slang-Vulkan | nsys | 47.51 | 0.52 | 6.8 | 0.074 | 1 % |
| `polyhedral_vulkan` | Vulkan | nsys | 49.81 | 0.48 | 7.3 | 0.071 | 1 % |

At 4.7–9.7 FLOP/byte this kernel is **bandwidth-bound**: ten implementations reach 63–91 % of the
961 GB/s memory roof, and their ranking follows the DRAM traffic. RAJA and OpenMP (a second
reduction kernel) sit well below it. The two Vulkan implementations are ~95× off the leaders, in
agreement with their archived steady-state wall clocks.

`init` (not plotted) takes 0.69–0.85 ms on the `ncu` paradigms, 1.1 ms on OpenCL/Boost and 283 ms on
`polyhedral_vulkan`, almost all of it pipeline compilation and staging.

### Cross-checks

| Executable | `nsys` | `ngfx` | independent timer |
|---|---|---|---|
| `matMul_ocl` | 37.16 ms / 5.27 GB | — | 39.78 ms (`clGetEventProfilingInfo`)¹ |
| `matMul_boost` | 37.08 ms / 2.77 GB | — | 39.63 ms (same)¹ |
| `matMul_vulkan` | 37.07 ms / 4.22 GB | 45.91 ms / 5.43 GB | 39.11 ms (application timer)¹ |
| `matMul_slang_vulkan` | 37.07 ms / 4.23 GB | 45.90 ms / 5.39 GB | 41.04 ms (same)¹ |
| `polyhedral_ocl` | 0.616 ms / 0.43 GB | — | 0.598 ms (wall clock, `Results_NVIDIA_RTX5080.csv`) |
| `polyhedral_boost` | 0.620 ms / 0.43 GB | — | 0.601 ms (same) |
| `polyhedral_vulkan` | 49.81 ms / 0.48 GB | 52.58 ms / 0.12 GB | 47.29 ms (same) |
| `polyhedral_slang_vulkan` | 47.51 ms / 0.52 GB | no trace | 59.09 ms (same) |

¹ Measured with driver 610.43.02 on 2026-09-08.

Sampled times lie within 3–10 % of the independent timers; `polyhedral_slang_vulkan` is the outlier
at −20 %. GPU Trace adds 6–24 % tracing overhead to the time. Its bytes agree with `nsys` to ~30 %
on the matrix multiplication but are 4× lower on `polyhedral_vulkan`, which is why intensities are
read as good to a factor of two.

## Tools that were tried and are not used

- **PTXprofiler** counts instructions statically; non-unrolled loops count once, which misses the
  K-loop and the per-segment loop.
- **`nsys --trace=opencl`** no longer exists in CUDA 13.3; **`--trace=vulkan`** gives a timeline
  without counters.
- **Nsight Graphics per-dispatch attribution** needs frames; a headless compute app has none, hence
  the submit-index bracketing.
- **LIKWID NvMarker** and **HPCToolkit/TAU** are blind to OpenCL/Vulkan counters (CUPTI) or give
  only times; see [`README.md`](README.md) step 5.
