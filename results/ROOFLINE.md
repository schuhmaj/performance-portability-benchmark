# Roofline models for every implementation

One roofline per problem covering **all** paradigms — including the four that reach the GPU without
a CUDA context and are therefore invisible to Nsight Compute. The general reproduction recipe, and
the LIKWID/HPCToolkit/TAU notes, are in [`README.md`](README.md). This file records only what
produced `matmul_roofline_NVIDIA_RTX5080.pdf` and `polyhedral_roofline_NVIDIA_RTX5080.pdf`.

RTX 5080 (GB203, sm_120), driver 610.43.02, Nsight Compute 2026.2.1, Nsight Systems 2026.1.3,
Nsight Graphics 2026.3.1, 2026-09-08.

## Inputs

| Problem | Profiling input | Why |
|---|---|---|
| Matrix multiplication | 4096² (was 16384²) | one replay pass at 16384² has to snapshot several GB; 4096² is the largest size every backend profiles in seconds |
| Polyhedral gravity | `SHAPE_SFM_3M_v20180804.obj`, 3 145 728 faces (was Eros) | Eros' 24 576 faces run for a few µs — too short for a counter- or sample-based tool |

Both are selected in the source under `PPB_PROFILING` only, so the measuring build is unchanged:
[`MatrixMultiplication.h`](../src/matrixMultiplication/MatrixMultiplication.h),
[`polyhedralGravity/main.cpp`](../src/polyhedralGravity/main.cpp).

## Which tool measures what

| Paradigm | FLOP | DRAM bytes | Kernel time | Tool |
|---|---|---|---|---|
| CUDA, HIP, Kokkos, RAJA, Alpaka, AdaptiveCpp, OpenMP target, OpenACC, Stdpar, Slang-Cuda | counted SASS | counted | counted | **`ncu`** |
| OpenCL, Boost.Compute, Vulkan, Slang-Vulkan | analytic model | sampled | sampled | **`nsys`** GPU metrics |
| Vulkan, Slang-Vulkan (cross-check only) | analytic model | counted | traced | **`ngfx`** GPU Trace |

`ncu` and LIKWID read their counters through CUPTI, which needs a *current CUDA context*; OpenCL and
Vulkan build their own. `nsys --gpu-metrics-devices` programs the same performance monitors in
**time-based sampling** mode — device-wide, no context filter, no kernel boundaries — so it sees them
regardless of the API. What it gives up is attribution, and the benchmark buys that back: it brackets
the work it is about with NVTX ranges, which `nsys --trace=nvtx` records on the same clock as the
samples.

## Regions

A profiler names a kernel whatever the compiler called it, which for several paradigms is nothing
useful: AdaptiveCpp launches four kernels all called `__acpp_sscp_kernel`, Kokkos eleven called
`Kokkos::cuda_parallel_launch_local_memory`. [`src/common/Marker.h`](../src/common/Marker.h)
therefore names the phases itself, and `PPB_PROFILING=ON` switches those names on
(`PPB_ENABLE_NVTX`):

| Region | Encloses |
|---|---|
| `matmul` | the matrix-multiplication call the benchmark times (`matmul-cublas` / `matmul-naive` in `matMul_cuda`, which links both) |
| `init` | the one-off computation of face normals and segment vectors of the polyhedral model |
| `evaluate` | one evaluation of the gravity model, including its reduction |

Both tools report the region a row belongs to: `ncu --nvtx` per kernel launch, `nsys` per sampled
window. Launches outside every region — Kokkos' architecture query, desul's lock arrays, a
framework's buffer staging — are dropped from the CSV, so the tables hold the benchmark's own
kernels and nothing else. That is also what separates `init` from `evaluate`: the polyhedral
initialization runs *once* no matter how many iterations follow, so the two-point subtraction of
step 2 divides each region by how many more times *that region* was entered, not by the iteration
count.

Three things make the sampled numbers trustworthy; all three are implemented in
`ppbcc/profiling/nsys.py`.

* **The analytic FLOP model is exact for matrix multiplication and borrowed for the polyhedral
  kernel.** For the matrix multiplication `ncu` counts exactly 137 438 953 472 = 2·M·N·K = 2·4096³
  for *every one* of the ten CUDA-backed implementations (cuBLAS is +0.085 %, its epilogue does
  slightly different arithmetic), so substituting the closed form for the four sampled points costs
  nothing. The polyhedral model has no closed form — a per-face loop over three segments, with
  branches — so its constant is the *median of the counted implementations*: 1121 FLOP/face over
  3 145 728 faces, 3.5259e9 FLOP. The nine counted implementations spread from 1000 FLOP/face
  (AdaptiveCpp, OpenMP target) to 1360 (Slang-Cuda), a factor of 1.36, and the four sampled points
  inherit that as an uncertainty on top of everything else: their FLOP/s is a *time* measurement
  rescaled by another implementation's work. The rule is scoped to the `evaluate` region, so the
  `init` rows carry no work rather than the wrong work.
* **One-time setup is subtracted, not amortised.** A lazily initialising backend compiles pipelines
  and fills buffers on its first call, and the sampler cannot tell that from a kernel launch — both
  are compute work on the same GPU. Every binary is therefore profiled twice, at 1 and at 20
  iterations, and the difference is divided by the 19 extra calls. `-O iterations` controls it.
  Without it `polyhedral_vulkan` measures 295 ms instead of 49 ms; with the regions, that 246 ms is
  attributed to `init` where it belongs.
* **Data-movement phases are excluded.** Kompute moves its tensors with a compute shader, so each
  Vulkan dispatch is bracketed by two compute-active windows that are not the kernel — and, because
  the region encloses everything the benchmark times, they sit inside it. They run at ~49 %
  occupancy against the dispatch's ~96 %, so a window below `-O activity-ratio` (0.7) times the
  busiest one in the same region is dropped.

**The sampled durations agree with independent timers to within 2–7 %**, which is the check that the
window detection works. The one exception is `polyhedral_slang_vulkan`, 22 % below its steady-state
wall clock:

| Executable | `nsys` window | independent value |
|---|---|---|
| `matMul_ocl` | 37.84 ms | 39.78 ms (`clGetEventProfilingInfo`) |
| `matMul_boost` | 38.01 ms | 39.63 ms (same) |
| `matMul_vulkan` | 38.17 ms | 39.11 ms (application timer) |
| `matMul_slang_vulkan` | 38.25 ms | 41.04 ms (same) |
| `polyhedral_ocl` | 0.615 ms | 0.598 ms (steady-state wall clock, `Results_NVIDIA_RTX5080.csv`) |
| `polyhedral_boost` | 0.618 ms | 0.601 ms (same) |
| `polyhedral_vulkan` | 48.87 ms | 47.29 ms (same) |
| `polyhedral_slang_vulkan` | 46.23 ms | 59.09 ms (same) |

The **byte** counts are the weaker half, on both tools.

*Sampled against counted.* Profiling the CUDA binaries with `nsys` as well as `ncu` puts the sampled
integral within 0.3 % of the counted one on the cuBLAS kernel but 29 % above it on the naive one.
The Nsight Graphics cross-check of step 4 lands 17–31 % above the sampled value on the Vulkan
kernels — the same order as the spread below.

*Counted against itself.* At 4096² one matrix is 67 MB against this card's 64 MB of L2, so whether a
tiled SGEMM re-reads its operand from DRAM or from L2 depends on where the driver happens to place
the allocation — which depends on what ran before it. `dram__bytes.sum` therefore varies by up to a
factor of two between runs of the *same binary* while the kernel time does not move at all:
`matMul_kokkos` measured 2.44 GB in the batch below and 4.44 / 4.48 / 4.43 GB in three separate
runs, both at 39.82 ms. The tables below are one consistent batch; `matMul_alpaka`, whose 4×4096
grid of 1024-thread blocks makes it the most sensitive, sits at the low end of its range there.

**Read the arithmetic intensity of every point as good to about a factor of two, and the kernel
times as good to a few percent.** The ranking by time is solid; the ranking by traffic is not.

## Reproduce

```bash
# Toolchains and the plotting environment.
source ~/load_all_llvm.sh && conda activate standard-math

# Profiling builds: one input, one iteration (PPB_PROFILING).
cmake --preset cuda-llvm-profiling && cmake --build build-cuda-llvm-profiling -j
# OpenACC and Stdpar need NVHPC; only those targets (Kokkos does not compile with nvc++, and
# polyhedral_stdpar hits an NVVM bug inside NVHPC 26.5's own stdexec headers).
# Configure into an empty directory: on an existing cache CMake decides that CMAKE_CUDA_COMPILER
# changed, wipes the cache and re-runs without the preset's variables, leaving PPB_PROFILING=OFF.
source ~/load_all_nvhpc.sh && rm -rf build-cuda-nvhpc-profiling && cmake --preset cuda-nvhpc-profiling
cmake --build build-cuda-nvhpc-profiling -j --target matMul_acc matMul_stdpar polyhedral_acc

# Time every binary unprofiled first, so a profiler that hangs is recognisable as a hang.
cd build-cuda-llvm-profiling
for e in $(find src/matrixMultiplication src/polyhedralGravity -maxdepth 2 -type f -executable); do
  echo "== $e"; time timeout 900 "./$e"; done; cd ..
```

Everything below runs from the repository root. All runs are bounded by `--timeout`, so one hanging
binary costs that many seconds rather than the session.

### 1. Nsight Compute — the ten paradigms with a CUDA context

```bash
# LLVM build: every paradigm except OpenACC/Stdpar. ~2 min.
ppbcc profile -b build-cuda-llvm-profiling -p src -r "matMul_.*" "polyhedral_.*" -x ".*_cpp$" \
  -d ../profiling-rtx5080 -H "NVIDIA RTX5080" --no-csv --timeout 420
# NVHPC build: OpenACC and Stdpar, into the same report directory.
ppbcc profile -b build-cuda-nvhpc-profiling -p src -r "matMul_acc$" "matMul_stdpar$" "polyhedral_acc$" \
  -d ../profiling-rtx5080 -H "NVIDIA RTX5080" --no-csv --timeout 420
# Re-read the reports without running anything and write the consolidated CSV.
ppbcc profile -b . -d profiling-rtx5080 -r ".*" --skip-profile -H "NVIDIA RTX5080" \
  -o results/Profiling_NCU_NVIDIA_RTX5080
```

`ppbcc profile` collects with `--nvtx`, so every launch carries its region and the runtime's own
bootstrap kernels drop out on their own — no kernel-name exclusion list is needed.

> [!NOTE]
> `ncu` used to hang on `matMul_kokkos` and `matMul_omp` — one CPU core at 100 %, the application
> suspended, the GPU idle. The cause is Google Benchmark's `MaybeReenterWithoutASLR`, which `execv`s
> the process: the profiler attaches to the pre-exec process and, with `--target-processes all`, to
> the re-executed one as well. `ppbcc profile` now launches through `setarch <arch> -R`, so Google
> Benchmark skips the re-exec and both finish in seconds. It was never the size of the working set.

### 2. Nsight Systems — the four paradigms without one

```bash
# Two runs per binary (1 and 20 iterations); the difference cancels one-time setup. ~3 min.
ppbcc profile --profiler nsys -b build-cuda-llvm-profiling -p src \
  -r "matMul_ocl$" "matMul_boost$" "matMul_vulkan$" "matMul_slang_vulkan$" \
     "polyhedral_ocl$" "polyhedral_boost$" "polyhedral_vulkan$" "polyhedral_slang_vulkan$" \
  -d ../profiling-nsys-rtx5080 -H "NVIDIA RTX5080" --timeout 900 -O iterations=20 \
  --peak-performance 5.74e13 --peak-bandwidth 9.592e11 \
  --analytic-flop "matMul_.*=137438953472" --analytic-flop 'polyhedral_.*\[evaluate\]=3.5259e9' \
  -o "$PWD/results/Profiling_NSYS_NVIDIA_RTX5080"
```

`-o` is absolute here because `--build-dir` is the working directory of the whole pipeline, so a
relative output path would land inside the build folder.

`--analytic-flop` matches against `<executable>[<region>]`, which is how the work model reaches the
`evaluate` region only: `init` computes normals and segment vectors, not the gravity model, so it
gets no FLOP count rather than the wrong one.

The ceilings are the ones step 1 *measured* (`peak_sustained` counters scaled by the clock each unit
actually ran at). The sampler reports percentages of peak but not the peak itself, so the bandwidth
is also what turns the sampled percentages into bytes.

### 3. The plots

```bash
# Merge both tools' tables. --region picks which phase is plotted; the CSVs keep every region.
ppbcc profile --from-csv results/Profiling_NCU_NVIDIA_RTX5080.csv results/Profiling_NSYS_NVIDIA_RTX5080.csv \
  -r "matMul_" -H "NVIDIA RTX5080" --no-csv --label-points --region "^matmul" \
  --roofline "results/matmul_roofline_NVIDIA_RTX5080.pdf"
ppbcc profile --from-csv results/Profiling_NCU_NVIDIA_RTX5080.csv results/Profiling_NSYS_NVIDIA_RTX5080.csv \
  -r "polyhedral_" -H "NVIDIA RTX5080" --no-csv --label-points --region "^evaluate$" \
  --roofline "results/polyhedral_roofline_NVIDIA_RTX5080.pdf"
```

### 4. Nsight Graphics — cross-check on the two Vulkan paradigms

```bash
# Bounds the trace to one queue submission -- without a swapchain there are no frames, so submit
# index is the only boundary. Submission 1 is the steady-state dispatch; submission 0 also carries
# the lazy initialisation, which is why '-O submit=auto' (busiest submission) picks the wrong
# one for the polyhedral binaries.
ppbcc profile --profiler ngfx -b build-cuda-llvm-profiling -p src \
  -r "matMul_vulkan$" "matMul_slang_vulkan$" "polyhedral_vulkan$" "polyhedral_slang_vulkan$" \
  -d ../profiling-ngfx-rtx5080 -H "NVIDIA RTX5080" --timeout 600 -O submit=1 \
  --peak-performance 5.74e13 --peak-bandwidth 9.592e11 \
  --analytic-flop "matMul_.*=137438953472" --analytic-flop "polyhedral_.*=3.5259e9" \
  -o "$PWD/results/Profiling_NGFX_NVIDIA_RTX5080"
```

It agrees with the sampled numbers to within a third on both quantities:

| Executable | `nsys` ms / GB | `ngfx` ms / GB | application timer |
|---|---|---|---|
| `matMul_vulkan` | 38.17 / 5.23 | 47.36 / 6.13 | 39.11 ms |
| `matMul_slang_vulkan` | 38.25 / 5.10 | 47.69 / 6.11 | 41.04 ms |
| `polyhedral_vulkan` | 48.87 / 0.55 | 59.31 / 0.72 | 47.29 ms (steady-state wall clock) |

The traced durations are 21–24 % above the sampled ones and above the application's own timer, which
is the tracing overhead — GPU Trace collects counters over the whole submission, Nsight Systems
samples them. The DRAM traffic agrees to 17–20 % on the matrix multiplication and 31 % on the
polyhedral kernel, which is the same order as the run-to-run spread of `ncu`'s own counted bytes.

`polyhedral_slang_vulkan` has no cross-check: its submission 1 exports an empty trace, and
submission 0 is the one that also carries the initialisation.

## Results

One row per implementation and region, summed over every kernel launch in it. `Tool` says where the
row came from; the polyhedral table shows `evaluate`, and the `init` region is in the CSV next to it.

### Matrix multiplication, 4096², FP32

| Executable | Paradigm | Region | Tool | ms | DRAM GB | AI [FLOP/byte] | TFLOP/s |
|---|---|---|---|---:|---:|---:|---:|
| `matMul_cuda` | CUDA | matmul-cublas | ncu | 3.60 | 0.54 | 252.6 | 38.24 |
| `matMul_ocl` | OpenCL | matmul | nsys | 37.84 | 3.29 | 41.8 | 3.63 |
| `matMul_boost` | Boost | matmul | nsys | 38.01 | 3.37 | 40.8 | 3.62 |
| `matMul_vulkan` | Vulkan | matmul | nsys | 38.17 | 5.23 | 26.3 | 3.60 |
| `matMul_slang_vulkan` | Slang-Vulkan | matmul | nsys | 38.25 | 5.10 | 26.9 | 3.59 |
| `matMul_raja` | RAJA | matmul | ncu | 39.36 | 2.46 | 55.9 | 3.49 |
| `matMul_kokkos` | Kokkos | matmul | ncu | 39.82 | 2.44 | 56.3 | 3.45 |
| `matMul_acpp` | AdaptiveCpp | matmul | ncu | 39.94 | 1.95 | 70.3 | 3.44 |
| `matMul_hip` | HIP | matmul | ncu | 43.00 | 2.77 | 49.5 | 3.20 |
| `matMul_cuda` | CUDA | matmul-naive | ncu | 43.12 | 2.81 | 48.8 | 3.19 |
| `matMul_acc` | OpenACC | matmul | ncu | 57.12 | 1.86 | 74.0 | 2.41 |
| `matMul_stdpar` | Stdpar | matmul | ncu | 58.51 | 5.66 | 24.3 | 2.35 |
| `matMul_alpaka` | Alpaka | matmul | ncu | 59.22 | 1.20 | 114.1 | 2.32 |
| `matMul_omp` | OpenMP | matmul | ncu | 61.91 | 2.57 | 53.5 | 2.22 |
| `matMul_slang_cuda` | Slang-Cuda | matmul | ncu | 83.02 | 1.99 | 69.2 | 1.66 |

Every implementation does exactly 137.44 GFLOP, and every hand-written one lands **below both roofs**
— 3.6 TFLOP/s against a 57.4 TFLOP/s compute ceiling at an intensity of 26–70 FLOP/byte, where the
memory roof is already 25 TFLOP/s and above. None of these kernels is bandwidth-bound; they are
latency-bound, which is the expected signature of a naive tiled SGEMM. The spread across the
fourteen is a factor of 2.2 in time.

cuBLAS is in the table because `matMul_cuda` links it next to the naive kernel, and the NVTX regions
now tell the two apart. It is the only point that behaves like a tuned SGEMM: 10.7× faster than the
binary's own kernel and 6.1× faster than the best portable one, at 67 % of the compute roof. It is
what the roofline is for — everything else has an order of magnitude in hand.

### Polyhedral gravity, SHAPE_SFM_3M (3 145 728 faces), FP32, `evaluate` region

| Executable | Paradigm | Tool | ms | DRAM GB | AI [FLOP/byte] | TFLOP/s |
|---|---|---|---:|---:|---:|---:|
| `polyhedral_acc` | OpenACC | ncu | 0.53 | 0.46 | 7.7 | 6.66 |
| `polyhedral_kokkos` | Kokkos | ncu | 0.55 | 0.38 | 9.1 | 6.39 |
| `polyhedral_alpaka` | Alpaka | ncu | 0.61 | 0.48 | 7.2 | 5.67 |
| `polyhedral_ocl` | OpenCL | nsys | 0.62 | 0.43 | 8.2 | 5.73 |
| `polyhedral_boost` | Boost | nsys | 0.62 | 0.43 | 8.2 | 5.71 |
| `polyhedral_acpp` | AdaptiveCpp | ncu | 0.63 | 0.39 | 8.0 | 5.02 |
| `polyhedral_hip` | HIP | ncu | 0.88 | 0.72 | 4.9 | 4.00 |
| `polyhedral_cuda` | CUDA | ncu | 0.90 | 0.73 | 4.8 | 3.91 |
| `polyhedral_slang_cuda` | Slang-Cuda | ncu | 1.18 | 0.85 | 5.0 | 3.62 |
| `polyhedral_raja` | RAJA | ncu | 1.25 | 0.37 | 9.9 | 2.94 |
| `polyhedral_omp` | OpenMP | ncu | 12.22 | 0.69 | 4.6 | 0.26 |
| `polyhedral_slang_vulkan` | Slang-Vulkan | nsys | 46.23 | 0.47 | 7.5 | 0.08 |
| `polyhedral_vulkan` | Vulkan | nsys | 48.87 | 0.55 | 6.5 | 0.07 |

The four `nsys` rows carry the borrowed 1121 FLOP/face rather than a counted one, so read their
FLOP/s and intensity as their measured *time* placed on the same axis as the rest, not as an
independent measurement of their arithmetic.

Here the points sit *on* the memory roof rather than under it: at 4.6–9.9 FLOP/byte the ceiling is
4.4–9.5 TFLOP/s, and the leaders reach 73–90 % of it. The kernel is genuinely bandwidth-bound, and
the ranking follows the DRAM traffic — the leaders move roughly half the bytes of the CUDA/HIP pair and are
~1.6× faster. The two Vulkan implementations are ~90× off the leaders; that is not a measurement
artefact, the archived steady-state wall-clock times in `Results_NVIDIA_RTX5080.csv` show the same
47 ms and 59 ms.

Separating `init` from `evaluate` is what makes this table comparable at all. The initialization is
a real kernel — 0.7–0.8 ms on the CUDA paradigms, and 246 ms of pipeline compilation and staging on
`polyhedral_vulkan` — and folding it into the same point would have put the Vulkan implementations
5× further down for work they do exactly once.

`polyhedral_stdpar` is missing from both: it does not build with NVHPC 26.5, which fails inside its
own `stdexec/static_thread_pool.hpp` with `NVVM_ERROR_COMPILATION`.

## Tools that were tried and are not used

| Route | Verdict |
|---|---|
| [PTXprofiler](https://github.com/ProjectPhysX/PTXprofiler) | Counts instructions statically from PTX. Its own README notes that "non-unrolled loops [are] counted once but may execute multiple times", which is exactly the K-loop of the matrix multiplication and the per-segment loop of the polyhedral kernel. It cannot produce the FLOP count these two need, and where it could, `ncu` already counts the executed instructions rather than the static ones. |
| `nsys --trace=opencl` | Gone: CUDA 13.3's Nsight Systems no longer lists `opencl` among the trace values, so there is no OpenCL API timeline at all. GPU-metrics sampling is unaffected — it does not care which API submitted the work, which is the whole reason step 2 works. |
| `nsys --trace=vulkan` | Works, but yields a dispatch timeline without counters. GPU metrics are the useful part. |
| Nsight Graphics GPU Trace, per-dispatch attribution | The `GPUTRACE_REGIMES` table stays empty for a headless compute application: without a swapchain there are no frames to attribute to, and `--time-every-action` does not change that. Only the trace-wide aggregate is available, hence the submit-index bracketing. |
| LIKWID NvMarker | Still a good CUDA cross-check (see [`README.md`](README.md) step 5.2), but blind to OpenCL and Vulkan for the same CUPTI reason as `ncu`, and to `matMul_slang_cuda`, which uses its own driver-API context. |
| HPCToolkit / TAU | Give OpenCL *kernel times* but no counters; `nsys` supplies both, so they add nothing here. |
