# Roofline models for every implementation

One roofline per problem covering **all** paradigms — including the four that reach the GPU without
a CUDA context and are therefore invisible to Nsight Compute. The general reproduction recipe, and
the LIKWID/HPCToolkit/TAU notes, are in [`README.md`](README.md). This file records only what
produced `matmul_roofline_NVIDIA_RTX5080.pdf` and `polyhedral_roofline_NVIDIA_RTX5080.pdf`.

RTX 5080 (GB203, sm_120), driver 610.43.02, Nsight Compute 2026.2.1, Nsight Systems 2026.1.3,
Nsight Graphics 2026.3.1. Matrix multiplication 2026-09-15 (first taken 2026-09-08); polyhedral
gravity retaken 2026-09-11 after its kernel was rewritten (see
[What changed](#what-changed-in-the-polyhedral-kernel)), and `polyhedral_omp` once more on
2026-09-13 after its reduction moved into the evaluation kernel.

The reports behind the CSVs live in `profiling/`: the matrix multiplication in `profiling-rtx5080`,
`profiling-nsys-rtx5080` and `profiling-ngfx-rtx5080`, the polyhedral gravity in the
`*.polyhedral-update` folder next to each. The polyhedral reports in the plain folders are the
kernel *before* the rewrite (1121 FLOP/face) and feed only the *ms before* column below. What every
CSV column means and how it is computed is in [Reading the CSVs](#reading-the-csvs).

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
| `init` | the one-off computation of the plane unit normals of the polyhedral model |
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
  branches — so its constant is the *median of the counted implementations*: 587 FLOP/face over
  3 145 728 faces, 1.84705e9 FLOP. The ten counted implementations spread from 539 FLOP/face
  (AdaptiveCpp) to 655 (RAJA), a factor of 1.22, and the four sampled points inherit that as an
  uncertainty on top of everything else: their FLOP/s is a *time* measurement rescaled by another
  implementation's work. The rule is scoped to the `evaluate` region, so the `init` rows carry no
  work rather than the wrong work.
* **One-time setup is subtracted, not amortised.** A lazily initialising backend compiles pipelines
  and fills buffers on its first call, and the sampler cannot tell that from a kernel launch — both
  are compute work on the same GPU. Every binary is therefore profiled twice, at 1 and at 20
  iterations, and the difference is divided by the 19 extra calls. `-O iterations` controls it.
  Without it `polyhedral_vulkan` measures 48 ms instead of 14 ms; with the regions, that 34 ms is
  attributed to `init` where it belongs.
* **Data-movement phases are excluded.** Kompute moves its tensors with a compute shader, so each
  Vulkan dispatch is bracketed by two compute-active windows that are not the kernel — and, because
  the region encloses everything the benchmark times, they sit inside it. They run at ~49 %
  occupancy against the dispatch's ~96 %, so a window below `-O activity-ratio` (0.7) times the
  busiest one in the same region is dropped.

**The sampled durations agree with independent timers to within 2–10 %**, which is the check that the
window detection works:

| Executable | `nsys` window | independent value |
|---|---|---|
| `matMul_ocl` | 37.16 ms | 39.78 ms (`clGetEventProfilingInfo`) |
| `matMul_boost` | 37.08 ms | 39.63 ms (same) |
| `matMul_vulkan` | 37.07 ms | 39.11 ms (application timer) |
| `matMul_slang_vulkan` | 37.07 ms | 41.04 ms (same) |
| `polyhedral_ocl` | 0.246 ms | 0.269 ms (steady-state wall clock, Google Benchmark, measuring build) |
| `polyhedral_boost` | 0.250 ms | 0.274 ms (same) |
| `polyhedral_vulkan` | 14.08 ms | 13.81 ms (same) |
| `polyhedral_slang_vulkan` | 13.17 ms | 12.87 ms (same) |

The two sub-millisecond OpenCL kernels are the loosest pair, 9 % shorter than the wall clock, which
also contains the host side of every call (argument setting, enqueueing, and reading the result back).

The **byte** counts are the weaker half, on both tools.

*Sampled against counted.* Profiling the CUDA binaries with `nsys` as well as `ncu` puts the sampled
integral within 0.3 % of the counted one on the cuBLAS kernel but 29 % above it on the naive one.
The Nsight Graphics cross-check of step 4 lands 28–29 % above the sampled value on the matrix
multiplication's Vulkan kernels, and 4.7× below it on `polyhedral_vulkan`.

*Counted against itself.* At 4096² one matrix is 67 MB against this card's 64 MB of L2, so whether a
tiled SGEMM re-reads its operand from DRAM or from L2 depends on where the driver happens to place
the allocation — which depends on what ran before it. `dram__bytes.sum` therefore varies by up to a
factor of two between runs of the *same binary* while the kernel time does not move at all:
`matMul_kokkos` measured 2.44 GB in the batch below and 4.44 / 4.48 / 4.43 GB in three separate
runs, both at 39.82 ms. The tables below are one consistent batch; `matMul_alpaka`, whose 4×4096
grid of 1024-thread blocks makes it the most sensitive, sits at the low end of its range there.
Re-profiling the matrix multiplication on 2026-09-15 moved the kernel times by at most 3 % but the
DRAM bytes by up to 34 % (`matMul_hip` 2.77 → 1.83 GB, `matMul_acc` 1.86 → 2.44 GB), the same effect
again.

**Read the arithmetic intensity of every point as good to about a factor of two, and the kernel
times as good to a few percent.** The ranking by time is solid; the ranking by traffic is not.

## Reproduce

```bash
# Toolchains and the plotting environment.
source ~/load_all_llvm.sh && conda activate standard-math

# Profiling builds: one input, one iteration (PPB_PROFILING).
cmake --preset cuda-llvm-profiling && cmake --build build-cuda-llvm-profiling -j
# OpenACC and Stdpar need NVHPC; only those targets (Kokkos does not compile with nvc++).
# Configure into an empty directory: on an existing cache CMake decides that CMAKE_CUDA_COMPILER
# changed, wipes the cache and re-runs without the preset's variables, leaving PPB_PROFILING=OFF.
source ~/load_all_nvhpc.sh && rm -rf build-cuda-nvhpc-profiling && cmake --preset cuda-nvhpc-profiling
cmake --build build-cuda-nvhpc-profiling -j --target matMul_acc matMul_stdpar polyhedral_acc polyhedral_stdpar

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
ppbcc profile -b build-cuda-nvhpc-profiling -p src \
  -r "matMul_acc$" "matMul_stdpar$" "polyhedral_acc$" "polyhedral_stdpar$" \
  -d ../profiling-rtx5080 -H "NVIDIA RTX5080" --no-csv --timeout 420
```

The archived CSV is re-read from the reports without running anything. The two problems come from
different folders (see the top of this file), so each is consolidated on its own and the two tables
are then merged, which is also what writes the unit-annotated header:

```bash
cd results/profiling
S=$(mktemp -d)
ppbcc profile -b . -d profiling-rtx5080 -r "matMul_.*" --skip-profile \
  -H "NVIDIA RTX5080" -o $S/matmul
ppbcc profile -b . -d profiling-rtx5080.polyhedral-update -r "polyhedral_.*" --skip-profile \
  -H "NVIDIA RTX5080" -o $S/polyhedral
ppbcc profile --from-csv $S/matmul.csv $S/polyhedral.csv -H "NVIDIA RTX5080" \
  -o ../Profiling_NCU_NVIDIA_RTX5080
```

Re-reading needs no `ncu` CLI: without one on the `PATH`, `ppbcc` opens the reports through Nsight
Compute's `ncu_report` Python module, which the macOS host application ships as well. The numbers
are identical; the only difference is that `Kernel` and `Kernel Signature` keep namespaces the CLI
abbreviates (`Kokkos::Impl::cuda_parallel_launch_local_memory` instead of
`Kokkos::cuda_parallel_launch_local_memory`).

`ppbcc profile` collects with `--nvtx`, so every launch carries its region and the runtime's own
bootstrap kernels drop out on their own — no kernel-name exclusion list is needed. An existing
report is reused rather than re-profiled; `--force` profiles that executable again.

`polyhedral_stdpar` exits with `SIGABRT` (status 6) under `ncu`, after its report and Google
Benchmark JSON have been written; run without a profiler it exits cleanly.

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
  --analytic-flop "matMul_.*=137438953472" --analytic-flop 'polyhedral_.*\[evaluate\]=1.84705e9' \
  -o "$PWD/results/Profiling_NSYS_NVIDIA_RTX5080"
```

`-o` is absolute here because `--build-dir` is the working directory of the whole pipeline, so a
relative output path would land inside the build folder.

`--analytic-flop` matches against `<executable>[<region>]`, which is how the work model reaches the
`evaluate` region only: `init` computes plane normals, not the gravity model, so it gets no FLOP
count rather than the wrong one.

The ceilings are the ones step 1 *measured* (`peak_sustained` counters scaled by the clock each unit
actually ran at). The sampler reports percentages of peak but not the peak itself, so the bandwidth
is also what turns the sampled percentages into bytes.

To re-read the samples without profiling again — after step 1 moved the median, say — replace
`-b build-cuda-llvm-profiling -p src` by `-b . -d profiling-nsys-rtx5080 --skip-profile`. The
patterns then select among the SQLite exports (`polyhedral_ocl.sqlite`), so anchor them as
`"^polyhedral_ocl(\.|$)"` instead of `"polyhedral_ocl$"`, or nothing matches. The archived CSV is
consolidated like step 1, per problem and then merged:

```bash
cd results/profiling
S=$(mktemp -d)
ROOFS="--peak-performance 5.74e13 --peak-bandwidth 9.592e11"
ppbcc profile --profiler nsys -b . -d profiling-nsys-rtx5080 --skip-profile -O iterations=20 $ROOFS \
  -r "^matMul_(ocl|boost|vulkan|slang_vulkan)(\.|$)" --analytic-flop "matMul_.*=137438953472" \
  -H "NVIDIA RTX5080" -o $S/matmul
ppbcc profile --profiler nsys -b . -d profiling-nsys-rtx5080.polyhedral-update --skip-profile \
  -O iterations=20 $ROOFS -r "^polyhedral_(ocl|boost|vulkan|slang_vulkan)(\.|$)" \
  --analytic-flop 'polyhedral_.*\[evaluate\]=1.84705e9' -H "NVIDIA RTX5080" -o $S/polyhedral
ppbcc profile --from-csv $S/matmul.csv $S/polyhedral.csv -H "NVIDIA RTX5080" \
  -o ../Profiling_NSYS_NVIDIA_RTX5080
```

Step 4's `Profiling_NGFX_NVIDIA_RTX5080.csv` follows the same pattern with `--profiler ngfx`, the
`profiling-ngfx-rtx5080[.polyhedral-update]` folders and `-r "^matMul_(vulkan|slang_vulkan)"` /
`-r "^polyhedral_(vulkan|slang_vulkan)"`.

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
  --analytic-flop "matMul_.*=137438953472" --analytic-flop "polyhedral_.*=1.84705e9" \
  -o "$PWD/results/Profiling_NGFX_NVIDIA_RTX5080"
```

On the matrix multiplication it agrees with the sampled numbers to within 30 % on both
quantities; on the polyhedral kernel only the duration does:

| Executable | `nsys` ms / GB | `ngfx` ms / GB | application timer |
|---|---|---|---|
| `matMul_vulkan` | 37.07 / 4.22 | 45.91 / 5.43 | 39.11 ms |
| `matMul_slang_vulkan` | 37.07 / 4.23 | 45.90 / 5.39 | 41.04 ms |
| `polyhedral_vulkan` | 14.08 / 0.15 | 15.94 / 0.03 | 13.81 ms (steady-state wall clock) |

The traced durations are 13–24 % above the sampled ones and above the application's own timer, which
is the tracing overhead — GPU Trace collects counters over the whole submission, Nsight Systems
samples them. The DRAM traffic agrees to 28–29 % on the matrix multiplication. On the rewritten
polyhedral kernel the trace counts 4.7× fewer bytes than the sampler (before the rewrite the two were
31 % apart, at 0.55 and 0.72 GB). The sampled 0.15 GB is the plausible one of the two: the kernels
`ncu` counts move 0.14–0.68 GB per call for the same work.

`polyhedral_slang_vulkan` has no cross-check: none of the probed submissions carries compute work in
its trace.

## Reading the CSVs

All three tables share their leading columns; `Profiling_NCU_*` appends every raw Nsight Compute
counter behind them. Each header carries its unit in brackets. A roofline point is one
implementation and region: its rows are summed (`FLOP`, `Memory Traffic`, `Duration`) before the
ratios are formed, so never average `Arithmetic Intensity` or `Performance` over rows.

### Identifying columns

| Column | Meaning |
|---|---|
| `Benchmark Problem`, `Paradigm`, `Precision` | Read from the Google Benchmark JSON written next to the report; `Precision` also picks which `FLOP FPxx` becomes `FLOP`. |
| `Hardware` | The `-H` label. |
| `Executable`, `Region` | Binary and the NVTX range of [`Marker.h`](../src/common/Marker.h) the launch ran in (`matmul`, `init`, `evaluate`); empty for `ngfx`, whose submissions carry no range. |
| `Kernel ID` | Position of the launch in the report, counting the dropped bootstrap kernels, which is why Kokkos starts at 5 (`ncu`); a window index (`nsys`); always 0 (`ngfx`). |
| `Kernel`, `Kernel Signature` | Short and full kernel name (`ncu`); a description of the sampled window or trace (`nsys`, `ngfx`). |
| `Grid Size [blocks]`, `Block Size [threads/block]` | CUDA launch configuration as (x, y, z); launched GPU threads = ∏grid · ∏block. Explains a point far below the roof, enters no formula. |
| `Memory Level` | Which counter `Memory Traffic` was read from (`DRAM`, or `L2`/`L1/TEX` with `-m`). |

### Roofline quantities

| Column | Meaning | Computed as (`ncu`) | Used for |
|---|---|---|---|
| `FLOP FP32 [FLOP]` | Floating-point operations the kernel executed in FP32. An FMA (`a·b + c`) is one multiply and one add, so it counts twice. | `fadd + fmul + 2·ffma` of the `sm__sass_thread_inst_executed_op_*_pred_on.sum` counters | `FLOP` |
| `FLOP FP64 [FLOP]`, `FLOP FP16 [FLOP]` | The same for the `d*` and `h*` counters. Kept so a mixed-precision kernel stays visible. | `dadd + dmul + 2·dfma`, `hadd + hmul + 2·hfma` | `FLOP` when that is the build precision |
| `FLOP [FLOP]` | The work of the launch in the precision the binary was built with. `nsys` and `ngfx` cannot count it, so they take it from `--analytic-flop` (2·4096³ or 587 FLOP/face · 3 145 728). | `FLOP FP32` for an FP32 build | `Arithmetic Intensity`, `Performance` |
| `Duration [s]` | GPU time of the launch. `nsys`: length of the compute-active sampled windows in the region; `ngfx`: traced submission time. | `gpu__time_duration.sum · 10⁻⁹` | `Performance` |
| `Memory Traffic [Byte]` | Bytes the kernel moved between the GPU cores and the chosen memory level. `nsys` integrates the sampled DRAM read and write throughput instead; `ngfx` reads `dram__sectors.sum`. | `dram__bytes.sum` (DRAM) | `Arithmetic Intensity`, memory-roof share |
| `Arithmetic Intensity [FLOP/Byte]` | Work per byte of traffic: the x axis of the roofline. | `FLOP / Memory Traffic` | x coordinate of the point |
| `Performance [FLOP/s]` | Attained FLOP rate: the y axis of the roofline. | `FLOP / Duration` | y coordinate of the point |
| `Peak Performance [FLOP/s]` | Compute roof: the FMA rate the SM units can sustain, scaled with the clock they actually ran at. Given by `--peak-performance` for `nsys`/`ngfx`. | `2 · ffma.sum.peak_sustained · sm__cycles_elapsed.avg.per_second` | horizontal roof; ridge point |
| `Peak Bandwidth [Byte/s]` | Memory roof: the byte rate DRAM can sustain at its actual clock. Given by `--peak-bandwidth` for `nsys`/`ngfx`, which also turns their sampled percentages into bytes. | `dram__bytes.sum.peak_sustained · dram__cycles_elapsed.avg.per_second` | slanted roof; ridge point |

The plot and the tables below combine these as follows:

| Derived value | Equation | Meaning |
|---|---|---|
| Roof at a point | `min(Peak Performance, Peak Bandwidth · Arithmetic Intensity)` | Highest `Performance` the hardware allows at this intensity. |
| Ridge point [FLOP/Byte] | `Peak Performance / Peak Bandwidth` | Intensity where the two roofs meet (60 FLOP/byte here); left of it memory limits, right of it compute. |
| *memory roof* [%] | `Performance / (Peak Bandwidth · Arithmetic Intensity)` = `Memory Traffic / (Duration · Peak Bandwidth)` | How close a kernel is to being bandwidth-bound; equals `gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed` for a single launch. |
| compute roof [%] | `Performance / Peak Performance` | How close a kernel is to being compute-bound. |
| FLOP/face | `FLOP / 3 145 728` | Work per face of the polyhedral kernel; the median of the counted ones is the `nsys` work model. |

### Raw Nsight Compute counters (`Profiling_NCU_*` only)

Units are Nsight Compute's own (`IMetric.unit()`); `.sum` adds and `.avg` averages over all units of
that kind on the GPU.

| Column | Meaning | Relation | Used for |
|---|---|---|---|
| `gpu__time_duration.sum [ns]` | Measured time of the kernel launch (user time where collectable, else wall clock). | — | `Duration` |
| `sm__cycles_elapsed.avg [cycle]` | Clock cycles that elapsed on a streaming multiprocessor (SM) during the launch. | `≈ Duration · sm__cycles_elapsed.avg.per_second` | context |
| `sm__cycles_elapsed.avg.per_second [cycle/s]` | The SM clock rate the launch actually ran at. | — | `Peak Performance` |
| `sm__sass_thread_inst_executed_op_{f,d,h}{add,mul,fma}_pred_on.sum [inst]` | Executed FP32 (`f`), FP64 (`d`), FP16 (`h`) additions, multiplications and FMAs whose predicates were all true, i.e. not masked by a branch. The `f` counters include the `FH*` variants. | — | `FLOP FP32/FP64/FP16` |
| `sm__sass_thread_inst_executed_op_{f,d,h}fma_pred_on.sum.peak_sustained [inst/cycle]` | FMAs per clock cycle the SMs can sustain at most (10 752 in FP32 on this card). | — | `Peak Performance` |
| `sm__inst_executed_pipe_tensor.sum [inst]` | Instructions executed by the tensor pipe. Never counted as FLOP. | — | context |
| `dram__bytes.sum [Byte]` | Bytes accessed in DRAM. | — | `Memory Traffic` (`-m dram`) |
| `dram__bytes.sum.peak_sustained [Byte/cycle]` | DRAM bytes per DRAM clock cycle the hardware can sustain (64). | — | `Peak Bandwidth` |
| `dram__cycles_elapsed.avg.per_second [cycle/s]` | DRAM clock rate during the launch. | — | `Peak Bandwidth` |
| `lts__t_bytes.sum [Byte]`, `lts__t_bytes.sum.peak_sustained [Byte/cycle]`, `lts__cycles_elapsed.avg.per_second [cycle/s]` | The same three for the L2 cache (LTS). | `Peak = peak_sustained · per_second` | `Memory Traffic`/`Peak Bandwidth` with `-m l2` |
| `l1tex__t_bytes.sum [Byte]`, `l1tex__t_bytes.sum.peak_sustained [Byte/cycle]`, `l1tex__cycles_elapsed.avg.per_second [cycle/s]` | The same three for the L1/texture cache. | `Peak = peak_sustained · per_second` | `Memory Traffic`/`Peak Bandwidth` with `-m l1` |
| `launch__grid_size [blocks]`, `launch__block_size [threads/block]` | Product of the grid and block dimensions of the launch. | `∏Grid Size`, `∏Block Size` | context |
| `launch__waves_per_multiprocessor [1]` | Nsight Compute's scheduling statistic for the launch; the report carries no description for it. | — | context |
| `sm__throughput.avg.pct_of_peak_sustained_elapsed [%]` | Nsight Compute's SM throughput as a share of peak, assuming ideal load balancing across the SMs. Not the same as `Performance / Peak Performance` (Kokkos `evaluate`: 46 % against 18 %). | — | context |
| `gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed [%]` | DRAM throughput as a share of peak. | `= Memory Traffic / (Duration · Peak Bandwidth)` | cross-check of *memory roof* |

## Why some regions hold several kernels

`Profiling_NCU_*` has one row per kernel **launch**. The polyhedral `evaluate` region sums up 3 145 728
per-face contributions, and the number of rows it gets depends on where that reduction runs:

| Paradigm | Rows in `evaluate` | Where the reduction runs |
|---|---|---|
| Kokkos | 1 | `Kokkos::parallel_reduce`: the reduction is folded into the one launch, in block-local memory (`cuda_parallel_launch_local_memory`). |
| OpenMP | 1 | `target teams distribute parallel for reduction(+ : result)`: inside the one offloaded kernel. |
| RAJA | 1 | `RAJA::ReduceSum<cuda_reduce>` inside the `forall` kernel. |
| OpenACC, Alpaka | 1 | **On the host.** The kernel writes every face's result into a device buffer, which is copied back and summed in a CPU loop. |
| AdaptiveCpp | 3 | `sycl::reduction`, a hierarchical reduction: the main kernel evaluates the faces and reduces each work group (336 groups · 128), then two dedicated reduction kernels reduce 336 → 3 → 1 partial results (grids 3 and 1). |
| CUDA, HIP, Slang-Cuda | 3 | `thrust::reduce` after the evaluation kernel: CUB's `DeviceReduceKernel` sums the per-face buffer block-wise, `DeviceReduceSingleTileKernel` combines the block sums. |
| Stdpar | 3 | `std::reduce(par_unseq)` after the `for_each`, which NVHPC also maps onto CUB's two reduce kernels. |

So yes, the reduction is sometimes separate, and that affects what a point measures. For CUDA the
reduction takes 0.17 of the 0.66 ms in the row sum; for AdaptiveCpp its two extra kernels take only
0.005 ms, because its main kernel already reduced almost everything. For OpenACC and Alpaka the
reduction and the device-to-host copy run outside every kernel, so their `evaluate` point **does not
include them**: they are in the region's wall clock but not on the roofline.

## Results

One row per implementation and region, summed over every kernel launch in it. `Tool` says where the
row came from; the polyhedral table shows `evaluate`, and the `init` region is in the CSV next to it.

### Matrix multiplication, 4096², FP32

| Executable | Paradigm | Region | Tool | ms | DRAM GB | AI [FLOP/byte] | TFLOP/s |
|---|---|---|---|---:|---:|---:|---:|
| `matMul_cuda` | CUDA | matmul-cublas | ncu | 3.60 | 0.53 | 260.7 | 38.26 |
| `matMul_vulkan` | Vulkan | matmul | nsys | 37.07 | 4.22 | 32.6 | 3.71 |
| `matMul_slang_vulkan` | Slang-Vulkan | matmul | nsys | 37.07 | 4.23 | 32.5 | 3.71 |
| `matMul_boost` | Boost | matmul | nsys | 37.08 | 2.77 | 49.6 | 3.71 |
| `matMul_ocl` | OpenCL | matmul | nsys | 37.16 | 5.27 | 26.1 | 3.70 |
| `matMul_raja` | RAJA | matmul | ncu | 39.36 | 2.41 | 57.1 | 3.49 |
| `matMul_kokkos` | Kokkos | matmul | ncu | 39.81 | 2.44 | 56.3 | 3.45 |
| `matMul_acpp` | AdaptiveCpp | matmul | ncu | 39.94 | 1.92 | 71.5 | 3.44 |
| `matMul_hip` | HIP | matmul | ncu | 42.96 | 1.83 | 74.9 | 3.20 |
| `matMul_cuda` | CUDA | matmul-naive | ncu | 43.07 | 2.01 | 68.4 | 3.19 |
| `matMul_stdpar` | Stdpar | matmul | ncu | 58.62 | 5.49 | 25.0 | 2.34 |
| `matMul_acc` | OpenACC | matmul | ncu | 58.73 | 2.44 | 56.3 | 2.34 |
| `matMul_alpaka` | Alpaka | matmul | ncu | 59.27 | 1.12 | 122.4 | 2.32 |
| `matMul_omp` | OpenMP | matmul | ncu | 62.47 | 2.04 | 67.5 | 2.20 |
| `matMul_slang_cuda` | Slang-Cuda | matmul | ncu | 83.16 | 1.98 | 69.3 | 1.65 |

Every implementation does exactly 137.44 GFLOP, and every hand-written one lands **below both roofs**
— 3.7 TFLOP/s against a 57.4 TFLOP/s compute ceiling at an intensity of 25–75 FLOP/byte (Alpaka
122), where the memory roof is already 24 TFLOP/s and above. None of these kernels is
bandwidth-bound; they are latency-bound, which is the expected signature of a naive tiled SGEMM. The
spread across the fourteen is a factor of 2.2 in time.

cuBLAS is in the table because `matMul_cuda` links it next to the naive kernel, and the NVTX regions
now tell the two apart. It is the only point that behaves like a tuned SGEMM: 12.0× faster than the
binary's own kernel and 10.3× faster than the best portable one, at 67 % of the compute roof. It is
what the roofline is for — everything else has an order of magnitude in hand.

### Polyhedral gravity, SHAPE_SFM_3M (3 145 728 faces), FP32, `evaluate` region

| Executable | Paradigm | Tool | ms | ms before | DRAM GB | AI [FLOP/byte] | TFLOP/s | memory roof |
|---|---|---|---:|---:|---:|---:|---:|---:|
| `polyhedral_kokkos` | Kokkos | ncu | 0.179 | 0.55 | 0.14 | 12.7 | 10.02 | 83 % |
| `polyhedral_ocl` | OpenCL | nsys | 0.246 | 0.62 | 0.13 | 14.8 | 7.52 | 53 % |
| `polyhedral_boost` | Boost | nsys | 0.250 | 0.62 | 0.13 | 14.7 | 7.40 | 52 % |
| `polyhedral_alpaka` | Alpaka | ncu | 0.310 | 0.61 | 0.23 | 8.1 | 6.00 | 77 % |
| `polyhedral_acc` | OpenACC | ncu | 0.313 | 0.53 | 0.23 | 8.2 | 6.00 | 76 % |
| `polyhedral_acpp` | AdaptiveCpp | ncu | 0.414 | 0.63 | 0.14 | 12.5 | 4.10 | 34 % |
| `polyhedral_omp` | OpenMP | ncu | 0.414 | 12.22 | 0.16 | 11.5 | 4.43 | 40 % |
| `polyhedral_stdpar` | Stdpar | ncu | 0.533 | — | 0.37 | 5.2 | 3.61 | 72 % |
| `polyhedral_raja` | RAJA | ncu | 0.608 | 1.25 | 0.14 | 14.4 | 3.39 | 25 % |
| `polyhedral_cuda` | CUDA | ncu | 0.663 | 0.90 | 0.49 | 3.7 | 2.71 | 76 % |
| `polyhedral_hip` | HIP | ncu | 0.665 | 0.88 | 0.49 | 3.7 | 2.72 | 76 % |
| `polyhedral_slang_cuda` | Slang-Cuda | ncu | 0.900 | 1.18 | 0.68 | 3.0 | 2.27 | 79 % |
| `polyhedral_slang_vulkan` | Slang-Vulkan | nsys | 13.17 | 46.23 | 0.14 | 13.0 | 0.14 | 1 % |
| `polyhedral_vulkan` | Vulkan | nsys | 14.08 | 48.87 | 0.15 | 12.0 | 0.13 | 1 % |

*ms before* is the 2026-09-08 batch, taken with the kernel as it was before the rewrite; *memory roof*
is the achieved TFLOP/s as a share of the 959 GB/s bandwidth times the point's intensity, which at
3.0–14.8 FLOP/byte is the lower of the two roofs for every point. The four `nsys` rows carry the
borrowed 587 FLOP/face rather than a counted one, so read their FLOP/s and intensity as their measured
*time* placed on the same axis as the rest, not as an independent measurement of their arithmetic.

Seven implementations sit just under the memory roof: Kokkos, Slang-Cuda, Alpaka, OpenACC, CUDA, HIP
and Stdpar reach 72–83 % of it, moving 690–790 GB/s of DRAM traffic. For them the kernel is
bandwidth-bound, and the time follows the bytes moved per call: Kokkos moves 0.14 GB and needs
0.18 ms, CUDA and HIP 0.49 GB and 0.66 ms, Slang-Cuda 0.68 GB and 0.90 ms. OpenCL and Boost (53 %),
OpenMP (40 %), AdaptiveCpp (34 %) and RAJA (25 %) move as few bytes as Kokkos but take 1.4–3.4× as
long, so something other than bandwidth holds them back. Kokkos is the only point with a noticeable
share of the compute roof, 18 % of 57.4 TFLOP/s. The two Vulkan implementations are ~75× off Kokkos at
1 % of their roof; that is not a measurement artefact, their steady-state wall clocks are the same
13.8 and 12.9 ms.

Separating `init` from `evaluate` still matters, if less than before the rewrite. The initialization
now only computes the plane normals: 0.07–0.15 ms on the paradigms `ncu` counts, 0.22 ms sampled on
OpenCL and Boost, and 34 ms of pipeline compilation and staging on `polyhedral_vulkan` (246 ms before).
Folded into the same point, the latter would put the Vulkan implementation another 3.4× further down
for work it does exactly once.

`polyhedral_stdpar` is new in the table: before the rewrite it did not compile with NVHPC 26.5 (an
NVVM error reported inside NVHPC's own `stdexec` headers, but triggered by the old kernel body).

### What changed in the polyhedral kernel

Every polyhedral implementation follows the optimized Kokkos backend of the
[polyhedral gravity model](https://github.com/esa/polyhedral-gravity-model/tree/kokkos-backend)
(branch `kokkos-backend`) since 2026-09-11:

* segment vectors and segment normals are recomputed per evaluation instead of cached, so only the
  plane unit normals remain, and `init` computes nothing else,
* `P'` is the cached unit normal scaled by `N_p · v_0`, so the Hessian plane is never formed, and `P''`
  is never formed either — `σ_pq`, `h_pq`, `s1` and `s2` all follow from projecting `P' - v_q` onto
  `n_pq` and onto `G_pq`,
* Tsoulis' options 1–3 collapse into `s1 = -u`, `s2 = |G_pq| - u`; only option 4 still branches,
* the logarithm and the arc tangent (one `atan` of the difference, plus a branch offset) are evaluated
  unconditionally and then selected, and the four singularity cases are one selection of a factor,
* `EPSILON_ZERO`, `π` and `sgn` are in the evaluation precision, which also fixes FP32 results that
  were off by up to 50× near the body.

That halves the work per face (1121 → 587 FLOP) and cuts the DRAM traffic by 1.2–4.3×; every
implementation but OpenMP (below) is 1.3–3.5× faster. Launch bounds — `__launch_bounds__(256, 3)` on CUDA and HIP,
`LaunchBounds<256, 3>` on Kokkos, `cuda_exec_explicit<256, 3>` on RAJA — were swept on `polyhedral_cuda`
and did not move its kernel (0.49–0.52 ms without bounds and at (256, 2), (256, 3), (256, 4)): at 70 %
DRAM throughput it waits for memory, not for registers. The other paradigms cannot express them.

`polyhedral_omp` gained the most, 12.22 → 0.41 ms, for a reason of its own: it wrote every face's
result into a device buffer and reduced that in a second kernel, a `target parallel for reduction`
that ran as a single block of 128 threads and took 11.8 of the 12.2 ms. The reduction now happens
inside the evaluation kernel (`target teams distribute parallel for reduction(+ : result)`).

## Tools that were tried and are not used

| Route | Verdict |
|---|---|
| [PTXprofiler](https://github.com/ProjectPhysX/PTXprofiler) | Counts instructions statically from PTX. Its own README notes that "non-unrolled loops [are] counted once but may execute multiple times", which is exactly the K-loop of the matrix multiplication and the per-segment loop of the polyhedral kernel. It cannot produce the FLOP count these two need, and where it could, `ncu` already counts the executed instructions rather than the static ones. |
| `nsys --trace=opencl` | Gone: CUDA 13.3's Nsight Systems no longer lists `opencl` among the trace values, so there is no OpenCL API timeline at all. GPU-metrics sampling is unaffected — it does not care which API submitted the work, which is the whole reason step 2 works. |
| `nsys --trace=vulkan` | Works, but yields a dispatch timeline without counters. GPU metrics are the useful part. |
| Nsight Graphics GPU Trace, per-dispatch attribution | The `GPUTRACE_REGIMES` table stays empty for a headless compute application: without a swapchain there are no frames to attribute to, and `--time-every-action` does not change that. Only the trace-wide aggregate is available, hence the submit-index bracketing. |
| LIKWID NvMarker | Still a good CUDA cross-check (see [`README.md`](README.md) step 5.2), but blind to OpenCL and Vulkan for the same CUPTI reason as `ncu`, and to `matMul_slang_cuda`, which uses its own driver-API context. |
| HPCToolkit / TAU | Give OpenCL *kernel times* but no counters; `nsys` supplies both, so they add nothing here. |
