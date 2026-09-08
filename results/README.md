# Reproducibility

Every step here uses [`ppbcc`](https://github.com/schuhmaj/performance-portability-code-complexity)
(`pip install .` in that repository). Run `ppbcc <command> --help` for the full option list, or see
its [documentation](https://schuhmaj.github.io/performance-portability-code-complexity/) — this file
only records the exact invocations that produced the archived data.

## Contents

| Path | Produced by |
|---|---|
| `<platform>/*.json` | Step 1 — raw Google Benchmark reports, one per executable |
| `Results_<platform>.csv` | Step 2 — one consolidated CSV per platform |
| `code-complexity/*.csv` | Step 3 — SLOC and Halstead metrics per file and per implementation |
| `Profiling_<platform>.csv` | Step 5.1 — per-kernel Nsight Compute counters and roofline quantities |
| `Profiling_NCU_<platform>.csv` | [`ROOFLINE.md`](ROOFLINE.md) step 1 — the same, at the profiling sizes that cover every paradigm |
| `Profiling_NSYS_<platform>.csv` | [`ROOFLINE.md`](ROOFLINE.md) step 2 — sampled counters for the paradigms without a CUDA context |
| `Profiling_NGFX_<platform>.csv` | [`ROOFLINE.md`](ROOFLINE.md) step 4 — GPU Trace counters, cross-check on Vulkan |
| `Profiling_LIKWID_<platform>.csv` | Step 5.2 — per-region LIKWID counters, same columns |
| `Profiling_comparison_<platform>.csv` | Step 5.3 — the two backends side by side, per executable |
| `matmul_roofline_likwid_<platform>.pdf` | Step 5.2 — matrix-multiplication roofline |
| `polyhedral_roofline_likwid_<platform>.pdf` | Step 5.2 — polyhedral-gravity roofline |
| `matmul_roofline_<platform>.pdf` | [`ROOFLINE.md`](ROOFLINE.md) — matrix multiplication, **all** paradigms |
| `polyhedral_roofline_<platform>.pdf` | [`ROOFLINE.md`](ROOFLINE.md) — polyhedral gravity, **all** paradigms |

Step 4 renders the plots into the working directory; they are not committed here. The `.ncu-rep`
and `.likwid-marker` files behind the profiling CSVs are not committed either — step 5 regenerates
them.

## 1. Run the benchmarks

From the build directory of the platform under test:

```bash
ppbcc benchmark -p src -H "Intel GPU Max 1550" \
  -r "vec_.*" "matMul_.*" "nbody_.*" "polyhedral_.*" -x ".*_cpp" --dry-run
```

`--dry-run` only lists the matched executables; drop it to run them. Adapt `-H` to the platform and
add `-o "Results_<platform>"` to name the consolidated CSV directly.

## 2. Consolidate archived reports into CSVs

`--skip-benchmark` runs nothing and only collects existing `*.json` reports, which is how the
archived CSVs in this folder are regenerated:

```bash
ppbcc benchmark -p ./nvidia-rtx3080 -r ".*\.json" --skip-benchmark \
  -H "NVIDIA RTX3080" -o "Results_NVIDIA_RTX3080"
```

Repeat for the remaining platforms:

| `-p` | `-H` | `-o` |
|---|---|---|
| `./nvidia-rtx3080` | `NVIDIA RTX3080` | `Results_NVIDIA_RTX3080` |
| `./nvidia-rtx4060` | `NVIDIA RTX4060` | `Results_NVIDIA_RTX4060` |
| `./nvidia-rtx5080` | `NVIDIA RTX5080` | `Results_NVIDIA_RTX5080` |
| `./nvidia-gh200` | `NVIDIA GH200` | `Results_NVIDIA_GH200` |
| `./amd-mi210` | `AMD MI210` | `Results_AMD_MI210` |
| `./intel-data_center_gpu_max_1550` | `Intel GPU Max 1550` | `Results_Intel_GPU_Max_1550` |

## 3. Code complexity

Both `code-complexity/code-complexity.csv` (per implementation, required by the plots below) and
`code-complexity/code-complexity-files.csv` (per file) are regenerated from the repository root
with:

```bash
python scripts/generate_code_complexity.py
```

The script wraps `ppbcc code-complexity`: the file-level CSV is a plain run over `src/`, and every
implementation aggregate is a `--aggregate` run over an explicit source manifest, whose `TOTAL` row
is collected with `distinct_operators`/`distinct_operands`/`total_operators`/`total_operands`
renamed to the conventional Halstead columns `n1`, `n2`, `N1`, `N2`. Use `--dry-run` to print every
concrete invocation, including all resolved shared files, without changing the results.

The manifest exists because the aggregates cannot be derived from directory structure alone. It
separates the AdaptiveCpp shared-memory, CUDA matrix, OpenMP host/device, Vulkan matrix and Kokkos
NBody reduction variants, and keeps NBody Naive, LinkedCells and VerletLists apart. Slang shaders
are deliberately aggregated twice — with the CUDA host code for `Slang-Cuda` and with the Vulkan
host code for `Slang-Vulkan` — and OpenCL kernels reused by Boost.Compute are counted in the Boost
aggregates as well. Quoted repository-local includes are resolved transitively and the
implementation units of required `src/common` utilities are added automatically; a dependency on a
*different* benchmark problem is rejected, which prevents unrelated application code from silently
inflating an aggregate.

## 4. Plots

The combined charts (application efficiency, ꟼP over complexity, platform ranking, and ꟼP over
problem size) need the code-complexity CSV from step 3:

```bash
# N-body
ppbcc p3analysis NBody ./Results_* --complexity ./code-complexity/code-complexity.csv -c combined \
  --complexity-metric halstead-difficulty --normalize --log-complexity \
  --non-zero-pp -s avg -x "VerletLists|LinkedCells|Reduction" --remove-description -l \
  --export-to-csv --legend--vertical
# Polyhedral gravity model
ppbcc p3analysis Polyhedral ./Results_* --complexity ./code-complexity/code-complexity.csv -c combined \
  --complexity-metric halstead-difficulty --normalize --log-complexity \
  --non-zero-pp --remove-description -s avg -l --export-to-csv
# Matrix multiplication
ppbcc p3analysis MatrixMultiplication ./Results_* --complexity ./code-complexity/code-complexity.csv -c combined \
  --complexity-metric halstead-difficulty --normalize --log-complexity \
  --non-zero-pp --remove-description -s avg -x "Cublas" -l --export-to-csv
# Vector addition
ppbcc p3analysis VecAdd ./Results_* --complexity ./code-complexity/code-complexity.csv -c combined \
  --complexity-metric halstead-difficulty --normalize --log-complexity \
  --non-zero-pp --remove-description -s avg -x "Cublas" -l --export-to-csv
```

`--normalize` replaces the `--additive` these charts used before. Expressing complexity as a
percentage of the sequential C++ baseline keeps every value positive, which is what makes
`--log-complexity` usable — and the log axis is what the polyhedral chart needs, because its two
CUDA implementations sit at roughly 500 % of the baseline and squeeze the other twelve paradigms
into the left quarter of a linear axis. Ranking is unaffected either way: both options apply the
same transform to every paradigm.

> [!NOTE]
> The lower-right panel is a **heatmap** over the benchmark sizes, one row per implementation and
> one column per size, and no longer a line plot: with fourteen implementations the lines
> overlapped to the point of being unreadable. Its rows keep the descending-ꟼP order of the
> platform-ranking panel to its left, so the two lower panels line up row for row and the paradigm
> labels are not repeated. `--log-size` is therefore obsolete — the axis is categorical — and is
> accepted but ignored, with a warning.
>
> Its column labels use exponents whenever every benchmark size is an exact power of the same
> base: `2^5 … 2^14` for matrix multiplication and `10^1 … 10^8` for N-body and vector addition.
> That is short enough to carry the same font size as the cells themselves. The polyhedral meshes
> (2780, 12796, …) are no such sweep and keep decimal labels at a smaller size, because upright
> labels that long would otherwise grow the figure's bounding box.

### 4.1 SLOC against Halstead difficulty

`-c complexity-comparison` plots two complexity metrics against each other, one marker per
paradigm, with the identity line drawn in: above it the y metric charges a paradigm more than the
x metric does — *dense vocabulary* — and below it the x metric charges more — *verbose code*. Both
axes are relative to the sequential C++ baseline, which is what makes the identity line meaningful,
so the chart always normalizes and rejects `--additive`.

```bash
# Vector addition
ppbcc p3analysis VecAdd ./Results_* --complexity ./code-complexity/code-complexity.csv \
  -c complexity-comparison --complexity-metric halstead-difficulty --compare-metric sloc \
  --log-complexity --non-zero-pp -s avg -x "Cublas" --remove-description -l
# Matrix multiplication
ppbcc p3analysis MatrixMultiplication ./Results_* --complexity ./code-complexity/code-complexity.csv \
  -c complexity-comparison --complexity-metric halstead-difficulty --compare-metric sloc \
  --log-complexity --non-zero-pp -s avg -x "Cublas" --remove-description -l
# N-body
ppbcc p3analysis NBody ./Results_* --complexity ./code-complexity/code-complexity.csv \
  -c complexity-comparison --complexity-metric halstead-difficulty --compare-metric sloc \
  --log-complexity --non-zero-pp -s avg -x "VerletLists|LinkedCells|Reduction" \
  --remove-description -l
# Polyhedral gravity model
ppbcc p3analysis Polyhedral ./Results_* --complexity ./code-complexity/code-complexity.csv \
  -c complexity-comparison --complexity-metric halstead-difficulty --compare-metric sloc \
  --log-complexity --non-zero-pp -s avg --remove-description -l
```

`--compare-metric` names the x-axis metric and `--complexity-metric` the y-axis one; both accept
the same names and aliases as everywhere else. `-l` suppresses the in-plot legend *and* the
per-point paradigm labels, because the paper keys these charts to the shared vertical legend of
step 4; drop it to get a self-contained chart with both.

Spearman's rho and Kendall's tau are **not** drawn by default — they belong in the running text,
where they can be discussed. `--legend-complexity-comparison-coefficients` boxes them in the
top-right corner; with points hugging the identity line that corner is not always empty, so the
box is drawn *under* the markers and may sit behind one.

The boxplots need no complexity data and use `-s all` instead of `-s avg`:

```bash
ppbcc p3analysis NBody ./Results_* -c boxplot \
  --non-zero-pp -s all -x "VerletLists|LinkedCells|Reduction" --remove-description
ppbcc p3analysis Polyhedral ./Results_* -c boxplot \
  --non-zero-pp -s all --remove-description
ppbcc p3analysis MatrixMultiplication ./Results_* -c boxplot \
  --non-zero-pp -s all -x "Cublas" --remove-description
ppbcc p3analysis VecAdd ./Results_* -c boxplot \
  --non-zero-pp -s all -x "Cublas" --remove-description
```

`--export-to-csv` additionally writes the underlying application-efficiency and
performance-portability tables, so every quoted number can be checked against a CSV rather than
read off a plot.

## 5. Roofline models

The roofline models need a **separate build** of the benchmark: a profiler replays every kernel
launch it sees, so the executables must be reduced to one input and one iteration first. That is
what the CMake option `PPB_PROFILING` does, and the two `*-profiling` presets switch it on:

```bash
# From the repository root
cmake --preset cuda-llvm-profiling && cmake --build build-cuda-llvm-profiling -j
# Only for OpenACC and Stdpar (they need the NVHPC toolchain)
cmake --preset cuda-nvhpc-profiling && cmake --build build-cuda-nvhpc-profiling -j
```

`ppbcc profile` then drives a profiler over the resulting binaries. `--profiler` picks which one:

| `--profiler` | Tool | Instrumentation | One row is |
|---|---|---|---|
| `ncu` (default) | Nsight Compute | none — attaches from outside | one kernel launch |
| `likwid` | LIKWID NvMarker | `PPB_ENABLE_LIKWID=ON`, see [`src/common/Marker.h`](../src/common/Marker.h) | one marked region |
| `nsys` | Nsight Systems GPU metrics | none — samples the GPU device-wide | one marked region |
| `ngfx` | Nsight Graphics GPU Trace | none — injects a Vulkan layer | one queue submission |

The last two exist because `ncu` and LIKWID both read their counters through CUPTI, which needs a
current CUDA context: OpenCL and Vulkan build their own and are invisible to both. See
[`ROOFLINE.md`](ROOFLINE.md) for the roofline that covers every paradigm.

All four write the same columns, so they share the CSV writer and the plot. They differ in what they
can attribute: `ncu` names every individual kernel, while LIKWID keeps all kernels of one marked
region together — which is the more useful unit whenever an implementation reaches the GPU through
a runtime that launches several kernels per call.

Every row also carries the region the launch belongs to. A profiler names a kernel whatever the
compiler called it, which for several paradigms is nothing useful — AdaptiveCpp launches four
kernels all called `__acpp_sscp_kernel` — so the benchmark names the work itself: the regions of
[`src/common/Marker.h`](../src/common/Marker.h) become NVTX ranges (`matmul` for the matrix
multiplication, `init` and `evaluate` for the polyhedral gravity) that `ncu` and `nsys` both report.
Launches outside every region are the runtime setting itself up and are dropped from the CSV;
`--all-kernels` keeps them, `--region` restricts the plot to some of them.

### 5.1 Nsight Compute

Both builds write their reports into one shared folder (`--report-dir` is relative to
`--build-dir`), so the two runs can be consolidated together:

```bash
ppbcc profile -b build-cuda-llvm-profiling  -p src -r "polyhedral_.*"    \
  -d ../profiling-nvidia-rtx5080 -H "NVIDIA RTX5080" --no-csv
ppbcc profile -b build-cuda-nvhpc-profiling -p src -r "polyhedral_acc$" \
  -d ../profiling-nvidia-rtx5080 -H "NVIDIA RTX5080" --no-csv
```

Each run leaves `profiling-nvidia-rtx5080/<executable>.ncu-rep` next to the Google-Benchmark report
`<executable>.json`, which supplies the paradigm and precision that `ncu` itself does not know.
`--no-csv` skips the intermediate CSV; the final one comes from the consolidation step, which
re-parses the reports without running anything:

```bash
ppbcc profile -b . -d profiling-nvidia-rtx5080 -r "polyhedral_.*" --skip-profile \
  -H "NVIDIA RTX5080" -o results/Profiling_NVIDIA_RTX5080 \
  --roofline polyhedral_roofline_ncu.pdf
```

> [!NOTE]
> This step covers only the paradigms with a CUDA context, which is why the archived
> `polyhedral_roofline_<platform>.pdf` comes from [`ROOFLINE.md`](ROOFLINE.md) instead: it merges
> this table with the sampled one and carries all thirteen implementations.

Kokkos' architecture query and desul's lock-array initialization never reach the table: the runtime
launches them once at startup, outside every marked region, and they compute nothing. The plot shows
one point per implementation *and region*, so an implementation with a separate `init` kernel
contributes two.

> [!NOTE]
> `ncu` used to hang on `matMul_kokkos` and `matMul_omp` — one CPU core at 100 %, the application
> suspended, the GPU idle, no progress. The cause is not the size of the working set but Google
> Benchmark's `MaybeReenterWithoutASLR`, which `execv`s the process: the profiler attaches to the
> pre-exec process and, with `--target-processes all`, to the re-executed one as well. `ppbcc
> profile` now launches through `setarch <arch> -R`, so Google Benchmark skips the re-exec and both
> binaries finish in seconds. `-O aslr=on` restores the old behaviour, `--timeout` bounds it.

The plot places one point per implementation — work, traffic and kernel time summed over all its
launches — and uses the same paradigm colors as the `p3analysis` charts of step 4. Both roofs are
*measured*: every `peak_sustained` counter is scaled with the clock the corresponding unit actually
ran at, which on the RTX 5080 gives 57.4 TFLOP/s (FP32) and 959 GB/s. Use `-a dominant` for the
longest kernel only, `-a none` for one point per launch, and `-m l2`/`-m l1` to move the arithmetic
intensity to another level of the memory hierarchy.

### 5.2 LIKWID

LIKWID reads the counters from *inside* the application, around regions the source opens itself.
`src/common/Marker.h` provides the wrapper and every matrix-multiplication and polyhedral
implementation marks the kernel it already times — one region `matmul`, one region `evaluate`. The
markers are compiled in only with `PPB_ENABLE_LIKWID=ON` and expand to nothing otherwise, so the
measuring build is unaffected:

```bash
module load likwid
LIKWID_PREFIX=$(dirname $(dirname $(which likwid-perfctr)))
cmake --preset cuda-llvm-profiling -B build-likwid \
  -DPPB_ENABLE_LIKWID=ON -DLIKWID_ROOT="$LIKWID_PREFIX"
cmake --build build-likwid -j
```

LIKWID exposes no `peak_sustained` counters, so the two roofs have to be supplied. They are a
property of the hardware rather than of the measuring tool, so the values step 5.1 measured are the
right ones to pass:

```bash
# Collect. Every executable runs twice, once per event group.
ppbcc profile --profiler likwid -b build-likwid -p src \
  -r "matMul_.*" "polyhedral_.*" -x ".*_cpp$" \
  -O lib="$LIKWID_PREFIX/lib" -d ../profiling-likwid-marker \
  --peak-performance 5.74e13 --peak-bandwidth 9.592e11 \
  -H "NVIDIA RTX5080" -o "$PWD/results/Profiling_LIKWID_NVIDIA_RTX5080"

# Plot, re-reading the marker files without running anything.
for problem in matMul polyhedral; do
  ppbcc profile --profiler likwid -b . -r "${problem}_.*" --skip-profile \
    -d profiling-likwid-marker -a none --label-points \
    --peak-performance 5.74e13 --peak-bandwidth 9.592e11 \
    -H "NVIDIA RTX5080" --no-csv --roofline \
    --roofline "results/${problem}_roofline_likwid_NVIDIA_RTX5080.pdf"
done
```

`-a none` puts one point per marked region rather than one per implementation. It matters for
`matMul_cuda`, which links cuBLAS and the naive kernel into one binary: they are separate regions
(`matmul-cublas`, `matmul-naive`) and the default `-a sum` would add them together.

> [!IMPORTANT]
> A region tag has to be unique within an executable. LIKWID accumulates every entry of a tag into
> one record, so two implementations sharing a tag report their counters merged, with no way to tell
> them apart afterwards.

Three things about LIKWID 5.5.1 are worth knowing, because `ppbcc` works around all of them.

**`likwid-perfctr` is not used.** It programs the counters from a separate daemon process, which on
this hardware fails with `CUPTI_ERROR_INVALID_PARAMETER` — identically under CUDA 12.9 and 13.3, so
it is not a CUDA-version problem. `ppbcc` instead sets `LIKWID_NVMON_GPUS`, `LIKWID_NVMON_EVENTS`
and `LIKWID_NVMON_FILEPATH` and runs the binary directly, so the instrumented process programs its
own counters.

**Every executable runs twice.** The floating-point (SMSP) and DRAM counters cannot be programmed in
one pass — `createConfigImage` rejects the combination — so the events are split into two groups and
merged afterwards. That is why the report directory holds `<executable>.flops.likwid-marker` and
`<executable>.memory.likwid-marker`. The two runs do not cost the same: programming three SMSP
counters perturbs a kernel far more than a single DRAM counter — the 16384² matrix multiplication
takes ~36 s in the first group and ~3.6 s in the second. The counters are unaffected by that
overhead, the clock is not, so the reported duration is the **smaller** of the two.

**The counters are corrected by a factor of two.** On this hardware LIKWID reports every Nvidia
counter at exactly half its true value. This was established against kernels with an analytically
known instruction count: for `FADD`, `FMUL`, `FFMA` and `DRAM_BYTES_SUM` alike, `ncu` reproduces the
analytic value exactly and LIKWID returns half of it, independently of the counter domain and of how
many launches a region contains. `COUNTER_SCALE` in `ppbcc/profiling/likwid.py` compensates.
Arithmetic intensity is a ratio of two equally scaled counters and is therefore unaffected either
way; only the absolute rates need the correction.

**What the counters cannot see.** NvMarker reads the counters through CUPTI, which needs a *current
CUDA context*. Five of the thirteen matrix-multiplication paradigms produce a timed region with all
counters at zero, and they are kept in the CSV precisely so the gap is visible:

| Executable | Why |
|---|---|
| `*_ocl`, `*_boost` | OpenCL builds its own context |
| `*_vulkan`, `*_slang_vulkan` | Vulkan builds its own context |
| `matMul_slang_cuda` | CUDA, but through the *driver* API: it calls `cuInit` and `cuCtxSetCurrent` with a context of its own, so the counter session LIKWID opened belongs to a different context |

The last row is the interesting one: being written in CUDA is not sufficient, the kernel has to run
in the context NvMarker bound to. `polyhedral_slang_cuda` does and is measured normally.

### 5.3 Cross-validating the two

`scripts/compare_profilers.py` merges the two CSVs per executable and prints the LIKWID/ncu ratio of
every roofline quantity, so the agreement is checkable rather than asserted:

```bash
python scripts/compare_profilers.py \
  results/Profiling_NVIDIA_RTX5080.csv results/Profiling_LIKWID_NVIDIA_RTX5080.csv \
  -o results/Profiling_comparison_NVIDIA_RTX5080.csv
```

### 5.4 What can profile OpenCL and Vulkan?

Both profilers above read their counters through **CUPTI**, which needs a current
CUDA context. Nine of the thirteen paradigms have one — CUDA, HIP-on-Nvidia,
Kokkos, RAJA, Alpaka, AdaptiveCpp, stdpar, OpenACC and OpenMP target all reach the
GPU through the CUDA driver, so both tools see them. OpenCL and Vulkan build their
own contexts and are invisible to both, which is a property of the vendor's
instrumentation stack rather than of the tool. On this machine:

* Nvidia's OpenCL exposes **no** counter extension. The device advertises
  `cl_nv_device_attribute_query` (static properties) and nothing matching
  performance, counter or profiling; CUPTI has no OpenCL interface; and Nsight
  Systems dropped OpenCL entirely — `nsys profile --trace` in CUDA 13.3 no
  longer lists it.
* `VK_KHR_performance_query`, the Vulkan extension that would expose hardware
  counters, is **absent** from the RTX 5080's device extension list. It is
  supported by Mesa (AMD, Intel), not by Nvidia's proprietary driver.

That leaves the following, in the order we would reach for them.

| Route | OpenCL | Vulkan | What it yields |
|---|:-:|:-:|---|
| API timing (`clGetEventProfilingInfo`, `vkCmdWriteTimestamp`) | ✅ | ✅ | Kernel duration only — already what the benchmark reports |
| Analytic work model | ✅ | ✅ | FLOPs and bytes from the algorithm, no tool needed |
| HPCToolkit `hpcrun -e gpu=opencl` | ✅ | ❌ | Per-kernel time and transfer bytes; no FLOP or DRAM counters |
| Nsight Systems `--trace=vulkan` | ❌ | ✅ | Dispatch timeline; no counters |
| Nsight Graphics, *GPU Trace Profiler* | ❌ | ✅ | Unit throughputs (SM, DRAM, L2) as % of peak |
| `VK_KHR_performance_query` | ❌ | ✅* | Vendor counters — *AMD and Intel only* |

Only the last two can close the counter gap at all, and both are Vulkan-only.
**Nsight Graphics' GPU Trace Profiler** is the one Nvidia tool that reads hardware
counters for a Vulkan workload; it reports time-sliced *unit throughputs* rather
than SASS instruction counts, so a roofline built from it uses (% of peak × peak)
instead of a counted FLOP total. It is not installed here, so we have not verified
it against this benchmark. For OpenCL on Nvidia there is no counter route at all.

**Recommendation.** Do not try to close the gap tool-side. The comparable quantity
across all thirteen paradigms is an **analytic work model**: matrix multiplication
does exactly `2·M·N·K` FLOPs and moves a known number of bytes, and the polyhedral
kernel has a fixed per-face operation count. Combined with the kernel time each
implementation already measures through its own API, that gives an arithmetic
intensity and an attained FLOP/s for *every* paradigm, on *every* vendor, with no
profiler in the loop. The counters from `ncu` then serve as a validation of that
model on the subset where they exist — which is exactly the role they play above,
where `ncu` reproduced an analytically known instruction count exactly.
