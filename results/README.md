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
| `profiling/` | Step 5 — profiling reports, CSVs and roofline plots; **only in the Zenodo release** |

Step 4 renders the plots into the working directory; they are not committed here.

## 1. Run the benchmarks

From the build directory of the platform under test:

```bash
ppbcc benchmark -p src -H "Intel Max 1550" \
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
ppbcc benchmark -p ./nvidia-rtx4060 -r ".*\.json" --skip-benchmark \
  -H "NVIDIA RTX4060" -o "Results_NVIDIA_RTX4060"
ppbcc benchmark -p ./nvidia-rtx5080 -r ".*\.json" --skip-benchmark \
  -H "NVIDIA RTX5080" -o "Results_NVIDIA_RTX5080"
ppbcc benchmark -p ./nvidia-gh200 -r ".*\.json" --skip-benchmark \
  -H "NVIDIA GH200" -o "Results_NVIDIA_GH200"
ppbcc benchmark -p ./amd-mi210 -r ".*\.json" --skip-benchmark \
  -H "AMD MI210" -o "Results_AMD_MI210"
ppbcc benchmark -p ./intel-data_center_gpu_max_1550 -r ".*\.json" --skip-benchmark \
  -H "Intel Max 1550" -o "Results_Intel_Max_1550"
```

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

The profiling instrumentation is measured as if it had never been added (`--exclude-macro`,
`--exclude-header`, which need a `ppbcc` with code-complexity exclusions): `src/common/Marker.h` and
`src/common/Profiling.h` are not counted, their `#include` lines and every `PPB_MARKER_*` region are
removed, and conditionals on `PPB_PROFILING`, `PPB_ENABLE_LIKWID` and `PPB_ENABLE_NVTX` are resolved
as the benchmark build compiles them. The only trace left is the call
`ppb::profiling::initialize(&argc, argv)` in place of `benchmark::Initialize(&argc, argv)`, one
operator and one operand more per vector-addition aggregate (the other problems call it from `main`
files, which the manifest does not count). Running the script on the sources before the
instrumentation (`7520ac3`) and after it gives exactly that difference and nothing else.

> [!NOTE]
> `src/common/cuda/helper_math.h` and `helper_math_double.h` are excluded as well (`--exclude-header
> common/cuda/helper_math*.h`): they are NVIDIA's vendored CUDA Samples vector-math library, i.e.
> third-party code like Thrust or Kokkos, and only one of the two precision copies is compiled per
> build. Counting them inflated the polyhedral `Cuda` and `Slang-Cuda` aggregates by ~1 970 SLOC.

## 4. Plots

The combined charts (application efficiency, ꟼP over complexity, platform ranking, and ꟼP over
problem size) come from `ppbcc p3analysis` and need the code-complexity CSV from step 3:

```bash
CC=./code-complexity/code-complexity.csv
# N-body
ppbcc p3analysis combined $CC ./Results_* -n NBody \
  -c halstead-difficulty --log-complexity \
  --non-zero-pp -s avg --average-over efficiency -x "VerletLists|LinkedCells|Reduction" \
  --remove-description -l --export-to-csv --legend--vertical
# Polyhedral gravity model
ppbcc p3analysis combined $CC ./Results_* -n Polyhedral \
  -c halstead-difficulty --log-complexity \
  --non-zero-pp --remove-description -s avg --average-over efficiency -l --export-to-csv
# Matrix multiplication
ppbcc p3analysis combined $CC ./Results_* -n MatrixMultiplication \
  -c halstead-difficulty --log-complexity \
  --non-zero-pp --remove-description -s avg --average-over efficiency -x "Cublas" -l \
  --export-to-csv
# Vector addition
ppbcc p3analysis combined $CC ./Results_* -n VecAdd \
  -c halstead-difficulty --log-complexity \
  --non-zero-pp --remove-description -s avg --average-over efficiency -x "Cublas" -l \
  --export-to-csv
```

`ppbcc p3analysis` takes the chart first, then the code-complexity CSV, then the benchmark CSVs.
`-n` selects the problem and may only be omitted when the CSVs hold a single problem, which the
consolidated `Results_*` files do not.

`--average-over efficiency` computes ꟼP from the application efficiencies averaged over the
benchmark sizes, i.e. as the harmonic mean of exactly the efficiency panel on the left, instead of
averaging one ꟼP score per size (`--average-over pp`, the default the paper originally used). The
per-size heatmap is the same in both modes. The exported `average` rows of
`*_performance_portability.csv` follow the option.

Complexity is expressed as a percentage of the sequential C++ baseline by default, which keeps every
value positive and is what makes `--log-complexity` usable. `--complexity-metric-absolute` plots the
raw metric instead; ranking is unaffected either way, because the scaling is the same for every
paradigm.

> [!NOTE]
> The lower-right panel is a **heatmap** over the benchmark sizes, one row per implementation and
> one column per size, and no longer a line plot: with fourteen implementations the lines
> overlapped to the point of being unreadable. Its rows keep the descending-ꟼP order of the
> platform-ranking panel to its left, so the two lower panels line up row for row and the paradigm
> labels are not repeated.
>
> Its column labels use exponents whenever every benchmark size is an exact power of the same
> base: `2^5 … 2^14` for matrix multiplication and `10^1 … 10^8` for N-body and vector addition.
> That is short enough to sit horizontally within one column, at a size slightly above the cell
> values. The polyhedral meshes
> (2780, 12796, …) are no such sweep and keep decimal labels at a smaller size, because upright
> labels that long would otherwise grow the figure's bounding box.

### 4.1 SLOC against Halstead difficulty

`complexity-comparison` plots two complexity metrics against each other, one marker per
paradigm, with the identity line drawn in: above it the y metric charges a paradigm more than the
x metric does — *dense lines* — and below it the x metric charges more — *verbose code*. Both axes
are relative to the sequential C++ baseline, which is what makes the identity line meaningful, so
the chart rejects `--complexity-metric-absolute`.

```bash
CC=./code-complexity/code-complexity.csv
# Vector addition
ppbcc p3analysis complexity-comparison $CC ./Results_* -n VecAdd \
  -c halstead-difficulty --compare-metric sloc \
  --log-complexity --non-zero-pp -s avg -x "Cublas" --remove-description -l
# Matrix multiplication
ppbcc p3analysis complexity-comparison $CC ./Results_* -n MatrixMultiplication \
  -c halstead-difficulty --compare-metric sloc \
  --log-complexity --non-zero-pp -s avg -x "Cublas" --remove-description -l
# N-body
ppbcc p3analysis complexity-comparison $CC ./Results_* -n NBody \
  -c halstead-difficulty --compare-metric sloc \
  --log-complexity --non-zero-pp -s avg -x "VerletLists|LinkedCells|Reduction" \
  --remove-description -l
# Polyhedral gravity model
ppbcc p3analysis complexity-comparison $CC ./Results_* -n Polyhedral \
  -c halstead-difficulty --compare-metric sloc \
  --log-complexity --non-zero-pp -s avg --remove-description -l
```

`--compare-metric` names the x-axis metric and `-c/--complexity-metric` the y-axis one; both accept
the same names and aliases as everywhere else. `-l` suppresses the in-plot legend *and* the
per-point paradigm labels, because the paper keys these charts to the shared vertical legend of
step 4; drop it to get a self-contained chart with both.

A key hanging from half height on the right states the absolute sequential C++ values behind the
100 % of both axes, e.g. `SLOC = 715` and `D = 232.22` for N-body, so the figure can be read
without the running text. It grows downwards into the lower-right corner, which the markers leave
free. Spearman's rho and Kendall's tau are not drawn; they belong in the running text, where they
can be discussed.

### 4.2 Benchmark-only charts

Charts that need no complexity data come from `ppbcc p2analysis`, which takes the chart and the
benchmark CSVs only. The boxplots use `-s all` instead of `-s avg`:

```bash
ppbcc p2analysis boxplot ./Results_* -n NBody \
  --non-zero-pp -s all -x "VerletLists|LinkedCells|Reduction" --remove-description
ppbcc p2analysis boxplot ./Results_* -n Polyhedral \
  --non-zero-pp -s all --remove-description
ppbcc p2analysis boxplot ./Results_* -n MatrixMultiplication \
  --non-zero-pp -s all -x "Cublas" --remove-description
ppbcc p2analysis boxplot ./Results_* -n VecAdd \
  --non-zero-pp -s all -x "Cublas" --remove-description
```

`--export-to-csv` additionally writes the underlying application-efficiency and
performance-portability tables, so every quoted number can be checked against a CSV rather than
read off a plot. With `p3analysis` the tables also carry every complexity metric.

### 4.3 Runtimes

`time-barplot` draws the measured runtime itself, one bar per paradigm in the paradigm's color,
grouped by platform, on a logarithmic axis. It uses the largest benchmark size unless `-s` names
another one. `-t` selects the runtime column: `wall-clock` (default), `kernel`, `force-update` or
`neighbor-search`. Kernel Time is only recorded for matrix multiplication and vector addition, so
N-body and polyhedral gravity use the wall-clock time; asking for a column the results do not
contain is an error.

```bash
ppbcc p2analysis time-barplot ./Results_* -n MatrixMultiplication -t kernel \
  -x "Cublas" --remove-description -l
ppbcc p2analysis time-barplot ./Results_* -n VecAdd -t kernel \
  -x "Cublas" --remove-description -l
ppbcc p2analysis time-barplot ./Results_* -n NBody \
  -x "VerletLists|LinkedCells|Reduction" --remove-description -l
ppbcc p2analysis time-barplot ./Results_* -n Polyhedral \
  --remove-description -l --normalize-time-to-peak
```

With `--remove-description` the fastest variant stands for a paradigm. `-H "NVIDIA RTX5080"` restricts
the chart to one platform and labels the bars with their paradigm instead of grouping them. Slots
stay empty where a paradigm has no result on a platform. The Slang-Cuda vector addition reports a
Kernel Time of zero on every NVIDIA GPU; those rows are dropped with a warning because a log axis
cannot show them.

`--normalize-time-to-peak` multiplies every runtime by the published peak performance of its
platform, i.e. plots the FLOPs the platform could have executed in that time, which makes runtimes
on GPUs of different capability comparable. The peaks come from `ppbcc.hardware.PEAK_PERFORMANCE`,
which documents the source of every value:

| Platform | FP32 [TFLOP/s] | FP64 [TFLOP/s] | Source |
|---|---:|---:|---|
| NVIDIA RTX3080 (10 GB) | 29.77 | 0.465 | [Wikipedia, RTX 30 series](https://en.wikipedia.org/wiki/GeForce_RTX_30_series), boost clock |
| NVIDIA RTX4060 | 15.11 | 0.236 | [Wikipedia, RTX 40 series](https://en.wikipedia.org/wiki/GeForce_RTX_40_series), boost clock |
| NVIDIA RTX5080 | 56.3 | 0.88 | [Wikipedia, RTX 50 series](https://en.wikipedia.org/wiki/GeForce_RTX_50_series) |
| NVIDIA GH200 | 66.9 | 33.5 | [Wikipedia, Nvidia Tesla](https://en.wikipedia.org/wiki/Nvidia_Tesla), H100 SXM |
| AMD MI210 | 181.0 | 22.63 | [Wikipedia, AMD Instinct](https://en.wikipedia.org/wiki/AMD_Instinct), boost clock, *Vector TFLOPS* |
| Intel Max 1550 | 26 | 26 | [flopper.io](https://flopper.io/gpu/intel-data-center-gpu-max-1550-128gb), 52 for the card, halved for one stack |

These are datasheet values and not measured ceilings: the Nsight Compute ceilings of step 5 differ,
because Nsight Compute scales its `peak_sustained` counters with the clock the card actually ran at.
The MI210 figure is labelled *Vector TFLOPS* in its source and may count a different unit of work
than the NVIDIA per-core FMA figures.

```bash
ppbcc p2analysis time-barplot ./Results_* -n MatrixMultiplication -t kernel \
  -x "Cublas" --remove-description --normalize-time-to-peak -l
```

### 4.4 Heatmap of Application Efficiencies

`heatmap` shows the application efficiency at one problem size, one row per platform and one column
per paradigm, ordered by descending mean efficiency, or by name with `--sort-alphabetically`. The
color scale is fixed to [0, 1], so heatmaps of different sizes and problems are comparable.

```bash
ppbcc p2analysis heatmap ./Results_* -n "Polyhedral" -s 3145728.0 -o "polyhedral_large" --sort-alphabetically
ppbcc p2analysis heatmap ./Results_* -n "Polyhedral" -s 14744.0 -o "polyhedral_small" --sort-alphabetically
```

A paradigm that was never benchmarked on a platform, e.g. CUDA on the AMD MI210, is drawn as a
black cell with a `-` instead of an efficiency of `0.00`. A measured result is always positive, so
a `0.00` that remains is a real, if tiny, efficiency (the GH200 Vulkan results, for instance). This
heatmap is a separate plot from the size-scaling heatmap in the `combined` chart of step 4, which
is unchanged.

### 4.5 Double Heatmap of Application Efficiencies

`double-heatmap` compares two problem sizes in one figure. Every cell is split along its diagonal:
the upper-left triangle (◤) shows the size given with `-s`, the lower-right triangle (◢) the size
given with `--second-size`. Both sizes must be exact numeric sizes, and both halves share the
[0, 1] color scale. A half whose paradigm was never benchmarked on the platform at that size is
black with a `-`.

```bash
ppbcc p2analysis double-heatmap ./Results_* -n "Polyhedral" -s 14744.0 --second-size 3145728.0 \
  -o "polyhedral_small_large" --sort-alphabetically
```

Columns are ordered by the mean efficiency over both sizes, or by name with `--sort-alphabetically`. Without `-o` the plot is written to
`<problem>_double_heatmap.pdf`.


### 4.6 Rank correlation across problems

`rank-correlation` is the one `p3analysis` output that spans the problems instead of picking one:
it asks whether a problem's ordering of the paradigms predicts the next problem's, and writes
Spearman's rho between every pair of problems as a CSV matrix rather than a figure.
`--correlation` selects what is ranked — `pp` (default), or a complexity metric such as
`halstead-difficulty` or `sloc`. `-n` takes a comma-separated list and defaults to every problem in
the CSVs.

```bash
CC=./code-complexity/code-complexity.csv
# Performance portability, ranked as in step 4
ppbcc p3analysis rank-correlation $CC ./Results_* --correlation pp \
  --non-zero-pp -s avg --average-over efficiency \
  -x "Cublas|VerletLists|LinkedCells|Reduction" -o "rank_correlation_pp"
# Halstead difficulty
ppbcc p3analysis rank-correlation $CC ./Results_* --correlation halstead-difficulty \
  -x "Cublas|VerletLists|LinkedCells|Reduction" -o "rank_correlation_halstead"
ppbcc p3analysis rank-correlation $CC ./Results_* --correlation sloc \
  -x "Cublas|VerletLists|LinkedCells|Reduction" -o "rank_correlation_sloc"
```

The two matrices, rounded — ꟼP first, Halstead difficulty second:

| ꟼP | MatMul | NBody | Polyhedral | VecAdd |
|---|---:|---:|---:|---:|
| MatrixMultiplication | 1.00 | 0.56 | 0.64 | 0.33 |
| NBody | 0.56 | 1.00 | 0.73 | 0.15 |
| Polyhedral | 0.64 | 0.73 | 1.00 | 0.01 |
| VecAdd | 0.33 | 0.15 | 0.01 | 1.00 |

| D | MatMul | NBody | Polyhedral | VecAdd |
|---|---:|---:|---:|---:|
| MatrixMultiplication | 1.00 | 0.88 | 0.79 | 0.93 |
| NBody | 0.88 | 1.00 | 0.91 | 0.89 |
| Polyhedral | 0.79 | 0.91 | 1.00 | 0.89 |
| VecAdd | 0.93 | 0.89 | 0.89 | 1.00 |

Complexity orderings agree across problems (0.79-0.93) far more than ꟼP orderings do. Among the
performance rankings, matrix multiplication does carry information (0.56 and 0.64), while vector
addition predicts nothing: it is the workload where Alpaka ranks 2nd and Kokkos 11th, and its rho
against the polyhedral gravity model is 0.01.

The ꟼP run takes the same options as step 4 — `-s`, `--average-over`, `--non-zero-pp`,
`-p/--precision` and the description filters all apply, so the ranking is the one the `combined`
charts show; the per-problem `-x` exclusions of step 4 are combined into one regex here. Complexity
metrics are read unscaled, since dividing a problem's paradigms by its own C++ baseline cannot
change that problem's order.

Both variables are reduced to one value per **paradigm**, because that is what the problems have in
common: the best variant stands for its paradigm in ꟼP, variants are reduced to their median for a
complexity metric, and only paradigms with benchmark results are ranked — so the C++ baseline and
complexity-only frameworks such as `Metal`, `Thrust` and `Cublas` stay out, and both runs above rank
the same fourteen paradigms. `-e/--export-to-csv` writes those per-paradigm values and their ranks
(best first) to `<prefix>_ranks.csv`. Having no figure, the chart rejects `-l/--legend`,
`--remove-description` (which it always implies) and `--log-complexity`.


## 5. Roofline models

The profiling reports, the consolidated profiling CSVs and the roofline PDFs are not part of this
repository; they are attached to the Zenodo release as the folder `profiling/`. The commands below
regenerate them. They are run from the repository root, with `D` pointing at that folder:

```bash
D=$PWD/results/profiling
```

### 5.1 Profile

The roofline models need a **separate build**: a profiler replays every kernel launch it sees, so the
executables must be reduced to one input and one iteration first. That is what the CMake option
`PPB_PROFILING` does, and the two `*-profiling` presets switch it on. It also names the work with
NVTX ranges (`matmul`, `init`/`evaluate`, `positions`/`forces`/`velocities`), which every row of the
profiling CSVs carries as its region.

```bash
cmake --preset cuda-llvm-profiling  && cmake --build build-cuda-llvm-profiling
# Only for OpenACC and Stdpar (they need the NVHPC toolchain)
cmake --preset cuda-nvhpc-profiling && cmake --build build-cuda-nvhpc-profiling
```

We profiled with two NVIDIA tools, driven by `ppbcc profile`:

| `--profiler` | Tool | Covers | One row is |
|---|---|---|---|
| `ncu` (default) | Nsight Compute | every paradigm with a CUDA context | one kernel launch, counted |
| `nsys` | Nsight Systems GPU metrics | OpenCL, Boost.Compute, Vulkan, Slang-Vulkan | one marked region, sampled |

`ncu` reads its counters through CUPTI, which needs a current CUDA context; OpenCL and Vulkan build
their own and are invisible to it. `nsys` samples the GPU device-wide and therefore sees them, but it
reports pipe utilisations rather than instruction counts: the FLOP count of its rows has to be
supplied with `--analytic-flop`, and both ceilings with `--peak-performance`/`--peak-bandwidth`
(take the ones `ncu` measured on the same machine).

> [!NOTE]
> `ppbcc profile` also offers `--profiler likwid` (LIKWID NvMarker, needs a build with
> `-DPPB_ENABLE_LIKWID=ON`). It is not **100% tested** and not used for any published result.

**Matrix multiplication and polyhedral gravity, `ncu`:**

```bash
ppbcc profile -b build-cuda-llvm-profiling -p src -r "matMul_.*" "polyhedral_.*" \
  -x ".*_cpp$" ".*_ocl$" ".*_boost$" ".*_vulkan$" -d $D/ncu-roofline-rtx5080 -H "NVIDIA RTX5080" \
  -O ncu-arg=--set=roofline --no-csv --timeout 900
ppbcc profile -b build-cuda-nvhpc-profiling -p src \
  -r "matMul_acc$" "matMul_stdpar$" "polyhedral_acc$" "polyhedral_stdpar$" \
  -d $D/ncu-roofline-rtx5080 -H "NVIDIA RTX5080" -O ncu-arg=--set=roofline --no-csv --timeout 900
```

Matrix multiplication at 16384² is the same with build directories configured with
`-DPPB_PROFILING_MATMUL_SIZE=16384`, only `matMul_*` and `-d $D/ncu-roofline-rtx5080-matmul16384`.

**Polyhedral gravity, `nsys`:**

```bash
ppbcc profile --profiler nsys -b build-cuda-llvm-profiling -p src \
  -r "polyhedral_ocl$" "polyhedral_boost$" "polyhedral_vulkan$" "polyhedral_slang_vulkan$" \
  -d $D/nsys-rtx5080 -H "NVIDIA RTX5080" --timeout 900 -O iterations=20 \
  --peak-performance 5.737e13 --peak-bandwidth 9.607e11 \
  --analytic-flop 'polyhedral_.*\[evaluate\]=1.84705e9' \
  -o $D/Profiling_NSYS_NVIDIA_RTX5080
```

The analytic count is the median of the `evaluate` FLOPs `ncu` counted for the CUDA-context
paradigms. Every executable is profiled twice (one and `iterations` iterations), and the difference
cancels the one-time setup.

**N-body (naive), `ncu` and `nsys`:**

```bash
ppbcc profile -b build-cuda-llvm-profiling -p src \
  -r "nbody_cuda$" "nbody_kokkos$" "nbody_raja$" "nbody_slang_cuda_naive$" "nbody_acpp$" \
  -d $D/ncu-roofline-rtx5080 -H "NVIDIA RTX5080" -O ncu-arg=--set=roofline \
  -O ncu-arg=--launch-skip=6 -O ncu-arg=--launch-count=12 --no-csv --timeout 1800
ppbcc profile --profiler nsys -b build-cuda-llvm-profiling -p src -r "nbody_ocl$" "nbody_boost$" \
  -d $D/nsys-rtx5080 -H "NVIDIA RTX5080" --timeout 1800 -O iterations=2 -O frequency=10000 \
  -O activity-ratio=0.05 --peak-performance 5.741e13 --peak-bandwidth 9.613e11 \
  --analytic-flop 'nbody_.*\[forces\]=2.799972e11' \
  -o $D/Profiling_NSYS_nbody_NVIDIA_RTX5080
```

`ncu` only collects four complete time steps after a warm-up step, since all 1000 time steps of the
profiling run launch the same kernels. `-O activity-ratio=0.05` effectively disables the filter that
separates compute-shader data movement from the kernel: the `forces` region contains nothing but the
force kernel.

> [!NOTE]
> `ppbcc profile` launches the binaries through `setarch <arch> -R`, which stops Google Benchmark
> from re-executing itself under the profiler; `ncu` otherwise hangs on `matMul_kokkos` and
> `matMul_omp`. `polyhedral_stdpar` aborts under `ncu` after its report is written, so the collection
> run logs it as failed — the consolidation below still reads its report.

### 5.2 Consolidate the reports into CSVs

`--skip-profile` runs nothing and only parses the reports already in `--report-dir` (relative to
`-b`), next to which each run leaves the Google Benchmark `.json` that supplies paradigm and
precision:

```bash
cd results/profiling
ppbcc profile -b . -d ncu-roofline-rtx5080 -r "matMul_.*" "polyhedral_.*" --skip-profile \
  -H "NVIDIA RTX5080" -o Profiling_NCU_NVIDIA_RTX5080
ppbcc profile -b . -d ncu-roofline-rtx5080-matmul16384 -r "matMul_.*" --skip-profile \
  -H "NVIDIA RTX5080" -o Profiling_NCU_matmul16384_NVIDIA_RTX5080
ppbcc profile -b . -d ncu-roofline-rtx5080 -r "nbody_.*" --skip-profile \
  -H "NVIDIA RTX5080" -o Profiling_NCU_nbody_NVIDIA_RTX5080
```

The `nsys` runs above already write their CSVs. To re-consolidate them with `--profiler nsys
--skip-profile` (and the same `--analytic-flop` and peaks), `ppbcc` reads the `.sqlite` exports,
which have to be recreated first:

```bash
for f in nsys-rtx5080/*.nsys-rep; do nsys export --type sqlite -o "${f%.nsys-rep}.sqlite" "$f"; done
```

`Profiling_ANALYTIC_{ncu,nsys}_nbody_NVIDIA_RTX5080.csv` are the `forces` rows of the two N-body
tables with the FLOP column replaced by the analytic count of the kernel, 22 FLOP per interaction,
i.e. `22·N·(N−1) = 2.199978e11` at N = 10⁵, so that every implementation is credited with the same
work.

### 5.3 Roofline plots

`--from-csv` skips discovery and profiling and draws one roofline from several profiling CSVs, which
is how the `ncu` and `nsys` tables become one chart. `--region` restricts the plot to the kernel under
study:

```bash
# Polyhedral gravity, all paradigms (ncu + nsys), and ncu only
ppbcc profile --from-csv Profiling_NCU_NVIDIA_RTX5080.csv Profiling_NSYS_NVIDIA_RTX5080.csv \
  -r "polyhedral_" -H "NVIDIA RTX5080" --no-csv --region "^evaluate$" \
  --roofline polyhedral_roofline_NVIDIA_RTX5080.pdf
ppbcc profile --from-csv Profiling_NCU_NVIDIA_RTX5080.csv \
  -r "polyhedral_" -H "NVIDIA RTX5080" --no-csv --region "^evaluate$" \
  --roofline polyhedral_roofline_ncu_NVIDIA_RTX5080.pdf

# Matrix multiplication, ncu (matMul_cuda keeps its cuBLAS and naive regions apart)
ppbcc profile --from-csv Profiling_NCU_NVIDIA_RTX5080.csv \
  -r "matMul_" -H "NVIDIA RTX5080" --no-csv \
  --roofline matmul_roofline_ncu_NVIDIA_RTX5080.pdf
ppbcc profile --from-csv Profiling_NCU_matmul16384_NVIDIA_RTX5080.csv \
  -r "matMul_" -H "NVIDIA RTX5080" --no-csv \
  --roofline matmul16384_roofline_ncu_NVIDIA_RTX5080.pdf

# N-body force kernel: analytic FLOP count for every row, and each tool's own count
ppbcc profile --from-csv Profiling_ANALYTIC_ncu_nbody_NVIDIA_RTX5080.csv \
  Profiling_ANALYTIC_nsys_nbody_NVIDIA_RTX5080.csv \
  -r "nbody_" -H "NVIDIA RTX5080" --no-csv --region "^forces$" \
  --roofline nbody_forces_roofline_NVIDIA_RTX5080.pdf
ppbcc profile --from-csv Profiling_NCU_nbody_NVIDIA_RTX5080.csv \
  Profiling_NSYS_nbody_NVIDIA_RTX5080.csv \
  -r "nbody_" -H "NVIDIA RTX5080" --no-csv --region "^forces$" \
  --roofline nbody_forces_roofline_counted_NVIDIA_RTX5080.pdf
```

One point is one implementation and region: FLOP, duration and memory traffic are summed over all
its launches (or sampled windows) before the two ratios are taken (`-a sum`, the default). Use
`-a dominant` for the longest kernel only, `-a none` for one point per launch, `-m l2`/`-m l1` to move
the arithmetic intensity to another level of the memory hierarchy, and `-l` to drop the legend. The
`ncu` ceilings are measured (`peak_sustained` scaled with the actual clock); the `nsys` ones are the
values passed on the command line.
