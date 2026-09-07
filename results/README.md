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
| `Profiling_<platform>.csv` | Step 5 — per-kernel Nsight Compute counters and roofline quantities |
| `polyhedral_roofline_<platform>.pdf` | Step 5 — the roofline model built from that CSV |

Step 4 renders the plots into the working directory; they are not committed here. The `.ncu-rep`
reports behind the profiling CSV are not committed either — step 5 regenerates them in about a
minute.

## 1. Run the benchmarks

From the build directory of the platform under test:

```bash
ppbcc benchmark -p src -H "INTEL Data Center GPU Max 1550" \
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
| `./amd-mi210` | `AMD Instinct MI210` | `Results_AMD_Instinct_MI210` |
| `./intel-data_center_gpu_max_1550` | `INTEL Data Center GPU Max 1550` | `Results_INTEL_Data_Center_GPU_Max_1550` |

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

The combined charts (application efficiency, ꟼP over problem size, and ꟼP over complexity)
need the code-complexity CSV from step 3:

```bash
# N-body
ppbcc p3analysis NBody ./Results_* --complexity ./code-complexity/code-complexity.csv -c combined \
  --complexity-metric halstead-difficulty --additive --log-size \
  --non-zero-pp -s avg -x "VerletLists|LinkedCells|Reduction" --remove-description -l \
  --export-to-csv --legend--vertical
# Polyhedral gravity model
ppbcc p3analysis Polyhedral ./Results_* --complexity ./code-complexity/code-complexity.csv -c combined \
  --complexity-metric halstead-difficulty --additive --log-size \
  --non-zero-pp --remove-description -s avg -l --export-to-csv
# Matrix multiplication
ppbcc p3analysis MatrixMultiplication ./Results_* --complexity ./code-complexity/code-complexity.csv -c combined \
  --complexity-metric halstead-difficulty --additive --log-size \
  --non-zero-pp --remove-description -s avg -x "Cublas" -l --export-to-csv
# Vector addition
ppbcc p3analysis VecAdd ./Results_* --complexity ./code-complexity/code-complexity.csv -c combined \
  --complexity-metric halstead-difficulty --additive --log-size \
  --non-zero-pp --remove-description -s avg -x "Cublas" -l --export-to-csv
```

> [!NOTE]
> `--log-complexity` is deliberately omitted: with `--additive`, matrix multiplication yields
> non-positive complexity values, which a logarithmic axis cannot show.

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

`ppbcc profile` then drives `ncu` over the resulting binaries. Both builds write their reports into
one shared folder (`--report-dir` is relative to `--build-dir`), so the two runs can be consolidated
together:

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
  --roofline --roofline-output results/polyhedral_roofline_NVIDIA_RTX5080.pdf \
  -k "init_lock_arrays" -k "query_cuda_kernel_arch"
```

The two `-k` patterns drop Kokkos' architecture query and desul's lock-array initialization from the
plot: the runtime launches them once at startup, they compute nothing, and they would distort the
per-implementation aggregate. They stay in the CSV.

> [!NOTE]
> Nsight Compute only sees **CUDA** kernels. `polyhedral_ocl`, `polyhedral_boost`,
> `polyhedral_vulkan`, `polyhedral_slang_vulkan` and `polyhedral_cpp` therefore produce no report
> and are skipped with a warning. `polyhedral_stdpar` is missing for an unrelated reason: NVHPC 26.5
> fails to compile it (`NVVM_ERROR_COMPILATION` in `static_thread_pool.hpp`), with or without
> `PPB_PROFILING`.

The plot places one point per implementation — work, traffic and kernel time summed over all its
launches — and uses the same paradigm colors as the `p3analysis` charts of step 4. Both roofs are
*measured*: every `peak_sustained` counter is scaled with the clock the corresponding unit actually
ran at, which on the RTX 5080 gives 57.4 TFLOP/s (FP32) and 959 GB/s. Use `-a dominant` for the
longest kernel only, `-a none` for one point per launch, and `-m l2`/`-m l1` to move the arithmetic
intensity to another level of the memory hierarchy.
