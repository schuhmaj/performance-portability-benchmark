# Reproducibility

## Convert Results and Generate CSV files

Collect the `*.json` files with the benchmark script and generate a `*.csv` by:

```bash
../scripts/benchmark.py -p ./nvidia-rtx3080 -r ".*\.json" --skip-benchmark -H "NVIDIA RTX3080" -o "Results_NVIDIA_RTX3080"
../scripts/benchmark.py -p ./nvidia-rtx4060 -r ".*\.json" --skip-benchmark -H "NVIDIA RTX4060" -o "Results_NVIDIA_RTX4060"
../scripts/benchmark.py -p ./nvidia-rtx5080 -r ".*.json" --skip-benchmark -H "NVIDIA RTX5080" -o "Results_NVIDIA_RTX5080"
../scripts/benchmark.py -p ./nvidia-gh200 -r ".*.json" --skip-benchmark -H "NVIDIA GH200" -o "Results_NVIDIA_GH200"
../scripts/benchmark.py -p ./amd-mi210 -r ".*.json" --skip-benchmark -H "AMD Instinct MI210" -o "Results_AMD_Instinct_MI210"
../scripts/benchmark.py -p ./intel-data_center_gpu_max_1550 -r ".*.json" --skip-benchmark -H "INTEL Data Center GPU Max 1550" -o "Results_INTEL_Data_Center_GPU_Max_1550"
```

## Generate Plots

To generate the plots, use the following commands:
```bash
# For NBody Plots
ppbcc p3analysis NBody ./Results_* --complexity ./code-complexity/code-complexity.csv -c combined \
  --complexity-metric halstead-difficulty --additive --log-size \
  --non-zero-pp -s avg -x "VerletLists|LinkedCells|Reduction" --remove-description -l --export-to-csv
# For Polyhedral Plots
ppbcc p3analysis Polyhedral ./Results_* --complexity ./code-complexity/code-complexity.csv -c combined \
  --complexity-metric halstead-difficulty --additive --log-size \
  --non-zero-pp --remove-description -s avg -l --export-to-csv
# For MatrixMultiplication Plots (no --log-complexity: --additive yields
# non-positive complexity values for this problem)
ppbcc p3analysis MatrixMultiplication ./Results_* --complexity ./code-complexity/code-complexity.csv -c combined \
  --complexity-metric halstead-difficulty --additive --log-size \
  --non-zero-pp --remove-description -s avg -x "Cublas" -l --export-to-csv
# For vector Addition Plots
ppbcc p3analysis VecAdd ./Results_* --complexity ./code-complexity/code-complexity.csv -c combined \
  --complexity-metric halstead-difficulty --additive --log-size \
  --non-zero-pp --remove-description -s avg -x "Cublas" -l --export-to-csv
```

```bash
# For NBody Plots
ppbcc p3analysis NBody ./Results_* -c boxplot \
  --non-zero-pp -s all -x "VerletLists|LinkedCells|Reduction" --remove-description
# For Polyhedral Plots
ppbcc p3analysis Polyhedral ./Results_* -c boxplot \
  --non-zero-pp --remove-description -s all
# For MatrixMultiplication Plots (no --log-complexity: --additive yields
# non-positive complexity values for this problem)
ppbcc p3analysis MatrixMultiplication ./Results_* -c boxplot \
  --non-zero-pp --remove-description -s all -x "Cublas"
# For vector Addition Plots
ppbcc p3analysis VecAdd ./Results_* -c boxplot \
  --non-zero-pp --remove-description -s all -x "Cublas"
```

## Generate Code-Complexity results

The plot commands above require `results/code-complexity/code-complexity.csv`.
Both it and the file-level CSV are (re-)generated from the repository root with:

```bash
python scripts/generate_code_complexity.py
```

The file-level results correspond to:

```bash
ppbcc code-complexity src \
  --output results/code-complexity/code-complexity-files.csv
```

Each implementation aggregate is generated with the same form, using the
explicit source manifest in `scripts/generate_code_complexity.py`:

```bash
ppbcc code-complexity <implementation-and-shared-source-files...> \
  --dialect <dialect-list> --aggregate \
  --metrics sloc distinct_operators distinct_operands total_operators total_operands \
  --output <temporary-csv>
```

The `TOTAL` rows are collected into the aggregate CSV, with
`distinct_operators`, `distinct_operands`, `total_operators`, and
`total_operands` renamed to the conventional `n1`, `n2`, `N1`, and `N2`
columns. To print every concrete `python -m code_complexity` invocation,
including all resolved shared files, without changing the results, run:

```bash
python scripts/generate_code_complexity.py --dry-run
```

The manifest separates the AdaptiveCpp shared-memory, CUDA matrix, OpenMP
host/device, Vulkan matrix, and Kokkos NBody reduction variants. NBody Naive,
LinkedCells, and VerletLists are separate rows. Slang shaders are deliberately
aggregated twice: with CUDA host code for `Slang-Cuda` and with Vulkan host
code for `Slang-Vulkan`. OpenCL kernels reused by Boost.Compute are included
in the Boost aggregates as well. Quoted repository-local includes are resolved
transitively, and implementation units for required utilities from
`src/common` are added automatically. The generator rejects dependencies on a
different benchmark problem, which prevents unrelated application code from
silently inflating an aggregate.
