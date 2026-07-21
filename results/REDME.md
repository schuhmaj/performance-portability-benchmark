# Reproducibility

## Convert Results and Generate CSV files

Collect the `*.json` files with the benchmark script and generate a `*.csv` by:

```bash
../scripts/benchmark.py -p ./nvidia-rtx5080 -r ".*.json" --skip-benchmark -H "NVIDIA RTX5080" -o Results_NVIDIA_RTX5080
```

## Generate Plots

To generate the plots, use the following commands:

```bash
# For NBody Plots
../scripts/p3_analysis.py -n NBody --complexity ./code-complexity/code-complexity.csv -c combined ./Results_* \
  --complexity-metric halstead-difficulty --additive --log-size \
  --non-zero-pp -s 100000 -x "VerletLists|LinkedCells|Reduction" --remove-description -l
# For Polyhedral Plots
../scripts/p3_analysis.py -n Polyhedral --complexity ./code-complexity/code-complexity.csv -c combined ./Results_* \
  --complexity-metric halstead-difficulty --additive --log-size \
  --non-zero-pp --remove-description -s 3145728 -l
# For MatrixMultiplication Plots (no --log-complexity: --additive yields
# non-positive complexity values for this problem)
../scripts/p3_analysis.py -n MatrixMultiplication --complexity ./code-complexity/code-complexity.csv -c combined ./Results_* \
  --complexity-metric halstead-difficulty --additive --log-size \
  --non-zero-pp --remove-description -s 8192 -l
# For vector Addition Plots
../scripts/p3_analysis.py -n VecAdd --complexity ./code-complexity/code-complexity.csv -c combined ./Results_* \
  --complexity-metric halstead-difficulty --additive --log-size \
  --non-zero-pp --remove-description -s 100000000 -l
```

## Generate Code-Complexity results

The plot commands above require `results/code-complexity/code-complexity.csv`.
Both it and the file-level CSV are (re-)generated from the repository root with:

```bash
python scripts/generate_code_complexity.py
```

The file-level results correspond to:

```bash
python -m code_complexity src \
  --output results/code-complexity/code-complexity-files.csv
```

Each implementation aggregate is generated with the same form, using the
explicit source manifest in `scripts/generate_code_complexity.py`:

```bash
python -m code_complexity <implementation-and-shared-source-files...> \
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
`src/common` are added automatically.
