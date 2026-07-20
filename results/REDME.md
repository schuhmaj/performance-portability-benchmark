# Convert Results

Collect the `*.json` files with the benchmark script and generate a `*.csv` by:

```bash
./scripts/benchmark.py -p ./nvidia-rtx5080 -r ".*.json" --skip-benchmark -H "NVIDIA RTX5080" -o Results_NVIDIA_RTX5080
```

To generate the plots, use the following commands:

```bash
# For NBody Plots
../scripts/p3_analysis.py -n NBody --complexity ./code-complexity/code-complexity.csv -c combined ./Results_* \ 
  --complexity-metric halstead-difficulty --additive --log-size --log-complexity \ 
  --non-zero-pp -s 100000 -x "VerletLists|LinkedCells|Reduction" --remove-description
# For Polyhedral Plots
../scripts/p3_analysis.py -n Polyhedral --complexity ./code-complexity/code-complexity.csv -c combined ./Results_* \ 
  --complexity-metric halstead-difficulty --additive --log-size --log-complexity \ 
  --non-zero-pp --remove-description -s 3145728
# For MatrixMultiplication Plots
../scripts/p3_analysis.py -n MatrixMultiplication --complexity ./code-complexity/code-complexity.csv -c combined ./Results_* \ 
  --complexity-metric halstead-difficulty --additive --log-size --log-complexity \ 
  --non-zero-pp --remove-description -s 8192
# For vector Addition Plots
../scripts/p3_analysis.py -n VecAdd --complexity ./code-complexity/code-complexity.csv -c combined ./Results_* \ 
  --complexity-metric halstead-difficulty --additive --log-size --log-complexity \ 
  --non-zero-pp --remove-description -s 100000000
```