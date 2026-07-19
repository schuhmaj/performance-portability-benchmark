# Convert Results

Collect the `*.json` files with the benchmark script and generate a `*.csv` by:

```bash
./scripts/benchmark.py -p ./nvidia-rtx5080 -r ".*.json" --skip-benchmark -H "NVIDIA RTX5080" -o Results_NVIDIA_RTX5080
```