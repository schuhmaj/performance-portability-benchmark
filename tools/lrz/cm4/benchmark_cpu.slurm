#!/bin/bash
#SBATCH -J "Performance Portability CPU Benchmark"
#SBATCH -D /dss/dsshome1/0F/ge25zov2/Programming/performance-portability-benchmark/build-cpu
#SBATCH -o log_%j.out
#SBATCH -e log_%j.err
#SBATCH --get-user-env
#SBATCH --mail-user=jonas.schuhmacher@tum.de
#SBATCH --mail-type=ALL
#SBATCH --clusters=serial
#SBATCH --partition=serial_std
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1


module load slurm_setup
module load gcc/15.2.0 python/3.14.0
source /dss/dsshome1/0F/ge25zov2/venv-standard-math/bin/activate
python ../scripts/benchmark.py -p ./src -r "matMul_.*" "vec_.*" "nbody_.*" "polyhedral_.*" -H "INTEL Ice Lake (CPU)"
python3 /home/schuhmaj/performance-portability-benchmark/scripts/plot_benchmark.py -p ./src -r "matMul_*"
