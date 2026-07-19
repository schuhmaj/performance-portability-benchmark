#!/bin/bash
#SBATCH -p lrz-hgx-h100-94x4
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --mail-user=jonas.schuhmacher@tum.de
#SBATCH --mail-type=ALL
#SBATCH -o log_%j.out
#SBATCH -e log_%j.err
#SBATCH --container-image=/dss/dsshome1/0B/ge25zov/enroot/nvhpc-dev.sqsh
#SBATCH --container-mounts=/dss/dsshome1/0B/ge25zov/performance-portability-benchmark:/home/schuhmaj/performance-portability-benchmark,/dss/dsshome1/0B/ge25zov/tmp/.acpp:/home/schuhmaj/.acpp,/dss/dsshome1/0B/ge25zov/tmp/.cache:/home/schuhmaj/.cache


cd /home/schuhmaj/performance-portability-benchmark
module use /home/schuhmaj/modules
module load llvm/20.1.8 google-benchmark/1.9.4 AdaptiveCpp/25.10.0
rm -rf build-cuda-llvm
cmake --preset cuda-llvm
cd build-cuda-llvm
cmake --build .
python3 ../scripts/benchmark.py -p ./src -r "vec_.*" "matMul_.*" "polyhedral_.*" "nbody_.*" -x ".*_cpp" -H "NVIDIA H100"
