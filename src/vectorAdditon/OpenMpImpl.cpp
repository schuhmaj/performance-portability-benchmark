#include <algorithm>
#include <benchmark/benchmark.h>
#include "VectorAddition.h"
#include "omp.h"


template <typename FloatType>
std::vector<FloatType> VectorAddition<FloatType>::operator()() {
    const size_t size = _inA.size();
    FloatType *a = _inA.data();
    FloatType *b = _inB.data();
    FloatType *c = _outC.data();
#pragma omp target parallel for map(to : a[0 : size], b[0 : size]) map(from : c[0 : size])
    for (size_t i = 0; i < size; ++i) {
        c[i] = a[i] + b[i];
    }
    checkValidity();
    return _outC;
}

template std::vector<float> VectorAddition<float>::operator()();
BENCHMARK(VectorAddition<float>::vectorAdditionBenchmark)
    ->Name("VecAdd-CStd-Float")
    ->RangeMultiplier(10)
    ->Range(1e3, 1e8)
    ->Complexity();

int main(int argc, char **argv) {
    int num_devices = omp_get_num_devices();
    printf("Number of available devices %d\n", num_devices);

#pragma omp target
    {
        if (omp_is_initial_device()) {
            printf("Running on host\n");
        }
        else {
            int nteams = omp_get_num_teams();
            int nthreads = omp_get_num_threads();
            printf("Running on device with %d teams in total and %d threads in each team\n", nteams, nthreads);
        }
    }
    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
}
