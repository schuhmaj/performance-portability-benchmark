#include "NBodySimulation.h"
#include "Kokkos_Core.hpp"
#include <iostream>


int main(int argc, char** argv) {
    Kokkos::ScopeGuard guard{argc, argv};

    std::cout << "Default Execution Space: " << Kokkos::DefaultExecutionSpace::name() << std::endl;

    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
}