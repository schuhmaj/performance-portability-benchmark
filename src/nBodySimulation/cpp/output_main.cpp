#include "NBodySimulation.h"
#include "CSVFileHandler.h"
// Bad Style, but functional
#include "Impl_Cpp.cpp"


int main() {
    using namespace ppb;
    NBodySimulation<float> nbodySimulation{100};
    nbodySimulation();
    CSVFileHandler<float> file{"cpp_end.csv"};
    file.write(nbodySimulation.getParticles());
    return 0;
}