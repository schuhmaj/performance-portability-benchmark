#pragma once
#ifdef PPB_ENABLE_VTK

#include <vtkSmartPointer.h>
#include <vtkUnstructuredGrid.h>
#include "Impl_Cuda.cuh"

namespace ppb {
    class VTKWriter {
    public: 
        VTKWriter() = default;
        ~VTKWriter() override = default;

        VTKWriter(const VTKWriter&) = delete;
        VTKWriter& operator=(const VTKWriter&) = delete;

        void plotParticles(CudaParticleSoA& particles, const std::string& filename, int iteration);
    }

} // namespace ppb

#endif