#ifdef PPB_ENABLE_VTK
#include "VTKWriter.cuh"

//Code taken from https://github.com/MaxieHelenBichmann/MolSim-WS25-GroupD/blob/main/src/io/outputWriter/VTKWriter.cpp (last accessed 04.06.2026)
#include <vtkCellArray.h>
#include <vtkDoubleArray.h>
#include <vtkFloatArray.h>
#include <vtkIntArray.h>
#include <vtkPointData.h>
#include <vtkXMLUnstructuredGridWriter.h>

#include <iomanip>
#include <sstream>

using namespace ppb;

template<typename FloatType>
void VTKWriter::plotParticles(std::vector<Particle<FloatType>>& particles, const std::string& filename, int iteration) const {
    // Initialize points
    auto points = vtkSmartPointer<vtkPoints>::New();

    // Create and configure data arrays
    vtkNew<vtkFloatArray> mass_array;
    mass_array->SetName("mass");
    mass_array->SetNumberOfComponents(1);

    vtkNew<vtkFloatArray> velocity_array;
    velocity_array->SetName("velocity");
    velocity_array->SetNumberOfComponents(3);

    vtkNew<vtkFloatArray> force_array;
    force_array->SetName("force");
    force_array->SetNumberOfComponents(3);

    vtkNew<vtkIntArray> type_array;
    type_array->SetName("type");
    type_array->SetNumberOfComponents(1);

    for (auto& p : particles) {
        points->InsertNextPoint(p.getPosition());
        mass_array->InsertNextValue(static_cast<float>(p.getM()));
        velocity_array->InsertNextTuple(p.getVelocity());
        force_array->InsertNextTuple(p.getForce());
        type_array->InsertNextValue(p.getType());
    }

    // Set up the grid
    auto grid = vtkSmartPointer<vtkUnstructuredGrid>::New();
    grid->SetPoints(points);

    // Add arrays to the grid
    grid->GetPointData()->AddArray(mass_array);
    grid->GetPointData()->AddArray(velocity_array);
    grid->GetPointData()->AddArray(force_array);
    grid->GetPointData()->AddArray(type_array);

    // Create filename with iteration number
    std::stringstream strstr;
    strstr << filename << "_" << std::setfill('0') << std::setw(4) << iteration << ".vtu";

    // Create writer and set data
    vtkNew<vtkXMLUnstructuredGridWriter> writer;
    writer->SetFileName(strstr.str().c_str());
    writer->SetInputData(grid);
    writer->SetDataModeToBinary();

    // Write the file
    writer->Write();

}

#endif