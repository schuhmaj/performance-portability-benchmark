#pragma once

#include <fstream>
#include <iostream>
#include <filesystem>
#include <string>
#include <exception>
#include <utility>
#include <vector>
#include "Particle.h"

namespace ppb {

/**
 * A helper class for reading and writing collections of Particle objects to and from CSV files.
 *
 * This class handles parsing and formatting of particle data, supporting both serialization and deserialization.
 * Template parameter FloatType determines the floating point precision used for particle properties.
 *
 * @tparam FloatType Type of floating point used for positions, velocities, and forces.
 */
template <typename FloatType>
class CSVFileHandler {

    /**
     * The path to the CSV file managed by this handler.
     */
    const std::string _filename;

public:

    /**
     * Constructs a CSVFileHandler with the provided filename.
     *
     * @param filename The file path to use for reading and writing CSV data.
     */
    explicit CSVFileHandler(std::string filename) : _filename{std::move(filename)} {

    }

    /**
     * Writes the given collection of particles to the CSV file.
     *
     * The file is overwritten if it already exists. The header and all particle fields
     * are written in CSV format.
     *
     * @param particles The collection of particles to write.
     * @throws std::runtime_error If the file cannot be opened for writing.
     */
    void write(const std::vector<Particle<FloatType>>& particles) {
        std::ofstream outFile{_filename};

        if (!outFile.is_open()) {
            throw std::runtime_error("Can't open file for writing");
        }
        outFile << "type,posX,posY,posZ,velX,velY,velZ,forceX,forceY,forceZ\n";

        // Write particle data
        for (const auto& p : particles) {
            outFile << p.getType() << ","
                    << p.getPosition()[0] << "," << p.getPosition()[1] << "," << p.getPosition()[2] << ","
                    << p.getVelocity()[0] << "," << p.getVelocity()[1] << "," << p.getVelocity()[2] << ","
                    << p.getForce()[0] << "," << p.getForce()[1] << "," << p.getForce()[2] << "\n";
        }

        outFile.close();
    }

    /**
     * Reads particles from the associated CSV file.
     *
     * The file must exist and must be formatted with the compatible header and field ordering.
     * Each row after the header line is parsed as a particle and added to the result vector.
     *
     * @return A vector of Particle objects parsed from the CSV file.
     * @throws std::runtime_error If the file does not exist or cannot be opened, or if the CSV format is invalid.
     */
    std::vector<Particle<FloatType>> read() {
        if (!std::filesystem::exists(_filename)) {
            throw std::runtime_error("File doesn't exist");
        }
        std::vector<Particle<FloatType>> particles;
        std::ifstream inFile{_filename};
        if (!inFile.is_open()) {
            throw std::runtime_error("Can't open file");
        }
        std::string line;

        // Skip first line
        std::getline(inFile, line);
        while (std::getline(inFile, line)) {
            std::istringstream iss{line};
            std::string token;
            int type = getNextToken<int>(iss);
            std::array<FloatType, 3> position{getNextToken<FloatType>(iss), getNextToken<FloatType>(iss), getNextToken<FloatType>(iss)};
            std::array<FloatType, 3> velocity{getNextToken<FloatType>(iss), getNextToken<FloatType>(iss), getNextToken<FloatType>(iss)};
            std::array<FloatType, 3> force{getNextToken<FloatType>(iss), getNextToken<FloatType>(iss), getNextToken<FloatType>(iss)};
            particles.emplace_back(position, velocity, force, type);
        }
        return particles;
    }

private:

    /**
     * Extracts the next value of specified type, delimited by commas, from the provided stream.
     *
     * This helper is used for parsing CSV fields, converting the string token to the requested type.
     *
     * @tparam T The type to convert the field to.
     * @param iss Input string stream to read the next field from.
     * @return The parsed value of type T from the next CSV field.
     * @throws std::runtime_error If the next token cannot be read, indicating invalid CSV formatting.
     */
    template<typename T>
    T getNextToken(std::istringstream& iss) {
        T result;
        std::string element;
        if (!std::getline(iss, element, ',')) {
            throw std::runtime_error("Invalid CSV format");
        }
        std::istringstream conv{element};
        conv >> result;
        return result;
    }
};

} // namespace ppb