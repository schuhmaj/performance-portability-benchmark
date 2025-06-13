#pragma once

#include <fstream>
#include <iostream>
#include <filesystem>
#include <string>
#include <exception>
#include <utility>
#include <vector>
#include <format>
#include "Particle.h"

namespace ppb {

    template <typename FloatType>
    class CSVFileHandler {

        const std::string _filename;

    public:

        explicit CSVFileHandler(std::string filename) : _filename{std::move(filename)} {

        }

        void write(const std::vector<Particle<FloatType>>& particles) {
            std::ofstream outFile{_filename};

            if (!outFile.is_open()) {
                throw std::runtime_error(std::format("Failed to open file {} for writing.", _filename));
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

        std::vector<Particle<FloatType>> read() {
            if (!std::filesystem::exists(_filename)) {
                throw std::runtime_error(std::format("File {} does not exist.", _filename));
            }
            std::vector<Particle<FloatType>> particles;
            std::ifstream inFile{_filename};
            if (!inFile.is_open()) {
                throw std::runtime_error(std::format("Failed to open file {} for reading.", _filename));
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