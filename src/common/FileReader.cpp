#include "FileReader.h"

#include <filesystem>
#include <exception>
#include <stdexcept>
#include <fstream>
#include <sstream>


std::string read(const std::string &fileName) {
    // Chek if the file exists
    if (!std::filesystem::exists(fileName)) {
        throw std::runtime_error("File does not exist");
    }
    std::ifstream file{fileName};
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file");
    }
    // Read everything into content
    std::stringstream buffer{};
    buffer << file.rdbuf();
    return buffer.str();
}