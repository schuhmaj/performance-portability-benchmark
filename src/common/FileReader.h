#pragma once

#include <string>

namespace ppb::FileReader {


    /**
     * Reads all lines of a given file to a string.
     * @param fileName - the path of the file
     * @return the content of the file as string
     *
     * @throws std::runtime_error if the file does not exists or an I/O error happens
     */
    std::string read(const std::string &fileName);

} // namespace ppb::FileReader