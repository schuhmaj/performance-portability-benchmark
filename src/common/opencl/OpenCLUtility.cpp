#include "OpenCLUtility.h"

namespace opencl_utility {

    std::vector<cl_platform_id> getPlatforms() {
        // Get number of platforms
        cl_uint numPlatforms;
        cl_int err = clGetPlatformIDs(0, nullptr, &numPlatforms);
        if (err != CL_SUCCESS) {
            throw std::runtime_error("Failed to find any OpenCL platforms.");
        }

        // Get platform IDs
        std::vector<cl_platform_id> platforms{numPlatforms};
        err = clGetPlatformIDs(numPlatforms, platforms.data(), nullptr);
        if (err != CL_SUCCESS) {
            throw std::runtime_error("Failed to get OpenCL platform IDs.");
        }
        return platforms;
    }

    std::vector<cl_device_id> getDevices(const cl_platform_id &platformId, const cl_device_type &type) {
        // Get number of devices in the platform
        cl_uint numDevices;
        cl_int err = clGetDeviceIDs(platformId, type, 0, nullptr, &numDevices);
        if (err != CL_SUCCESS) {
            throw std::runtime_error("Failed to find any devices on platform!");
        }

        // Get device IDs
        std::vector<cl_device_id> devices(numDevices);
        err = clGetDeviceIDs(platformId, type, numDevices, devices.data(), nullptr);
        if (err != CL_SUCCESS) {
            throw std::runtime_error("Failed to get device IDs on platform!");
        }
        return devices;
    }

    std::string getDeviceName(const cl_device_id &deviceId) {
        char deviceName[1024];
        cl_int err = clGetDeviceInfo(deviceId, CL_DEVICE_NAME, sizeof(deviceName), deviceName, nullptr);
        if (err != CL_SUCCESS) {
            throw std::runtime_error("Failed to get device name for device");
        }
        return std::string{deviceName};
    }

    cl_device_id getFirstGPU() {
        // Scan every platform for a GPU instead of assuming platform[0] exposes
        // one. Systems with the Intel oneAPI runtime advertise several OpenCL
        // platforms (CPU runtime, FPGA emulation, GPU runtime) in an arbitrary
        // order, so the GPU is frequently not the first platform.
        std::string available; // human-readable listing for the error message
        for (const cl_platform_id &platform : getPlatforms()) {
            cl_uint numDevices = 0;
            // CL_DEVICE_NOT_FOUND here just means "no GPU on this platform";
            // skip it rather than treating it as a fatal error.
            const cl_int err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, nullptr, &numDevices);

            char platformName[256] = "<unknown>";
            clGetPlatformInfo(platform, CL_PLATFORM_NAME, sizeof(platformName), platformName, nullptr);
            available += std::string("  - ") + platformName + ": " +
                         std::to_string(err == CL_SUCCESS ? numDevices : 0) + " GPU device(s)\n";

            if (err != CL_SUCCESS || numDevices == 0) {
                continue;
            }

            std::vector<cl_device_id> devices(numDevices);
            if (clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, numDevices, devices.data(), nullptr) == CL_SUCCESS &&
                !devices.empty()) {
                return devices.front();
            }
        }
        throw std::runtime_error("No OpenCL GPU device found on any platform. Detected platforms:\n" + available);
    }

} // namespace opencl_utility
