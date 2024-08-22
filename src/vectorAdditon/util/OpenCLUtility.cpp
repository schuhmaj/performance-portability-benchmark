#include "OpenCLUtlity.h"

namespace util {

    std::vector<cl_platform_id> getOpenCLPlattforms() {
        // Get number of platforms
        cl_uint numPlatforms;
        cl_int err = clGetPlatformIDs(0, nullptr, &numPlatforms);
        if(err != CL_SUCCESS) {
            throw std::runtime_error("Failed to find any OpenCL platforms.");
        }

        // Get platform IDs
        std::vector<cl_platform_id> platforms{numPlatforms};
        err = clGetPlatformIDs(numPlatforms, platforms.data(), nullptr);
        if(err != CL_SUCCESS) {
            throw std::runtime_error( "Failed to get OpenCL platform IDs.");
        }
        return platforms;
    }

    std::vector<cl_device_id> getOpenCLDevices(const cl_platform_id& platformId, const cl_device_type& type) {
        // Get number of devices in the platform
        cl_uint numDevices;
        cl_int err = clGetDeviceIDs(platformId, type, 0, nullptr, &numDevices);
        if(err != CL_SUCCESS) {
            throw std::runtime_error("Failed to find any devices on platform!");
        }

        // Get device IDs
        std::vector<cl_device_id> devices(numDevices);
        err = clGetDeviceIDs(platformId, type, numDevices, devices.data(), nullptr);
        if(err != CL_SUCCESS) {
            throw std::runtime_error("Failed to get device IDs on platform!");
        }
        return devices;
    }

    std::string getDeviceName(const cl_device_id &deviceId) {
        char deviceName[1024];
        cl_int err = clGetDeviceInfo(deviceId, CL_DEVICE_NAME, sizeof(deviceName), deviceName, nullptr);
        if(err != CL_SUCCESS) {
            throw std::runtime_error("Failed to get device name for device");
        }
        return std::string{deviceName};
    }

}