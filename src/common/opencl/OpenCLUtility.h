#pragma once

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#include <exception>
#include <stdexcept>
#include <string>
#include <vector>

namespace opencl_utilities {
    /**
     * Returns all available opencl_utilities platforms of the computer.
     * @return vector of platform ids
     */
    std::vector<cl_platform_id> getPlatforms();

    /**
     * Returns all available opencl_utilities devices for a given platform of the computer. Further, this method allows
     * to specify the type of the returned devices
     * @param platformId the platform id for which to return devices
     * @param type the type of the devices to return, e.g. CL_DEVICE_TYPE_ALL or CL_DEVICE_TYPE_GPU
     * @return vector of device ids
     */
    std::vector<cl_device_id> getDevices(const cl_platform_id &platformId,
                                               const cl_device_type &type = CL_DEVICE_TYPE_ALL);

    /**
     * Resolveds a device id to the humand readable name.
     * @param deviceId the id of a deivce
     * @return string containing the name of the device
     */
    std::string getDeviceName(const cl_device_id &deviceId);


    /**
     * Returns the first GPU which can be found on the system.
     * @return cl_device_id of the first GPU
     */
    cl_device_id getFirstGPU();
} // namespace opencl_utilities
