#include "VulkanUtility.h"

#include <iostream>


vulkan_utility::VulkanManager::VulkanManager(bool enableValidationLayers) :
    instancePtr{createInstance(enableValidationLayers)}, physicalDevicePtr{createPhysicalDevice(instancePtr)},
    computeQueueFamilyIndex{findComputeQueueFamilyIndex(*physicalDevicePtr).value()},
    devicePtr{createLogicalDevice(physicalDevicePtr, computeQueueFamilyIndex)} {}

std::shared_ptr<kp::Sequence> vulkan_utility::VulkanManager::sequence() {
    return std::make_shared<kp::Sequence>(
        physicalDevicePtr, devicePtr, createComputeQueue(devicePtr, computeQueueFamilyIndex), computeQueueFamilyIndex);
}

std::shared_ptr<vk::Instance> vulkan_utility::VulkanManager::createInstance(bool enableValidationLayers) {
    // Query validation layer availability and add the KHRONOS_validation
    // (which is a combination out of several useful logs/ validators for shaders - no need to manually enable them individually)
    std::vector<const char *> layers;
    if (enableValidationLayers) {
        constexpr char *kValidationLayer = "VK_LAYER_KHRONOS_validation";
        const auto availableLayers = vk::enumerateInstanceLayerProperties();
        for (const auto &lp : availableLayers) {
            if (std::string(lp.layerName.data()) == kValidationLayer) {
                layers.push_back(kValidationLayer);
            }
        }
    }

    // Application Information
    vk::ApplicationInfo appInfo{};
    appInfo.pApplicationName = "PPB";
    appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.pEngineName = "PPB";
    appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.apiVersion = VK_API_VERSION_1_1;

    // Instance extensions (handle portability on macOS/MoltenVK)
    std::vector<const char *> instExtensions;
    std::vector<vk::ExtensionProperties> instExtProps = vk::enumerateInstanceExtensionProperties();
    if (std::any_of(instExtProps.begin(), instExtProps.end(), [](const vk::ExtensionProperties &prop) {
            return std::string(prop.extensionName.data()) == "VK_KHR_portability_enumeration";
        })) {
        instExtensions.push_back("VK_KHR_portability_enumeration");
    }

    // Actual Instance Creation
    vk::InstanceCreateInfo instanceCI{};
    instanceCI.pApplicationInfo = &appInfo;
    instanceCI.enabledLayerCount = static_cast<uint32_t>(layers.size());
    instanceCI.ppEnabledLayerNames = layers.empty() ? nullptr : layers.data();
    instanceCI.enabledExtensionCount = static_cast<uint32_t>(instExtensions.size());
    instanceCI.ppEnabledExtensionNames = instExtensions.empty() ? nullptr : instExtensions.data();
    if (!instExtensions.empty()) {
        instanceCI.flags |= vk::InstanceCreateFlagBits::eEnumeratePortabilityKHR;
    }
    vk::Instance instance = vk::createInstance(instanceCI);
    return std::shared_ptr<vk::Instance>(new vk::Instance(instance), [](vk::Instance *p) {
        if (p && *p) {
            p->destroy();
        }
        delete p;
    });
}

std::shared_ptr<vk::PhysicalDevice>
vulkan_utility::VulkanManager::createPhysicalDevice(const std::shared_ptr<vk::Instance> &instancePtr) {
    // Get the available physical Devices (e.g. RTX 2080 or Apple M1 pro)
    const auto physicalDevices = instancePtr->enumeratePhysicalDevices();
    if (physicalDevices.empty()) {
        throw std::runtime_error("No Vulkan physical devices found.");
    }

    // Filter for the first device which supports general compute capabilities
    std::optional<vk::PhysicalDevice> chosenPD{std::nullopt};
    for (const auto &pd : physicalDevices) {
        const std::optional<uint32_t> queueFamilyIndex = findComputeQueueFamilyIndex(pd);
        if (queueFamilyIndex.has_value()) {
            chosenPD = pd;
            break;
        }
    }
    if (!chosenPD.has_value()) {
        throw std::runtime_error("No compute-capable queue family found.");
    }

    // Return this physical device
    return std::shared_ptr<vk::PhysicalDevice>(new vk::PhysicalDevice(chosenPD.value()),
                                               [](vk::PhysicalDevice *p) { delete p; });
}

std::shared_ptr<vk::Device>
vulkan_utility::VulkanManager::createLogicalDevice(const std::shared_ptr<vk::PhysicalDevice> &physicalDevicePtr,
                                                   uint32_t computeQueueFamilyIndex) {
    // Create Logical device with one Compute queue
    float queuePriority = 1.0f;
    vk::DeviceQueueCreateInfo dqci{};
    dqci.queueFamilyIndex = findComputeQueueFamilyIndex(*physicalDevicePtr).value();
    dqci.queueCount = 1;
    dqci.pQueuePriorities = &queuePriority;

    // Optional Device Features
    vk::DeviceCreateInfo dci{};
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos = &dqci;

    // Device extensions (handle portability on macOS/MoltenVK)
    std::vector<const char *> devExtensions;
    std::vector<vk::ExtensionProperties> devExtProps = physicalDevicePtr->enumerateDeviceExtensionProperties();
    if (std::any_of(devExtProps.begin(), devExtProps.end(), [](const vk::ExtensionProperties &prop) {
            return std::string(prop.extensionName.data()) == "VK_KHR_portability_subset";
        })) {
        devExtensions.push_back("VK_KHR_portability_subset");
    }
    if (!devExtensions.empty()) {
        dci.enabledExtensionCount = static_cast<uint32_t>(devExtensions.size());
        dci.ppEnabledExtensionNames = devExtensions.data();
    }

    // Creation of the actual logical device and wrapping into a smart pointer (with custom Vulkan Cleanup Call)
    vk::Device device = physicalDevicePtr->createDevice(dci);
    return std::shared_ptr<vk::Device>(new vk::Device(device), [](vk::Device *p) {
        if (p && *p) {
            p->destroy();
        }
        delete p;
    });
}

std::shared_ptr<vk::Queue>
vulkan_utility::VulkanManager::createComputeQueue(const std::shared_ptr<vk::Device> &devicePtr,
                                                  uint32_t computeQueueFamilyIndex) {
    // Create a new compute queue, given the logical device and its compute pipline
    std::shared_ptr<vk::Queue> computeQueue = std::make_shared<vk::Queue>();
    devicePtr->getQueue(computeQueueFamilyIndex, 0, computeQueue.get());
    return computeQueue;
}

std::optional<uint32_t>
vulkan_utility::VulkanManager::findComputeQueueFamilyIndex(const vk::PhysicalDevice &physicalDevice) {
    const std::vector<vk::QueueFamilyProperties> queueFamilies = physicalDevice.getQueueFamilyProperties();
    for (uint32_t i = 0; i < queueFamilies.size(); ++i) {
        if (queueFamilies[i].queueCount > 0 && (queueFamilies[i].queueFlags & vk::QueueFlagBits::eCompute)) {
            return i;
        }
    }
    return std::nullopt;
}
