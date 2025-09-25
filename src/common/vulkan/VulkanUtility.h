#pragma once
#include <algorithm>
#include <optional>
#include <memory>
#include <kompute/Kompute.hpp>
#include <vulkan/vulkan.h>

namespace vulkan_utility {

    /**
     * VulkanManager is a thin utility around Vulkan and Kompute that performs a manual Vulkan setup
     * and exposes a minimal API for creating Kompute tensors, algorithms and sequences.
     *
     * Why this class exists and why we don't use kp::Manager directly:
     * - On macOS, compatibility issues with MoltenVK require a manual and explicit Vulkan instance
     *   and device setup. Using kp::Manager can implicitly manage these objects in ways that conflict
     *   with MoltenVK on certain configurations. By handling the instance, physical device selection,
     *   logical device and compute queue creation ourselves, we ensure consistent behavior across
     *   platforms, particularly on macOS with MoltenVK.
     * - This class centralizes that setup so the rest of the code can remain clean and portable.
     */
    class VulkanManager {

        std::shared_ptr<vk::Instance> instancePtr;
        std::shared_ptr<vk::PhysicalDevice> physicalDevicePtr;
        uint32_t computeQueueFamilyIndex;
        std::shared_ptr<vk::Device> devicePtr;

    public:

        /**
         * Constructs a VulkanManager and performs the Vulkan initialization steps
         * (instance, physical device, logical device, and compute queue selection).
         *
         * @param enableValidationLayers whether to enable Vulkan validation layers
         */
        explicit VulkanManager(bool enableValidationLayers = false);

        /**
         * Creates a Kompute tensor backed by this manager's Vulkan device.
         *
         * @tparam T element type of the tensor
         * @param data initial tensor data
         * @return shared pointer to a Kompute tensor
         */
        template<typename T>
        std::shared_ptr<kp::TensorT<T>> tensor(const std::vector<T> &data) {
            return std::make_shared<kp::TensorT<T>>(physicalDevicePtr, devicePtr, data);
        }

        /**
         * Creates a Kompute algorithm for the given tensors and SPIR-V shader code.
         *
         * @param params list of tensor parameters
         * @param shader SPIR-V shader binary as a uint32_t vector
         * @param workgroup workgroup configuration (x, y, z)
         * @return shared pointer to a Kompute algorithm
         */
        std::shared_ptr<kp::Algorithm> algorithm(const std::vector<std::shared_ptr<kp::Tensor>> &params, const std::vector<uint32_t> &shader, const kp::Workgroup &workgroup);

        /**
         * Creates an empty Kompute sequence that can record and run operations.
         *
         * @return shared pointer to a Kompute sequence
         */
        std::shared_ptr<kp::Sequence> sequence();

    private:

        /**
         * Creates a Vulkan instance.
         * @param enableValidationLayers whether to enable validation layers
         * @return shared pointer to the created instance
         */
        static std::shared_ptr<vk::Instance> createInstance(bool enableValidationLayers);

        /**
         * Picks a suitable physical device from the provided instance.
         * @param instancePtr Vulkan instance
         * @return shared pointer to the selected physical device
         */
        static std::shared_ptr<vk::PhysicalDevice> createPhysicalDevice(const std::shared_ptr<vk::Instance> &instancePtr);

        /**
         * Creates a logical device for compute operations.
         * @param physicalDevicePtr selected physical device
         * @param computeQueueFamilyIndex index of the compute-capable queue family
         * @return shared pointer to the logical device
         */
        static std::shared_ptr<vk::Device> createLogicalDevice(const std::shared_ptr<vk::PhysicalDevice> &physicalDevicePtr, uint32_t computeQueueFamilyIndex);

        /**
         * Retrieves the compute queue from the logical device.
         * @param devicePtr logical device
         * @param computeQueueFamilyIndex index of the compute-capable queue family
         * @return shared pointer to the compute queue
         */
        static std::shared_ptr<vk::Queue> createComputeQueue(const std::shared_ptr<vk::Device> &devicePtr, uint32_t computeQueueFamilyIndex);

        /**
         * Finds a queue family index that supports compute operations.
         * @param physicalDevice physical device to query
         * @return index of compute queue family, or empty if none was found
         */
        static std::optional<uint32_t> findComputeQueueFamilyIndex(const vk::PhysicalDevice& physicalDevice);

    };

} // vulkan_utility
