#include <algorithm>
#include <benchmark/benchmark.h>
#include <chrono>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <vector>
#include <vulkan/vulkan.hpp>
#include "VectorAdditionShader.h"
#include "kompute/Kompute.hpp"
#include "vectorAdditon/VectorAddition.h"

namespace ppb {
    template <typename FloatType>
    struct ImplVulkan {
        using float_type = FloatType;

        std::pair<std::vector<FloatType>, double> operator()(const std::vector<FloatType> &a, const std::vector<FloatType> &b) {
            const unsigned int size = a.size();
            std::vector<FloatType> result(size, 0.0);

            // Step 1: Create Vulkan Instance (optionally enable validation layer if available)
            std::shared_ptr<vk::Instance> instancePtr;
            std::shared_ptr<vk::PhysicalDevice> physicalDevicePtr;
            std::shared_ptr<vk::Device> devicePtr;
            try {
                // Query validation layer availability
                bool enableValidation = false;
                const char* kValidationLayer = "VK_LAYER_KHRONOS_validation";
                auto availableLayers = vk::enumerateInstanceLayerProperties();
                for (const auto &lp : availableLayers) {
                    if (std::string(lp.layerName.data()) == kValidationLayer) {
                        enableValidation = true;
                        break;
                    }
                }

                vk::ApplicationInfo appInfo{};
                appInfo.pApplicationName = "PPB-VecAdd";
                appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
                appInfo.pEngineName = "PPB";
                appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
                appInfo.apiVersion = VK_API_VERSION_1_1;

                std::vector<const char*> layers;
                // if (enableValidation) {
                //     layers.push_back(kValidationLayer);
                // }

                // Instance extensions (handle portability on macOS/MoltenVK)
                std::vector<vk::ExtensionProperties> instExtProps = vk::enumerateInstanceExtensionProperties();
                std::vector<const char*> instExtensions;
                auto hasInstExt = [&](const char* name){
                    for (const auto& ep : instExtProps) { if (std::string(ep.extensionName.data()) == name) return true; } return false;
                };
                if (hasInstExt("VK_KHR_portability_enumeration")) {
                    instExtensions.push_back("VK_KHR_portability_enumeration");
                }

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
                instancePtr = std::shared_ptr<vk::Instance>(new vk::Instance(instance), [](vk::Instance* p){ if (p && *p) { p->destroy(); } delete p; });

                // Step 2: Select a physical device with compute capability
                auto physicalDevices = instancePtr->enumeratePhysicalDevices();
                if (physicalDevices.empty()) {
                    throw std::runtime_error("No Vulkan physical devices found.");
                }

                std::optional<uint32_t> computeQueueFamilyIndex;
                vk::PhysicalDevice chosenPD;
                for (const auto &pd : physicalDevices) {
                    auto queueFamilies = pd.getQueueFamilyProperties();
                    for (uint32_t i = 0; i < queueFamilies.size(); ++i) {
                        if (queueFamilies[i].queueCount > 0 && (queueFamilies[i].queueFlags & vk::QueueFlagBits::eCompute)) {
                            computeQueueFamilyIndex = i;
                            chosenPD = pd;
                            break;
                        }
                    }
                    if (computeQueueFamilyIndex.has_value()) break;
                }
                if (!computeQueueFamilyIndex.has_value()) {
                    throw std::runtime_error("No compute-capable queue family found.");
                }

                physicalDevicePtr = std::shared_ptr<vk::PhysicalDevice>(new vk::PhysicalDevice(chosenPD), [](vk::PhysicalDevice* p){ delete p; });

                // Step 3: Create logical device with one compute queue
                float queuePriority = 1.0f;
                vk::DeviceQueueCreateInfo dqci{};
                dqci.queueFamilyIndex = computeQueueFamilyIndex.value();
                dqci.queueCount = 1;
                dqci.pQueuePriorities = &queuePriority;

                // Enable optional device features if needed (none for simple add)
                vk::DeviceCreateInfo dci{};
                dci.queueCreateInfoCount = 1;
                dci.pQueueCreateInfos = &dqci;

                // Device extensions (handle portability on macOS/MoltenVK)
                std::vector<vk::ExtensionProperties> devExtProps = chosenPD.enumerateDeviceExtensionProperties();
                std::vector<const char*> devExtensions;
                auto hasDevExt = [&](const char* name){
                    for (const auto& ep : devExtProps) { if (std::string(ep.extensionName.data()) == name) return true; } return false;
                };
                if (hasDevExt("VK_KHR_portability_subset")) {
                    devExtensions.push_back("VK_KHR_portability_subset");
                }
                if (!devExtensions.empty()) {
                    dci.enabledExtensionCount = static_cast<uint32_t>(devExtensions.size());
                    dci.ppEnabledExtensionNames = devExtensions.data();
                }

                vk::Device device = chosenPD.createDevice(dci);
                devicePtr = std::shared_ptr<vk::Device>(new vk::Device(device), [](vk::Device* p){ if (p && *p) { p->destroy(); } delete p; });

                // Step 4: Initialize Kompute Manager with explicit Vulkan objects
                kp::Manager mgr(instancePtr, physicalDevicePtr, devicePtr);

                auto tensorA = mgr.tensor(a);
                auto tensorB = mgr.tensor(b);
                std::shared_ptr<kp::TensorT<float>> tensorC = mgr.tensor(result);

                std::vector<std::shared_ptr<kp::Tensor>> params = {tensorA, tensorB, tensorC};
                std::vector<uint32_t> shader(VECTORADDITIONSHADER_COMP_SPV.begin(), VECTORADDITIONSHADER_COMP_SPV.end());
                kp::Workgroup workgroup{{size, 1, 1}};

                auto algorithm = mgr.algorithm(
                    params,
                    shader,
                    workgroup
                );

                // Manually construct a Kompute Sequence using our explicit queue
                std::shared_ptr<vk::Queue> computeQueue = std::make_shared<vk::Queue>();
                devicePtr->getQueue(computeQueueFamilyIndex.value(), 0, computeQueue.get());
                auto sequence = std::make_shared<kp::Sequence>(
                    physicalDevicePtr,
                    devicePtr,
                    computeQueue,
                    computeQueueFamilyIndex.value(),
                    0
                );
                sequence->template record<kp::OpTensorSyncDevice>(params)->eval();

                const auto start = std::chrono::high_resolution_clock::now();

                sequence->template record<kp::OpAlgoDispatch>(algorithm)->eval();

                const auto end = std::chrono::high_resolution_clock::now();
                const double elapsed_seconds = std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count();

                sequence->template record<kp::OpTensorSyncLocal>(params)->eval();
                result = tensorC->vector();
                return std::make_pair(result, elapsed_seconds);
            } catch (const std::exception &ex) {
                std::cerr << "Vulkan/Kompute initialization or execution failed: " << ex.what() << std::endl;
                return std::make_pair(result, 0.0);
            }
        }
    };

    template class ImplVulkan<float>;
};

BENCHMARK(ppb::VectorAddition<ppb::ImplVulkan<float>>::benchmark)
    ->Name("VecAdd-CStd-Float")
    ->RangeMultiplier(10)
    ->Range(1e3, 1e8)
#ifdef PPB_MEASURE_ONLY_KERNEL
    ->UseManualTime()
#endif
    ->Complexity();

int main(int argc, char **argv) {
    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
}
