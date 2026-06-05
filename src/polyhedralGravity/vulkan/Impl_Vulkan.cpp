#include "polyhedralGravity/PolyhedralGravityDefinitions.h"
#include <glm/glm.hpp>
#include <vulkan/vulkan.hpp>
#include <vulkan/vulkan_raii.hpp>

#include "init.h"
#include "eval.h"

const uint32_t vulkan_init_len = sizeof(INIT_SPRIV);
const uint32_t kompute_eval_len = sizeof(EVAL_SPIRV);

// uint32_t kompute_eval[] = #include "shader/eval.hpp";

const uint32_t workgroupSize = 32;

GlobalResources::GlobalResources(int &argc, char *argv[]) {
}

GlobalResources::~GlobalResources() = default;

#if FLOAT_BITS == 32
using VectorType = glm::vec3;
#elif FLOAT_BITS == 64
using VectorType = glm::dvec3;
#else
#error "Invliad float bits size"
#endif

struct PushConstants {
    VectorType point;
    glm::uint num_faces;
};

class GravityEvaluable : public GravityEvaluableBase {
public:
    GravityEvaluable(
            const std::vector<Array3> &Vertices,
            const std::vector<IndexArray3> &Faces,
            const double density)
        : GravityEvaluableBase(Vertices, Faces, density), _vk_context(), _instance(nullptr), _device(nullptr),
          _bufferVertices(nullptr),
          _bufferFaces(nullptr),
          _bufferNormals(nullptr),
          _bufferSegmentVectors(nullptr),
          _bufferSegmentNormals(nullptr), _bufferResultPotential(nullptr), _bufferResultAcceleration(nullptr), _bufferResults(nullptr),
          _memoryVertices(nullptr),
          _memoryFaces(nullptr),
          _memoryNormals(nullptr),
          _memorySegmentVectors(nullptr),
          _memorySegmentNormals(nullptr), _memoryResultPotential(nullptr), _memoryResultAcceleration(nullptr), _memoryResults(nullptr),
          _descriptorSetLayout(nullptr),
          _pipelineLayout(nullptr),
          _pipelineCache(nullptr),
          _descriptorPool(nullptr),
          _commandPool(nullptr),
          _shaderModuleInit(nullptr),
          _shaderModuleEval(nullptr),
          _pipelineInit(nullptr), _pipelineEval(nullptr),
          _queue(nullptr),
          _fence(nullptr) {
        vk::ApplicationInfo AppInfo{
                "VulkanCompute",  // Application Name
                1,                // Application Version
                nullptr,          // Engine Name or nullptr
                0,                // Engine Version
                VK_API_VERSION_1_1// Vulkan API version
        };

        const std::vector<const char *> Layers = {
                // "VK_LAYER_KHRONOS_validation"
        };
        vk::InstanceCreateInfo InstanceCreateInfo(
                vk::InstanceCreateFlags(),// Flags
                &AppInfo,                 // Application Info
                Layers,                   // Layers
                {}                        // Extensions
        );
        _instance = vk::raii::Instance(_vk_context, InstanceCreateInfo);

        // for (auto &d: _instance.enumeratePhysicalDevices()) {
        //     auto DeviceProps = d.getProperties();
        //     std::cout << "Device Name    : " << DeviceProps.deviceName << std::endl;
        // }

        int device_index = 0;
        if (_instance.enumeratePhysicalDevices().size() > 2) {
            device_index = 2;
        }
        // std::cout << "using device index: " << device_index << std::endl;

        vk::raii::PhysicalDevice PhysicalDevice = _instance.enumeratePhysicalDevices().at(device_index);
        vk::PhysicalDeviceProperties DeviceProps = PhysicalDevice.getProperties();

        // std::cout << "Device Name    : " << DeviceProps.deviceName << std::endl;
        const uint32_t ApiVersion = DeviceProps.apiVersion;
        // std::cout << "Vulkan Version : " << VK_VERSION_MAJOR(ApiVersion) << "." << VK_VERSION_MINOR(ApiVersion) << "." << VK_VERSION_PATCH(ApiVersion) << std::endl;

        std::vector<vk::QueueFamilyProperties> QueueFamilyProps = PhysicalDevice.getQueueFamilyProperties();
        auto PropIt = std::find_if(QueueFamilyProps.begin(), QueueFamilyProps.end(),
                                   [](const vk::QueueFamilyProperties &Prop) {
                                       return Prop.queueFlags & vk::QueueFlagBits::eCompute;
                                   });
        const uint32_t ComputeQueueFamilyIndex = std::distance(QueueFamilyProps.begin(), PropIt);
        // std::cout << "Compute Queue Family Index: " << ComputeQueueFamilyIndex << std::endl;

        const float QueuePriority = 1.0f;
        vk::DeviceQueueCreateInfo DeviceQueueCreateInfo(vk::DeviceQueueCreateFlags(),// Flags
                                                        ComputeQueueFamilyIndex,     // Queue Family Index
                                                        1,                           // Number of Queues
                                                        &QueuePriority);

        vk::PhysicalDeviceFeatures requestedFeatures = {};
        // requestedFeatures.shaderFloat64 = VK_TRUE;

        vk::DeviceCreateInfo DeviceCreateInfo(vk::DeviceCreateFlags(),// Flags
                                              DeviceQueueCreateInfo); // Device Queue Create Info struct
        DeviceCreateInfo.pEnabledFeatures = &requestedFeatures;

        _device = PhysicalDevice.createDevice(DeviceCreateInfo);

        vk::BufferCreateInfo BufferCreateInfoVertices{
                vk::BufferCreateFlags(),                 // Flags
                _vertices.size() * sizeof(FloatType) * 4,// Size
                vk::BufferUsageFlagBits::eStorageBuffer, // Usage
                vk::SharingMode::eExclusive,             // Sharing mode
                1,                                       // Number of queue family indices
                &ComputeQueueFamilyIndex                 // List of queue family indices
        };

        vk::BufferCreateInfo BufferCreateInfoFaces{
                vk::BufferCreateFlags(),                 // Flags
                _faces.size() * sizeof(unsigned int) * 4,// Size
                vk::BufferUsageFlagBits::eStorageBuffer, // Usage
                vk::SharingMode::eExclusive,             // Sharing mode
                1,                                       // Number of queue family indices
                &ComputeQueueFamilyIndex                 // List of queue family indices
        };

        vk::BufferCreateInfo BufferCreateInfoNormals{
                vk::BufferCreateFlags(),                // Flags
                _faces.size() * sizeof(FloatType) * 4,  // Size
                vk::BufferUsageFlagBits::eStorageBuffer,// Usage
                vk::SharingMode::eExclusive,            // Sharing mode
                1,                                      // Number of queue family indices
                &ComputeQueueFamilyIndex                // List of queue family indices
        };

        vk::BufferCreateInfo BufferCreateInfoSegmentVectors{
                vk::BufferCreateFlags(),                // Flags
                _faces.size() * sizeof(FloatType) * 12, // Size
                vk::BufferUsageFlagBits::eStorageBuffer,// Usage
                vk::SharingMode::eExclusive,            // Sharing mode
                1,                                      // Number of queue family indices
                &ComputeQueueFamilyIndex                // List of queue family indices
        };

        vk::BufferCreateInfo BufferCreateInfoSegmentNormals{
                vk::BufferCreateFlags(),                // Flags
                _faces.size() * sizeof(FloatType) * 12, // Size
                vk::BufferUsageFlagBits::eStorageBuffer,// Usage
                vk::SharingMode::eExclusive,            // Sharing mode
                1,                                      // Number of queue family indices
                &ComputeQueueFamilyIndex                // List of queue family indices
        };

        vk::BufferCreateInfo BufferCreateInfoResultPotential{
                vk::BufferCreateFlags(),                // Flags
                sizeof(FloatType),                      // Size
                vk::BufferUsageFlagBits::eStorageBuffer,// Usage
                vk::SharingMode::eExclusive,            // Sharing mode
                1,                                      // Number of queue family indices
                &ComputeQueueFamilyIndex                // List of queue family indices
        };

        vk::BufferCreateInfo BufferCreateInfoResultAcceleration{
                vk::BufferCreateFlags(),                // Flags
                sizeof(VectorType),                     // Size
                vk::BufferUsageFlagBits::eStorageBuffer,// Usage
                vk::SharingMode::eExclusive,            // Sharing mode
                1,                                      // Number of queue family indices
                &ComputeQueueFamilyIndex                // List of queue family indices
        };

        vk::BufferCreateInfo BufferCreateInfoResults{
                vk::BufferCreateFlags(),                // Flags
                6 * sizeof(FloatType),                  // Size
                vk::BufferUsageFlagBits::eStorageBuffer,// Usage
                vk::SharingMode::eExclusive,            // Sharing mode
                1,                                      // Number of queue family indices
                &ComputeQueueFamilyIndex                // List of queue family indices
        };

        _bufferVertices = _device.createBuffer(BufferCreateInfoVertices);
        _bufferFaces = _device.createBuffer(BufferCreateInfoFaces);
        _bufferNormals = _device.createBuffer(BufferCreateInfoNormals);
        _bufferSegmentVectors = _device.createBuffer(BufferCreateInfoSegmentVectors);
        _bufferSegmentNormals = _device.createBuffer(BufferCreateInfoSegmentNormals);
        _bufferResultPotential = _device.createBuffer(BufferCreateInfoResultPotential);
        _bufferResultAcceleration = _device.createBuffer(BufferCreateInfoResultAcceleration);
        _bufferResults = _device.createBuffer(BufferCreateInfoResults);

        vk::MemoryRequirements MemoryRequirementsVertices = _bufferVertices.getMemoryRequirements();
        vk::MemoryRequirements MemoryRequirementsFaces = _bufferFaces.getMemoryRequirements();
        vk::MemoryRequirements MemoryRequirementsNormals = _bufferNormals.getMemoryRequirements();
        vk::MemoryRequirements MemoryRequirementsSegmentVectors = _bufferSegmentVectors.getMemoryRequirements();
        vk::MemoryRequirements MemoryRequirementsSegmentNormals = _bufferSegmentNormals.getMemoryRequirements();
        vk::MemoryRequirements MemoryRequirementsResultPotential = _bufferResultPotential.getMemoryRequirements();
        vk::MemoryRequirements MemoryRequirementsResultAcceleration = _bufferResultAcceleration.getMemoryRequirements();
        vk::MemoryRequirements MemoryRequirementsResults = _bufferResults.getMemoryRequirements();

        vk::PhysicalDeviceMemoryProperties MemoryProperties = PhysicalDevice.getMemoryProperties();

        uint32_t MemoryTypeIndex = uint32_t(~0);
        vk::DeviceSize MemoryHeapSize = uint32_t(~0);
        for (uint32_t CurrentMemoryTypeIndex = 0; CurrentMemoryTypeIndex < MemoryProperties.memoryTypeCount; ++CurrentMemoryTypeIndex) {
            vk::MemoryType MemoryType = MemoryProperties.memoryTypes[CurrentMemoryTypeIndex];
            if ((vk::MemoryPropertyFlagBits::eHostVisible & MemoryType.propertyFlags) &&
                (vk::MemoryPropertyFlagBits::eHostCoherent & MemoryType.propertyFlags)) {
                MemoryHeapSize = MemoryProperties.memoryHeaps[MemoryType.heapIndex].size;
                MemoryTypeIndex = CurrentMemoryTypeIndex;
                break;
            }
        }

        // std::cout << "Memory Type Index: " << MemoryTypeIndex << std::endl;
        // std::cout << "Memory Heap Size : " << MemoryHeapSize / 1024 / 1024 / 1024 << " GB" << std::endl;

        vk::MemoryAllocateInfo MemoryAllocateInfoVertices(MemoryRequirementsVertices.size, MemoryTypeIndex);
        vk::MemoryAllocateInfo MemoryAllocateInfoFaces(MemoryRequirementsFaces.size, MemoryTypeIndex);
        vk::MemoryAllocateInfo MemoryAllocateInfoNormals(MemoryRequirementsNormals.size, MemoryTypeIndex);
        vk::MemoryAllocateInfo MemoryAllocateInfoSegmentVectors(MemoryRequirementsSegmentVectors.size,
                                                                MemoryTypeIndex);
        vk::MemoryAllocateInfo MemoryAllocateInfoSegmentNormals(MemoryRequirementsSegmentNormals.size,
                                                                MemoryTypeIndex);
        vk::MemoryAllocateInfo MemoryAllocateInfoResultPotential(MemoryRequirementsResultPotential.size,
                                                                 MemoryTypeIndex);
        vk::MemoryAllocateInfo MemoryAllocateInfoResultAcceleration(MemoryRequirementsResultAcceleration.size,
                                                                    MemoryTypeIndex);
        vk::MemoryAllocateInfo MemoryAllocateInfoResults(MemoryRequirementsResults.size, MemoryTypeIndex);

        _memoryVertices = vk::raii::DeviceMemory(_device, MemoryAllocateInfoVertices);
        _memoryFaces = vk::raii::DeviceMemory(_device, MemoryAllocateInfoFaces);
        _memoryNormals = vk::raii::DeviceMemory(_device, MemoryAllocateInfoNormals);
        _memorySegmentVectors = vk::raii::DeviceMemory(_device, MemoryAllocateInfoSegmentVectors);
        _memorySegmentNormals = vk::raii::DeviceMemory(_device, MemoryAllocateInfoSegmentNormals);
        _memoryResultPotential = vk::raii::DeviceMemory(_device, MemoryAllocateInfoResultPotential);
        _memoryResultAcceleration = vk::raii::DeviceMemory(_device, MemoryAllocateInfoResultAcceleration);
        _memoryResults = vk::raii::DeviceMemory(_device, MemoryAllocateInfoResults);

        auto *VerticesPtr = static_cast<FloatType *>(_memoryVertices.mapMemory(0, BufferCreateInfoVertices.size));
        for (int32_t i = 0; i < _vertices.size(); ++i) {
            VerticesPtr[i * 4 + 0] = _vertices[i][0];
            VerticesPtr[i * 4 + 1] = _vertices[i][1];
            VerticesPtr[i * 4 + 2] = _vertices[i][2];
        }
        _memoryVertices.unmapMemory();

        auto *FacesPtr = static_cast<uint32_t *>(_memoryFaces.mapMemory(0, BufferCreateInfoFaces.size));
        for (int32_t i = 0; i < _faces.size(); ++i) {
            FacesPtr[i * 4 + 0] = _faces[i][0];
            FacesPtr[i * 4 + 1] = _faces[i][1];
            FacesPtr[i * 4 + 2] = _faces[i][2];
        }
        _memoryFaces.unmapMemory();

        _bufferVertices.bindMemory(*_memoryVertices, 0);
        _bufferFaces.bindMemory(*_memoryFaces, 0);
        _bufferNormals.bindMemory(*_memoryNormals, 0);
        _bufferSegmentVectors.bindMemory(*_memorySegmentVectors, 0);
        _bufferSegmentNormals.bindMemory(*_memorySegmentNormals, 0);
        _bufferResultPotential.bindMemory(*_memoryResultPotential, 0);
        _bufferResultAcceleration.bindMemory(*_memoryResultAcceleration, 0);
        _bufferResults.bindMemory(*_memoryResults, 0);

        const std::vector<vk::DescriptorSetLayoutBinding> DescriptorSetLayoutBinding = {
                {0, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute},// Vertices
                {1, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute},// Faces
                {2, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute},// Normals
                {3, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute},// SegmentVectors
                {4, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute},// SegmentNormals
                {5, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute},// ResultsPotential
                {6, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute},// ResultsAcceleration
                {7, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute},// Results
        };
        vk::DescriptorSetLayoutCreateInfo DescriptorSetLayoutCreateInfo(
                vk::DescriptorSetLayoutCreateFlags(),
                DescriptorSetLayoutBinding);

        _descriptorSetLayout = _device.createDescriptorSetLayout(DescriptorSetLayoutCreateInfo);

        vk::PipelineLayoutCreateInfo PipelineLayoutCreateInfo(vk::PipelineLayoutCreateFlags(), *_descriptorSetLayout);

        vk::PushConstantRange pushConstantRange{
                vk::ShaderStageFlagBits::eCompute,
                0,
                sizeof(PushConstants)};
        PipelineLayoutCreateInfo.setPushConstantRanges(pushConstantRange);

        _pipelineLayout = _device.createPipelineLayout(PipelineLayoutCreateInfo);
        _pipelineCache = _device.createPipelineCache(vk::PipelineCacheCreateInfo());

        vk::DescriptorPoolSize
                DescriptorPoolSize(vk::DescriptorType::eStorageBuffer, DescriptorSetLayoutBinding.size());
        vk::DescriptorPoolCreateInfo DescriptorPoolCreateInfo(
                vk::DescriptorPoolCreateFlags() | vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet, 1,
                DescriptorPoolSize);
        _descriptorPool = _device.createDescriptorPool(DescriptorPoolCreateInfo);

        vk::DescriptorSetAllocateInfo DescriptorSetAllocInfo(*_descriptorPool, *_descriptorSetLayout);
        _descriptorSets = _device.allocateDescriptorSets(DescriptorSetAllocInfo);

        vk::DescriptorBufferInfo BufferInfoVertices(*_bufferVertices, 0, BufferCreateInfoVertices.size);
        vk::DescriptorBufferInfo BufferInfoFaces(*_bufferFaces, 0, BufferCreateInfoFaces.size);
        vk::DescriptorBufferInfo BufferInfoNormals(*_bufferNormals, 0, BufferCreateInfoNormals.size);
        vk::DescriptorBufferInfo BufferInfoSegmentVectors(*_bufferSegmentVectors, 0,
                                                          BufferCreateInfoSegmentVectors.size);
        vk::DescriptorBufferInfo BufferInfoSegmentNormals(*_bufferSegmentNormals, 0,
                                                          BufferCreateInfoSegmentNormals.size);
        vk::DescriptorBufferInfo BufferInfoResultPotential(*_bufferResultPotential, 0,
                                                           BufferCreateInfoResultPotential.size);
        vk::DescriptorBufferInfo BufferInfoResultAcceleration(*_bufferResultAcceleration, 0,
                                                              BufferCreateInfoResultAcceleration.size);
        vk::DescriptorBufferInfo BufferInfoResults(*_bufferResults, 0, BufferCreateInfoResults.size);

        const std::vector<vk::WriteDescriptorSet> WriteDescriptorSets = {
                {*_descriptorSets[0], 0, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &BufferInfoVertices},
                {*_descriptorSets[0], 1, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &BufferInfoFaces},
                {*_descriptorSets[0], 2, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &BufferInfoNormals},
                {*_descriptorSets[0], 3, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &BufferInfoSegmentVectors},
                {*_descriptorSets[0], 4, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &BufferInfoSegmentNormals},
                {*_descriptorSets[0], 5, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &BufferInfoResultPotential},
                {*_descriptorSets[0], 6, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &BufferInfoResultAcceleration},
                {*_descriptorSets[0], 7, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &BufferInfoResults},
        };
        _device.updateDescriptorSets(WriteDescriptorSets, {});

        vk::CommandPoolCreateInfo CommandPoolCreateInfo(vk::CommandPoolCreateFlags(), ComputeQueueFamilyIndex);
        _commandPool = _device.createCommandPool(CommandPoolCreateInfo);

        vk::CommandBufferAllocateInfo CommandBufferAllocInfo(
                *_commandPool,                   // Command Pool
                vk::CommandBufferLevel::ePrimary,// Level
                1);                              // Num Command Buffers

        _cmdBuffers = _device.allocateCommandBuffers(CommandBufferAllocInfo);

        vk::ShaderModuleCreateInfo ShaderModuleCreateInfoInit(
                vk::ShaderModuleCreateFlags(),
                vulkan_init_len,
                INIT_SPRIV);

        vk::ShaderModuleCreateInfo ShaderModuleCreateInfoEval(
                vk::ShaderModuleCreateFlags(),
                kompute_eval_len,
                EVAL_SPIRV);

        _shaderModuleInit = _device.createShaderModule(ShaderModuleCreateInfoInit);
        _shaderModuleEval = _device.createShaderModule(ShaderModuleCreateInfoEval);

        vk::PipelineShaderStageCreateInfo PipelineShaderCreateInfoInit(
                vk::PipelineShaderStageCreateFlags(),
                vk::ShaderStageFlagBits::eCompute,
                *_shaderModuleInit,
                "main");

        vk::PipelineShaderStageCreateInfo PipelineShaderCreateInfoEval(
                vk::PipelineShaderStageCreateFlags(),
                vk::ShaderStageFlagBits::eCompute,
                *_shaderModuleEval,
                "main");

        vk::ComputePipelineCreateInfo ComputePipelineCreateInfoInit(
                vk::PipelineCreateFlags(),   // Flags
                PipelineShaderCreateInfoInit,// Shader Create Info struct
                *_pipelineLayout);           // Pipeline Layout

        vk::ComputePipelineCreateInfo ComputePipelineCreateInfoEval(
                vk::PipelineCreateFlags(),   // Flags
                PipelineShaderCreateInfoEval,// Shader Create Info struct
                *_pipelineLayout);           // Pipeline Layout

        _pipelineInit = _device.createComputePipeline(_pipelineCache, ComputePipelineCreateInfoInit);
        _pipelineEval = _device.createComputePipeline(_pipelineCache, ComputePipelineCreateInfoEval);

        _queue = _device.getQueue(ComputeQueueFamilyIndex, 0);
        _fence = _device.createFence(vk::FenceCreateFlags());
    }

    GravityModelResult evaluate(const Array3 &Point) override {
        if (!_initialized) init();

        GravityModelResult g_result{};
        auto &[potential, acceleration, gradiometricTensor] = g_result;

        PushConstants pushConstantsData{};
        pushConstantsData.point.x = Point.data[0];
        pushConstantsData.point.y = Point.data[1];
        pushConstantsData.point.z = Point.data[2];
        pushConstantsData.num_faces = _faces.size();

        uint32_t dataSize = _faces.size();
        uint32_t numWorkgroupsX = (dataSize + workgroupSize - 1) / workgroupSize;

        vk::CommandBufferBeginInfo CmdBufferBeginInfo(vk::CommandBufferUsageFlagBits::eOneTimeSubmit);

        _cmdBuffers[0].begin(CmdBufferBeginInfo);
        _cmdBuffers[0].bindPipeline(vk::PipelineBindPoint::eCompute, *_pipelineEval);
        _cmdBuffers[0].fillBuffer(*_bufferResultPotential, 0, sizeof(FloatType), 0);
        _cmdBuffers[0].fillBuffer(*_bufferResultAcceleration, 0, sizeof(VectorType), 0);
        _cmdBuffers[0].fillBuffer(*_bufferResults, 0, sizeof(FloatType) * 6, 0);
        _cmdBuffers[0].bindDescriptorSets(vk::PipelineBindPoint::eCompute,// Bind point
                                          *_pipelineLayout,               // Pipeline Layout
                                          0,                              // First descriptor set
                                          {*_descriptorSets.front()},     // List of descriptor sets
                                          {});                            // Dynamic offsets
        _cmdBuffers[0].pushConstants<PushConstants>(*_pipelineLayout, vk::ShaderStageFlagBits::eCompute, 0, pushConstantsData);
        _cmdBuffers[0].dispatch(numWorkgroupsX, 1, 1);
        _cmdBuffers[0].end();

        vk::SubmitInfo SubmitInfo(
                nullptr,         // Wait Semaphores
                nullptr,         // Pipeline Stage Flags
                *_cmdBuffers[0]);// List of command buffers
        _queue.submit({SubmitInfo}, *_fence);
        auto result = _device.waitForFences({*_fence},    // List of fences
                                            true,         // Wait All
                                            uint64_t(-1));// Timeout

        _commandPool.reset();
        _device.resetFences(*_fence);

        auto *potential_results = (FloatType *) (_memoryResultPotential.mapMemory(0, sizeof(FloatType)));
        potential = potential_results[0];
        _memoryResultPotential.unmapMemory();

        VectorType *acceleration_results = (VectorType *) _memoryResultAcceleration.mapMemory(0, sizeof(VectorType));
        acceleration.data[0] = acceleration_results[0].x;
        acceleration.data[1] = acceleration_results[0].y;
        acceleration.data[2] = acceleration_results[0].z;
        _memoryResultAcceleration.unmapMemory();

        FloatType *results = (FloatType *) _memoryResults.mapMemory(0, 6 * sizeof(FloatType));
        for (int i = 0; i < 6; ++i)
            gradiometricTensor.data[i] = results[i];
        _memoryResults.unmapMemory();

        // 9. Step: Compute prefix consisting of GRAVITATIONAL_CONSTANT * density
        const double prefix = GRAVITATIONAL_CONSTANT * _density;

        // 10. Step: Final expressions after application of the prefix (and a division by 2 for the potential)
        potential = (potential * prefix) / 2.0;
        acceleration = acceleration * (-1.0 * prefix);
        gradiometricTensor = gradiometricTensor * prefix;
        return g_result;
    }

private:
    void init() {
        PushConstants pushConstantsData{};
        pushConstantsData.num_faces = _faces.size();

        vk::CommandBufferBeginInfo CmdBufferBeginInfo(vk::CommandBufferUsageFlagBits::eOneTimeSubmit);

        uint32_t dataSize = _faces.size();

        uint32_t numWorkgroupsX = (dataSize + workgroupSize - 1) / workgroupSize;

        _cmdBuffers[0].begin(CmdBufferBeginInfo);
        _cmdBuffers[0].bindPipeline(vk::PipelineBindPoint::eCompute, *_pipelineInit);
        _cmdBuffers[0].bindDescriptorSets(vk::PipelineBindPoint::eCompute,// Bind point
                                          *_pipelineLayout,               // Pipeline Layout
                                          0,                              // First descriptor set
                                          {*_descriptorSets.front()},     // List of descriptor sets
                                          {});                            // Dynamic offsets
        _cmdBuffers[0].pushConstants<PushConstants>(*_pipelineLayout, vk::ShaderStageFlagBits::eCompute, 0, pushConstantsData);
        _cmdBuffers[0].dispatch(numWorkgroupsX, 1, 1);
        _cmdBuffers[0].end();

        vk::SubmitInfo SubmitInfo(
                nullptr,         // Wait Semaphores
                nullptr,         // Pipeline Stage Flags
                *_cmdBuffers[0]);// List of command buffers
        _queue.submit({SubmitInfo}, *_fence);
        auto result = _device.waitForFences({*_fence},    // List of fences
                                            true,         // Wait All
                                            uint64_t(-1));// Timeout

        _commandPool.reset();
        _device.resetFences(*_fence);

        _initialized = true;
    }

    vk::raii::Context _vk_context;
    vk::raii::Instance _instance;

    vk::raii::Device _device;

    vk::raii::Buffer _bufferVertices;
    vk::raii::Buffer _bufferFaces;
    vk::raii::Buffer _bufferNormals;
    vk::raii::Buffer _bufferSegmentVectors;
    vk::raii::Buffer _bufferSegmentNormals;

    vk::raii::Buffer _bufferResultPotential;
    vk::raii::Buffer _bufferResultAcceleration;
    vk::raii::Buffer _bufferResults;

    vk::raii::DeviceMemory _memoryVertices;
    vk::raii::DeviceMemory _memoryFaces;
    vk::raii::DeviceMemory _memoryNormals;
    vk::raii::DeviceMemory _memorySegmentVectors;
    vk::raii::DeviceMemory _memorySegmentNormals;
    vk::raii::DeviceMemory _memoryResultPotential;
    vk::raii::DeviceMemory _memoryResultAcceleration;
    vk::raii::DeviceMemory _memoryResults;

    vk::raii::DescriptorSetLayout _descriptorSetLayout;
    vk::raii::PipelineLayout _pipelineLayout;
    vk::raii::PipelineCache _pipelineCache;
    vk::raii::DescriptorPool _descriptorPool;

    std::vector<vk::raii::DescriptorSet> _descriptorSets;

    vk::raii::CommandPool _commandPool;

    std::vector<vk::raii::CommandBuffer> _cmdBuffers;

    vk::raii::ShaderModule _shaderModuleInit;
    vk::raii::ShaderModule _shaderModuleEval;

    vk::raii::Pipeline _pipelineInit;
    vk::raii::Pipeline _pipelineEval;

    vk::raii::Queue _queue;
    vk::raii::Fence _fence;
};

std::unique_ptr<GravityEvaluableBase> create_gravity_evaluable(
        const std::vector<Array3> &Vertices,
        const std::vector<IndexArray3> &Faces,
        double density) {
    return std::make_unique<GravityEvaluable>(Vertices, Faces, density);
}
