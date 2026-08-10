#include <cassert>
#include <cstring>

#include "common.h"

#include <wgpu/wgpu.h>

#include "shader_eval.hpp"
#include "shader_init.hpp"

static void handle_request_adapter(WGPURequestAdapterStatus status,
                                   WGPUAdapter adapter, char const *message,
                                   void *userdata) {
    *(WGPUAdapter *) userdata = adapter;
}
static void handle_request_device(WGPURequestDeviceStatus status,
                                  WGPUDevice device, char const *message,
                                  void *userdata) {
    *(WGPUDevice *) userdata = device;
}

static void handle_buffer_map(WGPUBufferMapAsyncStatus status, void *userdata) {
    // printf(" buffer_map status=%#.8x\n", status);
}

WGPUBufferUsageFlags BUFFER_FLAGS = WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst | WGPUBufferUsage_CopySrc;
WGPUBufferUsageFlags STAGING_FLAGS = WGPUBufferUsage_MapRead | WGPUBufferUsage_CopyDst;

struct WebGpuMemory {
    WGPUBuffer storage_buffer;
    WGPUBufferDescriptor descriptor{};

    uint32_t size;

    WebGpuMemory(WGPUDevice device, const uint32_t byte_size, WGPUBufferUsageFlags usage_flags = BUFFER_FLAGS)
        : storage_buffer(nullptr), size(byte_size) {
        descriptor.label = "";
        descriptor.usage = usage_flags;
        descriptor.size = byte_size;
        descriptor.mappedAtCreation = false;

        storage_buffer = wgpuDeviceCreateBuffer(device, &descriptor);
        assert(storage_buffer);
    }

    ~WebGpuMemory() {
        wgpuBufferRelease(storage_buffer);
    }
};

struct ShaderModule {
    WGPUShaderModuleDescriptor descriptor{};
    WGPUShaderModule module{};
    WGPUShaderModuleWGSLDescriptor descriptor_wgsl{};

    ShaderModule(WGPUDevice device, const char *name, const char *code) {
        descriptor_wgsl.code = code;
        descriptor_wgsl.chain.sType = WGPUSType_ShaderModuleWGSLDescriptor;

        descriptor.label = name;
        descriptor.nextInChain = (const WGPUChainedStruct *) &descriptor_wgsl;

        module = wgpuDeviceCreateShaderModule(device, &descriptor);
        assert(module);
    }

    ~ShaderModule() {
        wgpuShaderModuleRelease(module);
    }
};

struct WriteItem {
    const WebGpuMemory &mem;
    void *data;
};

struct Pipeline {
    WGPUComputePipelineDescriptor descriptor{};
    WGPUComputePipeline pipeline{};
    WGPUBindGroupLayout bind_group_layout{};

    WGPUBindGroup bind_group{};
    WGPUBindGroupDescriptor bind_group_descriptor{};

    ShaderModule shader_module;


    explicit Pipeline(WGPUDevice device, const char *name, const char *code, const std::vector<WGPUBindGroupEntry> &buffer_group_entries)
        : shader_module(device, name, code) {
        descriptor.label = name;
        descriptor.compute.module = shader_module.module;
        descriptor.compute.entryPoint = "main";

        pipeline = wgpuDeviceCreateComputePipeline(device, &descriptor);
        assert(pipeline);

        bind_group_layout = wgpuComputePipelineGetBindGroupLayout(pipeline, 0);
        assert(bind_group_layout);

        bind_group_descriptor.label = name;
        bind_group_descriptor.layout = bind_group_layout;
        bind_group_descriptor.entryCount = buffer_group_entries.size();
        bind_group_descriptor.entries = buffer_group_entries.data();

        bind_group = wgpuDeviceCreateBindGroup(device, &bind_group_descriptor);
    }

    void dispatch(WGPUQueue queue, WGPUDevice device, WGPUBuffer output, WGPUBuffer staging, uint32_t output_size, void *cpu_out, uint32_t dispatchX, uint32_t dispatchY, uint32_t dispatchZ, const std::vector<WriteItem> &write_items) {
        WGPUCommandEncoderDescriptor command_encoder_descriptor{};
        WGPUCommandEncoder command_encoder{};

        command_encoder_descriptor.label = "command_encoder";
        command_encoder = wgpuDeviceCreateCommandEncoder(device, &command_encoder_descriptor);
        assert(command_encoder);

        WGPUComputePassDescriptor compute_pass_descriptor{};
        WGPUComputePassEncoder compute_pass_encoder{};

        compute_pass_descriptor.label = "compute_pass";
        compute_pass_encoder = wgpuCommandEncoderBeginComputePass(command_encoder, &compute_pass_descriptor);
        assert(compute_pass_encoder);

        wgpuComputePassEncoderSetPipeline(compute_pass_encoder, pipeline);
        wgpuComputePassEncoderSetBindGroup(compute_pass_encoder, 0, bind_group, 0, nullptr);

        wgpuComputePassEncoderDispatchWorkgroups(compute_pass_encoder, dispatchX, dispatchY, dispatchZ);
        wgpuComputePassEncoderEnd(compute_pass_encoder);
        wgpuComputePassEncoderRelease(compute_pass_encoder);

        wgpuCommandEncoderCopyBufferToBuffer(command_encoder, output, 0, staging, 0, output_size);

        WGPUCommandBufferDescriptor command_buffer_descriptor{};
        WGPUCommandBuffer command_buffer{};

        command_buffer_descriptor.label = "command_buffer";
        command_buffer = wgpuCommandEncoderFinish(command_encoder, &command_buffer_descriptor);
        assert(command_buffer);

        for (auto item: write_items) {
            wgpuQueueWriteBuffer(queue, item.mem.storage_buffer, 0, item.data, item.mem.size);
        }

        // TODO: wgpuQueueWriteBuffer

        wgpuQueueSubmit(queue, 1, &command_buffer);

        wgpuBufferMapAsync(staging, WGPUMapMode_Read, 0, output_size, handle_buffer_map, NULL);
        wgpuDevicePoll(device, true, NULL);

        auto *buf = wgpuBufferGetMappedRange(staging, 0, output_size);
        memcpy(cpu_out, buf, output_size);
        wgpuBufferUnmap(staging);

        wgpuCommandBufferRelease(command_buffer);
        wgpuCommandEncoderRelease(command_encoder);
    }

    ~Pipeline() {
        wgpuBindGroupLayoutRelease(bind_group_layout);
        wgpuComputePipelineRelease(pipeline);
    }
};

struct Instance {
    WGPUInstance instance;
    WGPUAdapter adapter;
    WGPUDevice device;
    WGPUQueue queue;

    std::vector<WebGpuMemory> buffers;
    std::vector<WGPUBindGroupEntry> buffer_group_entries;

    std::vector<Pipeline> pipelines;

    std::vector<WebGpuMemory> staging_buffers;

    explicit Instance(int n_buffers, int n_pipelines)
        : instance(nullptr), adapter(nullptr), device(nullptr), queue(nullptr) {
        instance = wgpuCreateInstance(nullptr);
        assert(instance);

        // wgpuInstanceRequestAdapter(instance, nullptr, handle_request_adapter, (void *) &adapter);
        // assert(adapter);

        WGPUAdapter adapters[32];
        wgpuInstanceEnumerateAdapters(instance, nullptr, adapters);

        int num_adapters = 0;
        for (int i = 0; i < 32; i++) {
            if (adapters[i] == nullptr) {
                break;
            }

            WGPUAdapterInfo info;
            wgpuAdapterGetInfo(adapters[i], &info);

            printf("Adapter: %s\n", info.device);
            num_adapters++;
        }

        int adapter_index = (num_adapters > 3) ? 2 : 0;
        printf("Using adapter number: %d\n", adapter_index);
        adapter = adapters[adapter_index];

        wgpuAdapterRequestDevice(adapter, nullptr, handle_request_device, (void *) &device);
        assert(device);

        queue = wgpuDeviceGetQueue(device);
        assert(queue);

        buffers.reserve(n_buffers);
        buffer_group_entries.reserve(n_buffers);
        pipelines.reserve(n_pipelines);

        staging_buffers.reserve(n_buffers);
    }

    template<typename T>
    void addBuffer(uint32_t n_elem) {
        auto &b = buffers.emplace_back(device, n_elem * sizeof(T));

        WGPUBindGroupEntry entry = {};
        entry.binding = buffer_group_entries.size();
        entry.buffer = b.storage_buffer;
        entry.offset = 0;
        entry.size = b.size;

        buffer_group_entries.push_back(entry);
    }

    template<typename T>
    void addStagingBuffer(uint32_t n_elem) {
        staging_buffers.emplace_back(device, n_elem * sizeof(T), STAGING_FLAGS);
    }

    void addPipeline(const char *name, const char *code) {
        pipelines.emplace_back(device, name, code, buffer_group_entries);
    }

    void runPipeline(uint32_t index, void *cpu_out, uint32_t dispatchX, uint32_t dispatchY, uint32_t dispatchZ, const std::vector<WriteItem> &write_items) {
        pipelines[index].dispatch(
                queue, device,
                buffers[5].storage_buffer, staging_buffers.back().storage_buffer,
                buffers[5].size,
                cpu_out,
                dispatchX, dispatchY, dispatchZ,
                write_items);
    }

    ~Instance() {
        wgpuQueueRelease(queue);
        wgpuDeviceRelease(device);
        // wgpuAdapterRelease(adapter);
        wgpuInstanceRelease(instance);
    }
};

GlobalResources::GlobalResources(int &argc, char *argv[]) {
}
GlobalResources::~GlobalResources() = default;

class GravityEvaluable : public GravityEvaluableBase {
public:
    GravityEvaluable(
            const std::vector<Array3> &Vertices,
            const std::vector<IndexArray3> &Faces,
            const double density)
        : GravityEvaluableBase(Vertices, Faces, density), instance(10, 5) {

        constexpr uint32_t workgroupSize = 256;
        const uint32_t numWorkgroups = (Faces.size() + workgroupSize - 1) / workgroupSize;
        dispatchX = numWorkgroups;
        dispatchY = 1;

        if (dispatchX > 65534) {
            dispatchX = 65534;
            dispatchY = ceil(numWorkgroups / 65534.0);
        };

        instance.addBuffer<FloatType>(Vertices.size() * 4);// vertices
        instance.addBuffer<uint32_t>(Faces.size() * 4);    // faces

        instance.addBuffer<FloatType>(Faces.size() * 4); // normals
        instance.addBuffer<FloatType>(Faces.size() * 12);// segment_vectors
        instance.addBuffer<FloatType>(Faces.size() * 12);// segment_normals

        instance.addBuffer<FloatType>(Faces.size() * 10);// results
        instance.addBuffer<FloatType>(4);                // settings

        instance.addStagingBuffer<FloatType>(Faces.size() * 10);// results staging

        instance.addPipeline("init", shader_init.data());
        instance.addPipeline("eval", shader_eval.data());

        results_cpu.resize(Faces.size() * 10);
    }

    GravityModelResult evaluate(const Array3 &Point) override {
        if (!_initialized) init();

        FloatType settings[4] = {
                Point[0], Point[1], Point[2], (FloatType) _faces.size()};

        const std::vector<WriteItem> items = {
                {instance.buffers[6], settings}};

        instance.runPipeline(1, results_cpu.data(), dispatchX, dispatchY, 1, items);

        GravityModelResult result{};
        auto &[potential, acceleration, gradiometricTensor] = result;

        for (int i = 0; i < _faces.size(); i++) {
            potential += results_cpu[i * 10 + 3];
            acceleration[0] += results_cpu[i * 10 + 0];
            acceleration[1] += results_cpu[i * 10 + 1];
            acceleration[2] += results_cpu[i * 10 + 2];

            for (int j = 0; j < 6; j++)
                gradiometricTensor.data[j] += results_cpu[i * 10 + j + 4];
        }

        // 9. Step: Compute prefix consisting of GRAVITATIONAL_CONSTANT * density
        const double prefix = GRAVITATIONAL_CONSTANT * _density;

        // 10. Step: Final expressions after application of the prefix (and a division by 2 for the potential)
        potential = (potential * prefix) / 2.0;
        acceleration = acceleration * (-1.0 * prefix);
        gradiometricTensor = gradiometricTensor * prefix;
        return result;
    }

private:
    void init() {
        std::vector<FloatType> vertex_tmp(_vertices.size() * 4);
        for (int i = 0; i < _vertices.size(); i++) {
            vertex_tmp[i * 4 + 0] = _vertices[i][0];
            vertex_tmp[i * 4 + 1] = _vertices[i][1];
            vertex_tmp[i * 4 + 2] = _vertices[i][2];
        }

        std::vector<uint32_t> face_tmp(_faces.size() * 4);
        for (int i = 0; i < _faces.size(); i++) {
            face_tmp[i * 4 + 0] = _faces[i][0];
            face_tmp[i * 4 + 1] = _faces[i][1];
            face_tmp[i * 4 + 2] = _faces[i][2];
        }
        FloatType settings[4] = {0.0, 0.0, 0.0, (FloatType) _faces.size()};

        const std::vector<WriteItem> items = {
                {instance.buffers[0], vertex_tmp.data()},
                {instance.buffers[1], face_tmp.data()},
                {instance.buffers[6], settings},
        };

        instance.runPipeline(0, results_cpu.data(), dispatchX, dispatchY, 1, items);

        _initialized = true;
    }

    uint32_t dispatchX;
    uint32_t dispatchY;

    Instance instance;

    std::vector<FloatType> results_cpu;
};

std::unique_ptr<GravityEvaluableBase> create_gravity_evaluable(
        const std::vector<Array3> &Vertices,
        const std::vector<IndexArray3> &Faces,
        double density) {
    return std::make_unique<GravityEvaluable>(Vertices, Faces, density);
}
