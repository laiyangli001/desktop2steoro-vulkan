#include "bridge.h"

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <d3d11.h>
#include <dxgi.h>
#include <wrl/client.h>

#include <openvino/openvino.hpp>
#include <openvino/core/preprocess/pre_post_process.hpp>
#include <openvino/runtime/intel_gpu/ocl/dx.hpp>

#include <cstring>
#include <memory>
#include <string>

namespace {
using Microsoft::WRL::ComPtr;
thread_local std::string g_last_error;

void set_error(const std::string& message) {
    g_last_error = message;
}

void set_hresult_error(const char* operation, HRESULT status) {
    set_error(std::string(operation) + " failed, HRESULT=0x" +
              std::to_string(static_cast<unsigned long>(status)));
}

unsigned long long adapter_luid(ID3D11Device* device) {
    if (!device) {
        return 0;
    }
    ComPtr<IDXGIDevice> dxgi_device;
    if (FAILED(device->QueryInterface(IID_PPV_ARGS(&dxgi_device)))) {
        return 0;
    }
    ComPtr<IDXGIAdapter> adapter;
    if (FAILED(dxgi_device->GetAdapter(&adapter))) {
        return 0;
    }
    DXGI_ADAPTER_DESC description{};
    if (FAILED(adapter->GetDesc(&description))) {
        return 0;
    }
    return (static_cast<unsigned long long>(static_cast<unsigned long>(description.AdapterLuid.HighPart)) << 32) |
           static_cast<unsigned long long>(static_cast<unsigned long>(description.AdapterLuid.LowPart));
}

struct State {
    ov::Core core;
    std::shared_ptr<ov::Model> model;
    std::shared_ptr<ov::intel_gpu::ocl::D3DContext> context;
    ov::CompiledModel compiled;
    ov::InferRequest request;

    ComPtr<ID3D11Device> device;
    ComPtr<ID3D11DeviceContext> device_context;
    ComPtr<ID3D11VideoDevice> video_device;
    ComPtr<ID3D11VideoContext> video_context;
    ComPtr<ID3D11VideoProcessorEnumerator> processor_enumerator;
    ComPtr<ID3D11VideoProcessor> processor;
    ComPtr<ID3D11Texture2D> nv12_texture;
    ComPtr<ID3D11VideoProcessorOutputView> output_view;
    unsigned long long adapter_luid = 0;
    int converted_width = 0;
    int converted_height = 0;
};

bool initialize_video_processor(State& state, int width, int height) {
    if (width <= 0 || height <= 0) {
        set_error("BGRA texture dimensions must be positive");
        return false;
    }
    if (state.processor && state.converted_width == width && state.converted_height == height) {
        return true;
    }

    state.processor.Reset();
    state.processor_enumerator.Reset();
    state.video_context.Reset();
    state.video_device.Reset();
    state.output_view.Reset();
    state.nv12_texture.Reset();

    state.device->GetImmediateContext(&state.device_context);
    if (!state.device_context) {
        set_error("ID3D11Device::GetImmediateContext returned no context");
        return false;
    }
    HRESULT status = S_OK;
    status = state.device.As(&state.video_device);
    if (FAILED(status)) {
        set_hresult_error("ID3D11Device::QueryInterface(ID3D11VideoDevice)", status);
        return false;
    }
    status = state.device_context.As(&state.video_context);
    if (FAILED(status)) {
        set_hresult_error("ID3D11DeviceContext::QueryInterface(ID3D11VideoContext)", status);
        return false;
    }

    D3D11_VIDEO_PROCESSOR_CONTENT_DESC content{};
    content.InputFrameFormat = D3D11_VIDEO_FRAME_FORMAT_PROGRESSIVE;
    content.InputFrameRate = {60, 1};
    content.InputWidth = static_cast<UINT>(width);
    content.InputHeight = static_cast<UINT>(height);
    content.OutputFrameRate = {60, 1};
    content.OutputWidth = static_cast<UINT>(width);
    content.OutputHeight = static_cast<UINT>(height);
    content.Usage = D3D11_VIDEO_USAGE_PLAYBACK_NORMAL;

    status = state.video_device->CreateVideoProcessorEnumerator(
        &content, &state.processor_enumerator);
    if (FAILED(status)) {
        set_hresult_error("CreateVideoProcessorEnumerator", status);
        return false;
    }
    status = state.video_device->CreateVideoProcessor(
        state.processor_enumerator.Get(), 0, &state.processor);
    if (FAILED(status)) {
        set_hresult_error("CreateVideoProcessor", status);
        return false;
    }

    D3D11_TEXTURE2D_DESC output_desc{};
    output_desc.Width = static_cast<UINT>(width);
    output_desc.Height = static_cast<UINT>(height);
    output_desc.MipLevels = 1;
    output_desc.ArraySize = 1;
    output_desc.Format = DXGI_FORMAT_NV12;
    output_desc.SampleDesc.Count = 1;
    output_desc.Usage = D3D11_USAGE_DEFAULT;
    output_desc.BindFlags = D3D11_BIND_RENDER_TARGET;
    status = state.device->CreateTexture2D(&output_desc, nullptr, &state.nv12_texture);
    if (FAILED(status)) {
        set_hresult_error("CreateTexture2D(NV12)", status);
        return false;
    }

    D3D11_VIDEO_PROCESSOR_OUTPUT_VIEW_DESC output_view_desc{};
    output_view_desc.ViewDimension = D3D11_VPOV_DIMENSION_TEXTURE2D;
    output_view_desc.Texture2D.MipSlice = 0;
    status = state.video_device->CreateVideoProcessorOutputView(
        state.nv12_texture.Get(), state.processor_enumerator.Get(),
        &output_view_desc, &state.output_view);
    if (FAILED(status)) {
        set_hresult_error("CreateVideoProcessorOutputView", status);
        return false;
    }

    state.converted_width = width;
    state.converted_height = height;
    return true;
}

bool convert_bgra_to_nv12(State& state, ID3D11Texture2D* source, int width, int height) {
    if (!initialize_video_processor(state, width, height)) {
        return false;
    }

    D3D11_VIDEO_PROCESSOR_INPUT_VIEW_DESC input_view_desc{};
    input_view_desc.ViewDimension = D3D11_VPIV_DIMENSION_TEXTURE2D;
    input_view_desc.Texture2D.MipSlice = 0;
    ComPtr<ID3D11VideoProcessorInputView> input_view;
    HRESULT status = state.video_device->CreateVideoProcessorInputView(
        source, state.processor_enumerator.Get(), &input_view_desc, &input_view);
    if (FAILED(status)) {
        set_hresult_error("CreateVideoProcessorInputView(BGRA8)", status);
        return false;
    }

    state.video_context->VideoProcessorSetStreamFrameFormat(
        state.processor.Get(), 0, D3D11_VIDEO_FRAME_FORMAT_PROGRESSIVE);
    D3D11_VIDEO_PROCESSOR_STREAM stream{};
    stream.Enable = TRUE;
    stream.OutputIndex = 0;
    stream.InputFrameOrField = 0;
    stream.PastFrames = 0;
    stream.FutureFrames = 0;
    stream.pInputSurface = input_view.Get();
    status = state.video_context->VideoProcessorBlt(
        state.processor.Get(), state.output_view.Get(), 0, 1, &stream);
    if (FAILED(status)) {
        set_hresult_error("VideoProcessorBlt(BGRA8->NV12)", status);
        return false;
    }
    return true;
}

bool prepare_model(State& state, const char* model_path) {
    state.model = state.core.read_model(model_path);

    // OpenVINO's D3D11 surface path consumes NV12 as two GPU planes. The
    // preprocessing graph performs color conversion/layout on the GPU before
    // the original depth model executes.
    ov::preprocess::PrePostProcessor processor(state.model);
    processor.input().tensor()
        .set_element_type(ov::element::u8)
        .set_color_format(ov::preprocess::ColorFormat::NV12_TWO_PLANES, {"y", "uv"})
        .set_memory_type(ov::intel_gpu::memory_type::surface);
    processor.input().preprocess().convert_color(ov::preprocess::ColorFormat::BGR);
    processor.input().model().set_layout("NCHW");
    state.model = processor.build();
    return state.model->inputs().size() >= 2;
}
}

extern "C" D2S_OPENVINO_D3D11_API int d2s_openvino_d3d11_capabilities(void) {
    // bit 0: OpenVINO NV12 surface RemoteTensor; bit 1: GPU BGRA8->NV12.
    return 0x03;
}

extern "C" D2S_OPENVINO_D3D11_API void* d2s_openvino_d3d11_create(
    const char* model_path, void* d3d11_device) {
    g_last_error.clear();
    if (!model_path || !d3d11_device) {
        set_error("model path and D3D11 device are required");
        return nullptr;
    }
    try {
        auto state = std::make_unique<State>();
        state->device = static_cast<ID3D11Device*>(d3d11_device);
        state->adapter_luid = adapter_luid(state->device.Get());
        if (!state->adapter_luid) {
            set_error("could not query the D3D11 device adapter LUID");
            return nullptr;
        }
        if (!prepare_model(*state, model_path)) {
            set_error("OpenVINO model must expose the two NV12 surface inputs after preprocessing");
            return nullptr;
        }
        state->context = std::make_shared<ov::intel_gpu::ocl::D3DContext>(
            state->core, state->device.Get());
        state->compiled = state->core.compile_model(state->model, *state->context);
        state->request = state->compiled.create_infer_request();
        return state.release();
    } catch (const std::exception& exc) {
        set_error(exc.what());
        return nullptr;
    }
}

extern "C" D2S_OPENVINO_D3D11_API unsigned long long d2s_openvino_d3d11_adapter_luid(void* handle) {
    auto* state = static_cast<State*>(handle);
    return state ? state->adapter_luid : 0;
}

extern "C" D2S_OPENVINO_D3D11_API int d2s_openvino_d3d11_nv12_surface(
    void* handle, void** texture, int* width, int* height) {
    auto* state = static_cast<State*>(handle);
    if (!state || !texture || !width || !height || !state->nv12_texture) {
        set_error("NV12 surface is not available; set_texture must succeed first");
        return 0;
    }
    *texture = state->nv12_texture.Get();
    *width = state->converted_width;
    *height = state->converted_height;
    return 1;
}

extern "C" D2S_OPENVINO_D3D11_API int d2s_openvino_d3d11_set_texture(
    void* handle, const char*, void* d3d11_texture, int width, int height) {
    auto* state = static_cast<State*>(handle);
    if (!state || !d3d11_texture || width <= 0 || height <= 0) {
        set_error("invalid D3D11 BGRA texture arguments");
        return 0;
    }
    try {
        auto* source = static_cast<ID3D11Texture2D*>(d3d11_texture);
        if (!convert_bgra_to_nv12(*state, source, width, height)) {
            return 0;
        }
        auto nv12_tensors = state->context->create_tensor_nv12(
            height, width, state->nv12_texture.Get());
        state->request.set_tensor(state->model->input(0), nv12_tensors.first);
        state->request.set_tensor(state->model->input(1), nv12_tensors.second);
        return 1;
    } catch (const std::exception& exc) {
        set_error(exc.what());
        return 0;
    }
}

extern "C" D2S_OPENVINO_D3D11_API int d2s_openvino_d3d11_infer(void* handle) {
    auto* state = static_cast<State*>(handle);
    if (!state) {
        set_error("invalid OpenVINO D3D11 bridge handle");
        return 0;
    }
    try {
        state->request.infer();
        return 1;
    } catch (const std::exception& exc) {
        set_error(exc.what());
        return 0;
    }
}

extern "C" D2S_OPENVINO_D3D11_API int d2s_openvino_d3d11_output_shape(
    void* handle, long long* dims, int capacity) {
    auto* state = static_cast<State*>(handle);
    if (!state || !dims || capacity <= 0) {
        set_error("invalid output shape arguments");
        return -1;
    }
    try {
        const auto shape = state->request.get_output_tensor(0).get_shape();
        if (capacity < static_cast<int>(shape.size())) {
            set_error("output shape buffer is too small");
            return -1;
        }
        for (size_t index = 0; index < shape.size(); ++index) {
            dims[index] = static_cast<long long>(shape[index]);
        }
        return static_cast<int>(shape.size());
    } catch (const std::exception& exc) {
        set_error(exc.what());
        return -1;
    }
}

extern "C" D2S_OPENVINO_D3D11_API int d2s_openvino_d3d11_read_output(
    void* handle, float* output, int capacity) {
    auto* state = static_cast<State*>(handle);
    if (!state || !output || capacity <= 0) {
        set_error("invalid output buffer arguments");
        return -1;
    }
    try {
        const auto tensor = state->request.get_output_tensor(0);
        if (tensor.get_element_type() != ov::element::f32) {
            set_error("native output ABI currently requires f32 model output");
            return -1;
        }
        const auto count = ov::shape_size(tensor.get_shape());
        if (capacity < static_cast<int>(count)) {
            set_error("output buffer is too small");
            return -1;
        }
        std::memcpy(output, tensor.data<const float>(), count * sizeof(float));
        return static_cast<int>(count);
    } catch (const std::exception& exc) {
        set_error(exc.what());
        return -1;
    }
}

extern "C" D2S_OPENVINO_D3D11_API int d2s_openvino_d3d11_last_error(
    char* output, int capacity) {
    if (!output || capacity <= 0) {
        return static_cast<int>(g_last_error.size());
    }
    const int count = static_cast<int>(g_last_error.size());
    const int copied = count < capacity - 1 ? count : capacity - 1;
    std::memcpy(output, g_last_error.data(), static_cast<size_t>(copied));
    output[copied] = '\0';
    return count;
}

extern "C" D2S_OPENVINO_D3D11_API void d2s_openvino_d3d11_destroy(void* handle) {
    delete static_cast<State*>(handle);
}

#endif
