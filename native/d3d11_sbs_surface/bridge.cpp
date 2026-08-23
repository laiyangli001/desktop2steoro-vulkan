#include "bridge.h"

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <d3d11.h>
#include <dxgi.h>
#include <dxgi1_2.h>
#include <wrl/client.h>

#include <cstring>
#include <memory>
#include <string>

namespace {
using Microsoft::WRL::ComPtr;
thread_local std::string g_last_error;

void set_error(const std::string& value) { g_last_error = value; }

void set_hresult_error(const char* operation, HRESULT status) {
    set_error(std::string(operation) + " failed, HRESULT=0x" +
              std::to_string(static_cast<unsigned long>(status)));
}

unsigned long long get_adapter_luid(ID3D11Device* device) {
    if (!device) return 0;
    ComPtr<IDXGIDevice> dxgi_device;
    ComPtr<IDXGIAdapter> adapter;
    DXGI_ADAPTER_DESC desc{};
    if (FAILED(device->QueryInterface(IID_PPV_ARGS(&dxgi_device))) ||
        FAILED(dxgi_device->GetAdapter(&adapter)) ||
        FAILED(adapter->GetDesc(&desc))) {
        return 0;
    }
    return (static_cast<unsigned long long>(static_cast<unsigned long>(desc.AdapterLuid.HighPart)) << 32) |
           static_cast<unsigned long long>(static_cast<unsigned long>(desc.AdapterLuid.LowPart));
}

struct State {
    ComPtr<ID3D11Device> device;
    ComPtr<ID3D11DeviceContext> context;
    ComPtr<ID3D11VideoDevice> video_device;
    ComPtr<ID3D11VideoContext> video_context;
    ComPtr<ID3D11VideoProcessorEnumerator> enumerator;
    ComPtr<ID3D11VideoProcessor> processor;
    ComPtr<ID3D11Texture2D> staging_bgra;
    ComPtr<ID3D11Texture2D> bgra;
    ComPtr<ID3D11Texture2D> external_bgra;
    ComPtr<ID3D11Texture2D> nv12;
    ComPtr<ID3D11VideoProcessorOutputView> output_view;
    unsigned long long adapter_luid = 0;
    int width = 0;
    int height = 0;
};

bool initialize(State& state, int width, int height, int adapter_index,
                ID3D11Device* existing_device = nullptr) {
    if (width <= 0 || height <= 0) {
        set_error("SBS surface dimensions must be positive");
        return false;
    }
    HRESULT status = S_OK;
    if (existing_device) {
        state.device = existing_device;
        state.device->GetImmediateContext(&state.context);
        if (!state.context) {
            set_error("existing D3D11 device has no immediate context");
            return false;
        }
    } else {
        UINT flags = D3D11_CREATE_DEVICE_BGRA_SUPPORT;
        D3D_FEATURE_LEVEL levels[] = {D3D_FEATURE_LEVEL_11_1, D3D_FEATURE_LEVEL_11_0};
        ComPtr<IDXGIAdapter1> selected_adapter;
        if (adapter_index >= 0) {
        ComPtr<IDXGIFactory1> factory;
        if (FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&factory)))) {
            set_error("CreateDXGIFactory1 failed");
            return false;
        }
        HRESULT enum_status = factory->EnumAdapters1(static_cast<UINT>(adapter_index), &selected_adapter);
        if (FAILED(enum_status)) {
            set_hresult_error("IDXGIFactory1::EnumAdapters1", enum_status);
            return false;
        }
        }
        D3D_FEATURE_LEVEL obtained{};
        status = D3D11CreateDevice(
            selected_adapter.Get(),
            selected_adapter ? D3D_DRIVER_TYPE_UNKNOWN : D3D_DRIVER_TYPE_HARDWARE,
            nullptr,
            flags,
            levels,
            ARRAYSIZE(levels),
            D3D11_SDK_VERSION,
            &state.device,
            &obtained,
            &state.context);
        if (FAILED(status)) {
            set_hresult_error("D3D11CreateDevice", status);
            return false;
        }
    }
    state.adapter_luid = get_adapter_luid(state.device.Get());
    if (!state.adapter_luid) {
        set_error("could not query D3D11 adapter LUID");
        return false;
    }
    state.device.As(&state.video_device);
    state.context.As(&state.video_context);
    if (!state.video_device || !state.video_context) {
        set_error("D3D11 video interfaces are unavailable");
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
    status = state.video_device->CreateVideoProcessorEnumerator(&content, &state.enumerator);
    if (FAILED(status)) { set_hresult_error("CreateVideoProcessorEnumerator", status); return false; }
    status = state.video_device->CreateVideoProcessor(state.enumerator.Get(), 0, &state.processor);
    if (FAILED(status)) { set_hresult_error("CreateVideoProcessor", status); return false; }

    D3D11_TEXTURE2D_DESC source_desc{};
    source_desc.Width = static_cast<UINT>(width);
    source_desc.Height = static_cast<UINT>(height);
    source_desc.MipLevels = 1;
    source_desc.ArraySize = 1;
    source_desc.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
    source_desc.SampleDesc.Count = 1;
    source_desc.Usage = D3D11_USAGE_DEFAULT;
    // VideoProcessor input views require either no bind flags or one of the
    // video-compatible flags. A shader-resource-only texture is rejected by
    // CreateVideoProcessorInputView on current Intel drivers.
    source_desc.BindFlags = 0;
    source_desc.MiscFlags = D3D11_RESOURCE_MISC_SHARED_NTHANDLE;
    status = state.device->CreateTexture2D(&source_desc, nullptr, &state.bgra);
    if (FAILED(status)) { set_hresult_error("CreateTexture2D(BGRA8)", status); return false; }
    source_desc.Usage = D3D11_USAGE_STAGING;
    source_desc.BindFlags = 0;
    source_desc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
    status = state.device->CreateTexture2D(&source_desc, nullptr, &state.staging_bgra);
    if (FAILED(status)) { set_hresult_error("CreateTexture2D(BGRA8 staging)", status); return false; }

    D3D11_TEXTURE2D_DESC output_desc{};
    output_desc.Width = static_cast<UINT>(width);
    output_desc.Height = static_cast<UINT>(height);
    output_desc.MipLevels = 1;
    output_desc.ArraySize = 1;
    output_desc.Format = DXGI_FORMAT_NV12;
    output_desc.SampleDesc.Count = 1;
    output_desc.Usage = D3D11_USAGE_DEFAULT;
    output_desc.BindFlags = D3D11_BIND_RENDER_TARGET;
    status = state.device->CreateTexture2D(&output_desc, nullptr, &state.nv12);
    if (FAILED(status)) { set_hresult_error("CreateTexture2D(NV12)", status); return false; }
    D3D11_VIDEO_PROCESSOR_OUTPUT_VIEW_DESC view_desc{};
    view_desc.ViewDimension = D3D11_VPOV_DIMENSION_TEXTURE2D;
    view_desc.Texture2D.MipSlice = 0;
    status = state.video_device->CreateVideoProcessorOutputView(
        state.nv12.Get(), state.enumerator.Get(), &view_desc, &state.output_view);
    if (FAILED(status)) { set_hresult_error("CreateVideoProcessorOutputView", status); return false; }
    state.width = width;
    state.height = height;
    return true;
}

bool convert(State& state) {
    D3D11_VIDEO_PROCESSOR_INPUT_VIEW_DESC input_desc{};
    input_desc.ViewDimension = D3D11_VPIV_DIMENSION_TEXTURE2D;
    input_desc.Texture2D.MipSlice = 0;
    ComPtr<ID3D11VideoProcessorInputView> input_view;
    ID3D11Texture2D* source = state.external_bgra
        ? state.external_bgra.Get()
        : state.bgra.Get();
    HRESULT status = state.video_device->CreateVideoProcessorInputView(
        source, state.enumerator.Get(), &input_desc, &input_view);
    if (FAILED(status)) { set_hresult_error("CreateVideoProcessorInputView(BGRA8)", status); return false; }
    state.video_context->VideoProcessorSetStreamFrameFormat(
        state.processor.Get(), 0, D3D11_VIDEO_FRAME_FORMAT_PROGRESSIVE);
    D3D11_VIDEO_PROCESSOR_STREAM stream{};
    stream.Enable = TRUE;
    stream.pInputSurface = input_view.Get();
    status = state.video_context->VideoProcessorBlt(
        state.processor.Get(), state.output_view.Get(), 0, 1, &stream);
    if (FAILED(status)) { set_hresult_error("VideoProcessorBlt(BGRA8->NV12)", status); return false; }
    state.context->Flush();
    return true;
}
}

extern "C" D2S_D3D11_SURFACE_API int d2s_d3d11_sbs_surface_probe(void) {
    return 0x07; // + external same-device BGRA texture import.
}

extern "C" D2S_D3D11_SURFACE_API void* d2s_d3d11_sbs_surface_create(
    int width, int height, int adapter_index) {
    g_last_error.clear();
    auto state = std::make_unique<State>();
    if (!initialize(*state, width, height, adapter_index)) return nullptr;
    return state.release();
}

extern "C" D2S_D3D11_SURFACE_API void* d2s_d3d11_sbs_surface_create_from_device(
    int width, int height, void* d3d11_device) {
    g_last_error.clear();
    if (!d3d11_device) {
        set_error("D3D11 device is required");
        return nullptr;
    }
    auto state = std::make_unique<State>();
    if (!initialize(*state, width, height, -1,
                    static_cast<ID3D11Device*>(d3d11_device))) {
        return nullptr;
    }
    return state.release();
}

extern "C" D2S_D3D11_SURFACE_API void* d2s_d3d11_sbs_surface_device(void* handle) {
    auto* state = static_cast<State*>(handle);
    return state ? state->device.Get() : nullptr;
}

extern "C" D2S_D3D11_SURFACE_API unsigned long long d2s_d3d11_sbs_surface_adapter_luid(void* handle) {
    auto* state = static_cast<State*>(handle);
    return state ? state->adapter_luid : 0;
}

extern "C" D2S_D3D11_SURFACE_API void* d2s_d3d11_sbs_surface_shared_handle(void* handle) {
    auto* state = static_cast<State*>(handle);
    if (!state || !state->bgra) {
        set_error("shared BGRA surface is unavailable");
        return nullptr;
    }
    ComPtr<IDXGIResource1> resource;
    HRESULT status = state->bgra.As(&resource);
    if (FAILED(status) || !resource) {
        set_hresult_error("QueryInterface(IDXGIResource1)", status);
        return nullptr;
    }
    HANDLE shared = nullptr;
    status = resource->CreateSharedHandle(
        nullptr,
        DXGI_SHARED_RESOURCE_READ | DXGI_SHARED_RESOURCE_WRITE,
        nullptr,
        &shared);
    if (FAILED(status)) {
        set_hresult_error("IDXGIResource1::CreateSharedHandle", status);
        return nullptr;
    }
    return shared;
}

extern "C" D2S_D3D11_SURFACE_API int d2s_d3d11_sbs_surface_set_bgra_texture(
    void* handle, void* bgra_texture, unsigned long long adapter_luid) {
    auto* state = static_cast<State*>(handle);
    if (!state || !bgra_texture) {
        set_error("BGRA texture is unavailable");
        return 0;
    }
    auto* texture = static_cast<ID3D11Texture2D*>(bgra_texture);
    D3D11_TEXTURE2D_DESC desc{};
    texture->GetDesc(&desc);
    if (desc.Format != DXGI_FORMAT_B8G8R8A8_UNORM ||
        static_cast<int>(desc.Width) != state->width ||
        static_cast<int>(desc.Height) != state->height) {
        set_error("external BGRA texture format or dimensions do not match");
        return 0;
    }
    ComPtr<ID3D11Device> texture_device;
    texture->GetDevice(&texture_device);
    if (!texture_device || texture_device.Get() != state->device.Get()) {
        set_error("external BGRA texture belongs to a different D3D11 device");
        return 0;
    }
    if (!adapter_luid || adapter_luid != state->adapter_luid) {
        set_error("external BGRA texture Adapter LUID does not match the surface");
        return 0;
    }
    state->external_bgra = texture;
    return convert(*state) ? 1 : 0;
}

extern "C" D2S_D3D11_SURFACE_API int d2s_d3d11_sbs_surface_upload_bgra(
    void* handle, const unsigned char* data, int stride, int width, int height) {
    auto* state = static_cast<State*>(handle);
    if (!state || !data || stride < width * 4 || width != state->width || height != state->height) {
        set_error("BGRA upload dimensions or stride do not match the surface");
        return 0;
    }
    state->external_bgra.Reset();
    D3D11_MAPPED_SUBRESOURCE mapped{};
    HRESULT status = state->context->Map(state->staging_bgra.Get(), 0, D3D11_MAP_WRITE, 0, &mapped);
    if (FAILED(status)) { set_hresult_error("Map(BGRA8 staging)", status); return 0; }
    for (int row = 0; row < height; ++row) {
        std::memcpy(static_cast<unsigned char*>(mapped.pData) + row * mapped.RowPitch,
                    data + row * stride, static_cast<size_t>(width) * 4);
    }
    state->context->Unmap(state->staging_bgra.Get(), 0);
    state->context->CopyResource(state->bgra.Get(), state->staging_bgra.Get());
    return convert(*state) ? 1 : 0;
}

extern "C" D2S_D3D11_SURFACE_API int d2s_d3d11_sbs_surface_nv12(
    void* handle, void** texture, int* width, int* height) {
    auto* state = static_cast<State*>(handle);
    if (!state || !texture || !width || !height || !state->nv12) {
        set_error("NV12 surface is unavailable");
        return 0;
    }
    *texture = state->nv12.Get();
    *width = state->width;
    *height = state->height;
    return 1;
}

extern "C" D2S_D3D11_SURFACE_API int d2s_d3d11_sbs_surface_last_error(char* output, int capacity) {
    if (!output || capacity <= 0) return static_cast<int>(g_last_error.size());
    const int count = static_cast<int>(g_last_error.size());
    const int copied = count < capacity - 1 ? count : capacity - 1;
    std::memcpy(output, g_last_error.data(), static_cast<size_t>(copied));
    output[copied] = '\0';
    return count;
}

extern "C" D2S_D3D11_SURFACE_API void d2s_d3d11_sbs_surface_destroy(void* handle) {
    delete static_cast<State*>(handle);
}

#else
extern "C" int d2s_d3d11_sbs_surface_probe(void) { return 0; }
extern "C" void* d2s_d3d11_sbs_surface_create(int, int, int) { return nullptr; }
extern "C" void* d2s_d3d11_sbs_surface_create_from_device(int, int, void*) { return nullptr; }
extern "C" void* d2s_d3d11_sbs_surface_device(void*) { return nullptr; }
extern "C" unsigned long long d2s_d3d11_sbs_surface_adapter_luid(void*) { return 0; }
extern "C" void* d2s_d3d11_sbs_surface_shared_handle(void*) { return nullptr; }
extern "C" int d2s_d3d11_sbs_surface_set_bgra_texture(void*, void*, unsigned long long) { return 0; }
extern "C" int d2s_d3d11_sbs_surface_upload_bgra(void*, const unsigned char*, int, int, int) { return 0; }
extern "C" int d2s_d3d11_sbs_surface_nv12(void*, void**, int*, int*) { return 0; }
extern "C" int d2s_d3d11_sbs_surface_last_error(char*, int) { return 0; }
extern "C" void d2s_d3d11_sbs_surface_destroy(void*) {}
#endif
