#include "amd_encoder.h"

#include <d3d11.h>
#include <windows.h>

#include <cstring>
#include <string>

#ifdef D2S_AMF_SDK
#include "public/common/AMFFactory.h"
#include "public/include/components/VideoEncoderVCE.h"
#include "public/include/components/VideoEncoderHEVC.h"
#endif

namespace {
thread_local std::string g_last_error;

void set_error(const char* message) {
    g_last_error = message ? message : "unknown AMD encoder error";
}

bool amf_runtime_present() {
    // AMF is loaded dynamically: the AMD display driver owns the runtime DLL.
    // The SDK headers are only required by the future surface encoder layer.
    HMODULE module = LoadLibraryW(L"amfrt64.dll");
    if (!module) {
        set_error("amfrt64.dll was not found; install an AMD Adrenalin driver");
        return false;
    }
    FreeLibrary(module);
    return true;
}

#ifdef D2S_AMF_SDK
struct AmdEncoder {
    amf::AMFContextPtr context;
    amf::AMFComponentPtr encoder;
    int width = 0;
    int height = 0;
    int fps = 0;
    bool hevc = false;
};
#endif
}  // namespace

extern "C" D2S_AMD_API int d2s_amd_encoder_probe() {
    if (!amf_runtime_present()) {
        return 0;
    }

    IDXGIFactory* factory = nullptr;
    HRESULT hr = CreateDXGIFactory(__uuidof(IDXGIFactory), reinterpret_cast<void**>(&factory));
    if (FAILED(hr) || !factory) {
        set_error("DXGI factory creation failed");
        return 0;
    }

    IDXGIAdapter* adapter = nullptr;
    bool found_amd = false;
    for (UINT index = 0; factory->EnumAdapters(index, &adapter) != DXGI_ERROR_NOT_FOUND; ++index) {
        DXGI_ADAPTER_DESC desc{};
        if (SUCCEEDED(adapter->GetDesc(&desc))) {
            const std::wstring name(desc.Description);
            if (name.find(L"AMD") != std::wstring::npos ||
                name.find(L"Radeon") != std::wstring::npos) {
                found_amd = true;
                adapter->Release();
                break;
            }
        }
        adapter->Release();
        adapter = nullptr;
    }
    factory->Release();
    if (!found_amd) {
        set_error("no AMD/Radeon DXGI adapter was found");
        return 0;
    }
    g_last_error.clear();
    return 1;
}

extern "C" D2S_AMD_API int d2s_amd_encoder_last_error(char* output, int capacity) {
    if (!output || capacity <= 0) {
        return static_cast<int>(g_last_error.size());
    }
    const int count = static_cast<int>(g_last_error.size());
    const int copied = (count < capacity - 1) ? count : capacity - 1;
    std::memcpy(output, g_last_error.data(), copied);
    output[copied] = '\0';
    return count;
}

extern "C" D2S_AMD_API void* d2s_amd_encoder_create(
    int width, int height, int fps, int bitrate, int hevc) {
#ifndef D2S_AMF_SDK
    (void)width; (void)height; (void)fps; (void)bitrate; (void)hevc;
    set_error("AMD bridge was built without the AMF SDK headers");
    return nullptr;
#else
    if (!amf_runtime_present()) return nullptr;
    auto* state = new AmdEncoder();
    state->width = width; state->height = height; state->fps = fps; state->hevc = hevc != 0;
    AMF_RESULT result = g_AMFFactory.Init();
    if (result != AMF_OK) { set_error("AMF factory initialization failed"); delete state; return nullptr; }
    result = g_AMFFactory.GetFactory()->CreateContext(&state->context);
    if (result != AMF_OK || !state->context) { set_error("AMF context creation failed"); delete state; return nullptr; }
    result = state->context->InitDX11(nullptr);
    if (result != AMF_OK) { set_error("AMF DX11 context initialization failed"); delete state; return nullptr; }
    const wchar_t* component = state->hevc ? AMFVideoEncoder_HEVC : AMFVideoEncoderVCE_AVC;
    result = g_AMFFactory.GetFactory()->CreateComponent(state->context, component, &state->encoder);
    if (result != AMF_OK || !state->encoder) { set_error("AMF video encoder component creation failed"); delete state; return nullptr; }
    const amf_int64 rate = static_cast<amf_int64>(bitrate);
    state->encoder->SetProperty(AMF_VIDEO_ENCODER_FRAMESIZE, AMFConstructSize(width, height));
    state->encoder->SetProperty(AMF_VIDEO_ENCODER_FRAMERATE, AMFConstructRate(fps, 1));
    state->encoder->SetProperty(AMF_VIDEO_ENCODER_TARGET_BITRATE, rate);
    result = state->encoder->Init(amf::AMF_SURFACE_NV12, width, height);
    if (result != AMF_OK) { set_error("AMF encoder initialization failed"); delete state; return nullptr; }
    g_last_error.clear();
    return state;
#endif
}

extern "C" D2S_AMD_API int d2s_amd_encoder_submit_texture(void* handle, void* d3d11_texture) {
#ifndef D2S_AMF_SDK
    (void)handle; (void)d3d11_texture; return -1;
#else
    auto* state = static_cast<AmdEncoder*>(handle);
    if (!state || !state->context || !state->encoder || !d3d11_texture) return -1;
    amf::AMFSurfacePtr surface;
    AMF_RESULT result = state->context->CreateSurfaceFromDX11Native(d3d11_texture, &surface, nullptr);
    if (result != AMF_OK || !surface) { set_error("AMF could not wrap the D3D11 texture"); return -1; }
    result = state->encoder->SubmitInput(surface);
    return result == AMF_OK || result == AMF_NEED_MORE_INPUT ? 1 : 0;
#endif
}

extern "C" D2S_AMD_API int d2s_amd_encoder_read_packet(void* handle, void* output, int capacity) {
#ifndef D2S_AMF_SDK
    (void)handle; (void)output; (void)capacity; return -1;
#else
    auto* state = static_cast<AmdEncoder*>(handle);
    if (!state || !output || capacity <= 0) return -1;
    amf::AMFDataPtr data;
    AMF_RESULT result = state->encoder->QueryOutput(&data);
    if (result != AMF_OK || !data) return 0;
    amf::AMFBufferPtr buffer(data);
    const auto size = static_cast<int>(buffer->GetSize());
    if (size > capacity) return -size;
    std::memcpy(output, buffer->GetNative(), static_cast<size_t>(size));
    return size;
#endif
}

extern "C" D2S_AMD_API void d2s_amd_encoder_destroy(void* handle) {
#ifdef D2S_AMF_SDK
    auto* state = static_cast<AmdEncoder*>(handle);
    if (!state) return;
    if (state->encoder) state->encoder->Terminate();
    if (state->context) state->context->Terminate();
    delete state;
    g_AMFFactory.Terminate();
#else
    (void)handle;
#endif
}
