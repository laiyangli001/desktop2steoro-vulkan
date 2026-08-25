#include "onevpl_d3d11_encoder.h"

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <d3d11.h>
#include <dxgi.h>
#include <wrl/client.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#ifdef D2S_ONEVPL_SDK
#include <vpl/mfx.h>
#include <vpl/mfxvideo.h>
#endif

namespace {
using Microsoft::WRL::ComPtr;
thread_local std::string g_error;

void set_error(const char* message) {
    g_error = message ? message : "oneVPL D3D11 encoder error";
}

void set_error(const std::string& message) {
    g_error = message;
}

unsigned long long get_adapter_luid(ID3D11Device* device) {
    if (!device) return 0;
    ComPtr<IDXGIDevice> dxgi_device;
    if (FAILED(device->QueryInterface(IID_PPV_ARGS(&dxgi_device)))) return 0;
    ComPtr<IDXGIAdapter> adapter;
    if (FAILED(dxgi_device->GetAdapter(&adapter))) return 0;
    DXGI_ADAPTER_DESC desc{};
    if (FAILED(adapter->GetDesc(&desc))) return 0;
    return (static_cast<unsigned long long>(
                static_cast<unsigned long>(desc.AdapterLuid.HighPart)) << 32) |
           static_cast<unsigned long long>(
                static_cast<unsigned long>(desc.AdapterLuid.LowPart));
}

struct State {
    ComPtr<ID3D11Device> device;
    unsigned long long adapter_luid = 0;
    int width = 0;
    int height = 0;
    int fps = 0;
    int bitrate = 0;
    bool hevc = false;
#ifdef D2S_ONEVPL_SDK
    mfxLoader loader = nullptr;
    mfxSession session = nullptr;
    mfxVideoParam params{};
    mfxBitstream bitstream{};
    std::vector<mfxU8> bitstream_storage;
    std::vector<unsigned char> packet;
#endif
};

#ifdef D2S_ONEVPL_SDK
bool configure_session(State& state) {
    state.loader = MFXLoad();
    if (!state.loader) { set_error("MFXLoad failed"); return false; }

    mfxConfig acceleration = MFXCreateConfig(state.loader);
    if (!acceleration) { set_error("MFXCreateConfig failed"); return false; }
    mfxVariant value{};
    value.Type = MFX_VARIANT_TYPE_U32;
    value.Data.U32 = MFX_ACCEL_MODE_VIA_D3D11;
    auto status = MFXSetConfigFilterProperty(
        acceleration,
        reinterpret_cast<mfxU8*>(const_cast<char*>("mfxImplDescription.AccelerationMode")),
        value);
    if (status != MFX_ERR_NONE) {
        set_error("oneVPL D3D11 acceleration filter failed");
        return false;
    }
    status = MFXCreateSession(state.loader, 0, &state.session);
    if (status != MFX_ERR_NONE || !state.session) {
        set_error("no oneVPL implementation with D3D11 acceleration is available");
        return false;
    }
    status = MFXVideoCORE_SetHandle(
        state.session, MFX_HANDLE_D3D11_DEVICE, state.device.Get());
    if (status != MFX_ERR_NONE) {
        set_error("MFXVideoCORE_SetHandle(D3D11) failed");
        return false;
    }

    auto& video = state.params.mfx;
    video.CodecId = state.hevc ? MFX_CODEC_HEVC : MFX_CODEC_AVC;
    video.TargetUsage = MFX_TARGETUSAGE_BEST_SPEED;
    video.RateControlMethod = MFX_RATECONTROL_CBR;
    video.TargetKbps = static_cast<mfxU16>(
        std::clamp(state.bitrate / 1000, 1, 65535));
    video.FrameInfo.FourCC = MFX_FOURCC_NV12;
    video.FrameInfo.ChromaFormat = MFX_CHROMAFORMAT_YUV420;
    video.FrameInfo.PicStruct = MFX_PICSTRUCT_PROGRESSIVE;
    video.FrameInfo.CropW = static_cast<mfxU16>(state.width);
    video.FrameInfo.CropH = static_cast<mfxU16>(state.height);
    video.FrameInfo.Width = static_cast<mfxU16>((state.width + 15) & ~15);
    video.FrameInfo.Height = static_cast<mfxU16>((state.height + 15) & ~15);
    video.FrameInfo.FrameRateExtN = static_cast<mfxU32>(state.fps);
    video.FrameInfo.FrameRateExtD = 1;
    state.params.IOPattern = MFX_IOPATTERN_IN_VIDEO_MEMORY;
    status = MFXVideoENCODE_Init(state.session, &state.params);
    if (status != MFX_ERR_NONE) {
        set_error("MFXVideoENCODE_Init failed");
        return false;
    }
    state.bitstream_storage.resize(4 * 1024 * 1024);
    state.bitstream.MaxLength = static_cast<mfxU32>(state.bitstream_storage.size());
    state.bitstream.Data = state.bitstream_storage.data();
    return true;
}
#endif
}

extern "C" D2S_ONEVPL_API int d2s_onevpl_d3d11_probe() {
    g_error.clear();
#ifdef D2S_ONEVPL_SDK
    mfxLoader loader = MFXLoad();
    if (!loader) { set_error("oneVPL dispatcher is unavailable"); return 0; }
    MFXUnload(loader);
    return 1;
#else
    set_error("oneVPL SDK was not configured at build time");
    return 0;
#endif
}

extern "C" D2S_ONEVPL_API int d2s_onevpl_d3d11_last_error(
    char* output, int capacity) {
    if (!output || capacity <= 0) return static_cast<int>(g_error.size());
    const int count = static_cast<int>(g_error.size());
    const int copied = std::min(count, capacity - 1);
    std::memcpy(output, g_error.data(), static_cast<size_t>(copied));
    output[copied] = '\0';
    return count;
}

extern "C" D2S_ONEVPL_API void* d2s_onevpl_d3d11_create(
    int width, int height, int fps, int bitrate, int hevc, void* d3d11_device) {
    g_error.clear();
    if (!d3d11_device || width <= 0 || height <= 0 || fps <= 0) {
        set_error("invalid encoder dimensions or D3D11 device");
        return nullptr;
    }
#ifndef D2S_ONEVPL_SDK
    set_error("oneVPL SDK was not configured at build time");
    return nullptr;
#else
    auto* state = new State();
    state->device = static_cast<ID3D11Device*>(d3d11_device);
    state->adapter_luid = get_adapter_luid(state->device.Get());
    state->width = width;
    state->height = height;
    state->fps = fps;
    state->bitrate = bitrate;
    state->hevc = hevc != 0;
    if (!state->adapter_luid || !configure_session(*state)) {
        d2s_onevpl_d3d11_destroy(state);
        return nullptr;
    }
    return state;
#endif
}

extern "C" D2S_ONEVPL_API unsigned long long d2s_onevpl_d3d11_adapter_luid(
    void* handle) {
    auto* state = static_cast<State*>(handle);
    return state ? state->adapter_luid : 0;
}

extern "C" D2S_ONEVPL_API int d2s_onevpl_d3d11_submit_nv12(
    void* handle, void* nv12_texture, long long timestamp) {
#ifndef D2S_ONEVPL_SDK
    (void)handle; (void)nv12_texture; (void)timestamp;
    set_error("oneVPL SDK was not configured at build time");
    return 0;
#else
    auto* state = static_cast<State*>(handle);
    if (!state || !nv12_texture) { set_error("invalid NV12 surface"); return 0; }
    auto* texture = static_cast<ID3D11Texture2D*>(nv12_texture);
    ComPtr<ID3D11Device> texture_device;
    texture->GetDevice(&texture_device);
    if (!texture_device || texture_device.Get() != state->device.Get()) {
        set_error("NV12 texture belongs to a different D3D11 device");
        return 0;
    }
    D3D11_TEXTURE2D_DESC desc{};
    texture->GetDesc(&desc);
    if (desc.Format != DXGI_FORMAT_NV12 ||
        static_cast<int>(desc.Width) < state->width ||
        static_cast<int>(desc.Height) < state->height) {
        set_error("QSV surface must be an NV12 texture with matching dimensions");
        return 0;
    }
    if (get_adapter_luid(state->device.Get()) != state->adapter_luid) {
        set_error("D3D11 adapter changed during oneVPL session");
        return 0;
    }
    mfxHDLPair native_handle{};
    native_handle.first = nv12_texture;
    native_handle.second = reinterpret_cast<mfxHDL>(
        static_cast<uintptr_t>(MFX_INFINITE));
    mfxFrameSurface1 surface{};
    surface.Info = state->params.mfx.FrameInfo;
    surface.Data.MemId = reinterpret_cast<mfxMemId>(&native_handle);
    surface.Data.TimeStamp = static_cast<mfxU64>(timestamp);
    mfxSyncPoint sync_point = nullptr;
    const mfxStatus status = MFXVideoENCODE_EncodeFrameAsync(
        state->session, nullptr, &surface, &state->bitstream, &sync_point);
    if (status == MFX_ERR_MORE_DATA) return 1;
    if (status != MFX_ERR_NONE || !sync_point) {
        set_error("MFXVideoENCODE_EncodeFrameAsync rejected the NV12 surface");
        return 0;
    }
    const mfxStatus sync_status = MFXVideoCORE_SyncOperation(
        state->session, sync_point, 5000);
    if (sync_status != MFX_ERR_NONE) {
        set_error("MFXVideoCORE_SyncOperation failed");
        return 0;
    }
    state->packet.clear();
    if (state->bitstream.DataLength > 0) {
        state->packet.assign(
            state->bitstream.Data + state->bitstream.DataOffset,
            state->bitstream.Data + state->bitstream.DataOffset + state->bitstream.DataLength);
        state->bitstream.DataOffset = 0;
        state->bitstream.DataLength = 0;
    }
    return 1;
#endif
}

extern "C" D2S_ONEVPL_API int d2s_onevpl_d3d11_read_packet(
    void* handle, void* output, int capacity) {
#ifndef D2S_ONEVPL_SDK
    (void)handle; (void)output; (void)capacity;
    return 0;
#else
    auto* state = static_cast<State*>(handle);
    if (!state || !output || capacity <= 0) { set_error("invalid packet buffer"); return -1; }
    const int count = static_cast<int>(state->packet.size());
    if (count > capacity) { set_error("packet buffer is too small"); return -1; }
    if (count) std::memcpy(output, state->packet.data(), static_cast<size_t>(count));
    state->packet.clear();
    return count;
#endif
}

extern "C" D2S_ONEVPL_API void d2s_onevpl_d3d11_destroy(void* handle) {
    auto* state = static_cast<State*>(handle);
    if (!state) return;
#ifdef D2S_ONEVPL_SDK
    if (state->session) { MFXVideoENCODE_Close(state->session); MFXClose(state->session); }
    if (state->loader) MFXUnload(state->loader);
#endif
    delete state;
}
#endif
