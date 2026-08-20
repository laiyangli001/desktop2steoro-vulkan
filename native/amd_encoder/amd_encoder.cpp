#include "amd_encoder.h"

#include <d3d11.h>
#include <windows.h>

#include <cstring>
#include <string>

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

