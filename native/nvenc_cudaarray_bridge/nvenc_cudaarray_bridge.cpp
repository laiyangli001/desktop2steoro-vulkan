#include "nvenc_cudaarray_bridge.h"

#include <windows.h>
#include <nvEncodeAPI.h>

#include <algorithm>
#include <cstring>
#include <deque>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

thread_local std::string g_last_error;

std::string status_text(NVENCSTATUS status) {
    return "NVENC status " + std::to_string(static_cast<int>(status));
}

class NvencError final : public std::runtime_error {
public:
    explicit NvencError(const std::string& message) : std::runtime_error(message) {}
};

void require_status(NVENCSTATUS status, const char* operation) {
    if (status != NV_ENC_SUCCESS) {
        throw NvencError(std::string(operation) + " failed: " + status_text(status));
    }
}

class NvencCudaArrayEncoder {
public:
    NvencCudaArrayEncoder(
        uint32_t width,
        uint32_t height,
        uint32_t fps,
        uint32_t bitrate,
        bool hevc,
        void* cuda_array
    ) : width_(width),
        height_(height),
        fps_(std::max<uint32_t>(1, fps)),
        bitrate_(std::max<uint32_t>(1, bitrate)),
        hevc_(hevc),
        cuda_array_(cuda_array) {
        if (width_ < 2 || height_ < 2 || (width_ & 1U) || (height_ & 1U)) {
            throw NvencError("NVENC CUDAARRAY requires positive even dimensions");
        }
        if (!cuda_array_) {
            throw NvencError("CUDA array pointer is null");
        }
        try {
            open();
        } catch (...) {
            close_noexcept();
            throw;
        }
    }

    ~NvencCudaArrayEncoder() {
        close_noexcept();
    }

    void submit(int64_t timestamp) {
        if (!encoder_ || !registered_ || !bitstream_) {
            throw NvencError("NVENC CUDAARRAY encoder is not initialized");
        }

        NV_ENC_MAP_INPUT_RESOURCE map{};
        map.version = NV_ENC_MAP_INPUT_RESOURCE_VER;
        map.registeredResource = registered_;
        require_status(
            api_.nvEncMapInputResource(encoder_, &map),
            "nvEncMapInputResource"
        );

        NVENCSTATUS encode_status = NV_ENC_SUCCESS;
        try {
            NV_ENC_PIC_PARAMS picture{};
            picture.version = NV_ENC_PIC_PARAMS_VER;
            picture.inputBuffer = map.mappedResource;
            picture.bufferFmt = NV_ENC_BUFFER_FORMAT_ABGR;
            picture.inputWidth = width_;
            picture.inputHeight = height_;
            picture.outputBitstream = bitstream_;
            picture.pictureStruct = NV_ENC_PIC_STRUCT_FRAME;
            picture.inputTimeStamp = static_cast<uint64_t>(timestamp);
            encode_status = api_.nvEncEncodePicture(encoder_, &picture);
            if (
                encode_status != NV_ENC_SUCCESS &&
                encode_status != NV_ENC_ERR_NEED_MORE_INPUT
            ) {
                require_status(encode_status, "nvEncEncodePicture");
            }
            if (encode_status == NV_ENC_SUCCESS) {
                collect_packet();
            }
        } catch (...) {
            api_.nvEncUnmapInputResource(encoder_, map.mappedResource);
            throw;
        }
        require_status(
            api_.nvEncUnmapInputResource(encoder_, map.mappedResource),
            "nvEncUnmapInputResource"
        );
    }

    void flush() {
        if (!encoder_ || flushed_) {
            return;
        }
        NV_ENC_PIC_PARAMS picture{};
        picture.version = NV_ENC_PIC_PARAMS_VER;
        picture.encodePicFlags = NV_ENC_PIC_FLAG_EOS;
        const NVENCSTATUS status = api_.nvEncEncodePicture(encoder_, &picture);
        if (
            status != NV_ENC_SUCCESS &&
            status != NV_ENC_ERR_NEED_MORE_INPUT
        ) {
            require_status(status, "nvEncEncodePicture(EOS)");
        }
        flushed_ = true;
    }

    int read_packet(uint8_t* destination, size_t capacity, size_t* packet_size) {
        if (!packet_size) {
            throw NvencError("packet_size pointer is null");
        }
        if (packets_.empty()) {
            *packet_size = 0;
            return 0;
        }
        const auto& packet = packets_.front();
        *packet_size = packet.size();
        if (!destination || capacity == 0) {
            return 1;
        }
        if (capacity < packet.size()) {
            return 2;
        }
        std::memcpy(destination, packet.data(), packet.size());
        packets_.pop_front();
        return 1;
    }

private:
    using NvEncodeAPICreateInstanceFn =
        NVENCSTATUS (NVENCAPI*)(NV_ENCODE_API_FUNCTION_LIST*);
    using NvEncodeAPIGetMaxSupportedVersionFn =
        NVENCSTATUS (NVENCAPI*)(uint32_t*);
    using CuCtxGetCurrentFn = int (WINAPI*)(void**);

    void open() {
        nvenc_module_ = LoadLibraryW(L"nvEncodeAPI64.dll");
        if (!nvenc_module_) {
            throw NvencError("nvEncodeAPI64.dll is unavailable");
        }
        auto get_max_version = reinterpret_cast<NvEncodeAPIGetMaxSupportedVersionFn>(
            GetProcAddress(nvenc_module_, "NvEncodeAPIGetMaxSupportedVersion")
        );
        auto create_instance = reinterpret_cast<NvEncodeAPICreateInstanceFn>(
            GetProcAddress(nvenc_module_, "NvEncodeAPICreateInstance")
        );
        if (!get_max_version || !create_instance) {
            throw NvencError("NVENC API entry points are unavailable");
        }
        uint32_t driver_version = 0;
        require_status(
            get_max_version(&driver_version),
            "NvEncodeAPIGetMaxSupportedVersion"
        );
        const uint32_t header_version =
            (NVENCAPI_MAJOR_VERSION << 4) | NVENCAPI_MINOR_VERSION;
        if (header_version > driver_version) {
            throw NvencError(
                "NVIDIA driver NVENC API is older than the bridge headers"
            );
        }

        api_ = {};
        api_.version = NV_ENCODE_API_FUNCTION_LIST_VER;
        require_status(create_instance(&api_), "NvEncodeAPICreateInstance");

        cuda_module_ = LoadLibraryW(L"nvcuda.dll");
        if (!cuda_module_) {
            throw NvencError("nvcuda.dll is unavailable");
        }
        auto cu_ctx_get_current = reinterpret_cast<CuCtxGetCurrentFn>(
            GetProcAddress(cuda_module_, "cuCtxGetCurrent")
        );
        if (!cu_ctx_get_current) {
            throw NvencError("cuCtxGetCurrent is unavailable");
        }
        void* cuda_context = nullptr;
        const int cuda_status = cu_ctx_get_current(&cuda_context);
        if (cuda_status != 0 || !cuda_context) {
            throw NvencError(
                "no current CUDA context; call from the active PyTorch CUDA thread"
            );
        }

        NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS session{};
        session.version = NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS_VER;
        session.deviceType = NV_ENC_DEVICE_TYPE_CUDA;
        session.device = cuda_context;
        session.apiVersion = NVENCAPI_VERSION;
        require_status(
            api_.nvEncOpenEncodeSessionEx(&session, &encoder_),
            "nvEncOpenEncodeSessionEx"
        );

        const GUID codec_guid =
            hevc_ ? NV_ENC_CODEC_HEVC_GUID : NV_ENC_CODEC_H264_GUID;
        NV_ENC_PRESET_CONFIG preset{};
        preset.version = NV_ENC_PRESET_CONFIG_VER;
        preset.presetCfg.version = NV_ENC_CONFIG_VER;
        require_status(
            api_.nvEncGetEncodePresetConfigEx(
                encoder_,
                codec_guid,
                NV_ENC_PRESET_P1_GUID,
                NV_ENC_TUNING_INFO_ULTRA_LOW_LATENCY,
                &preset
            ),
            "nvEncGetEncodePresetConfigEx"
        );

        NV_ENC_CONFIG config = preset.presetCfg;
        config.version = NV_ENC_CONFIG_VER;
        config.gopLength = fps_;
        config.frameIntervalP = 1;
        config.rcParams.rateControlMode = NV_ENC_PARAMS_RC_CBR;
        config.rcParams.averageBitRate = bitrate_;
        config.rcParams.maxBitRate = std::max<uint32_t>(
            bitrate_,
            static_cast<uint32_t>(static_cast<uint64_t>(bitrate_) * 12U / 10U)
        );
        config.rcParams.enableLookahead = 0;
        config.rcParams.enableAQ = 0;
        config.rcParams.zeroReorderDelay = 1;
        if (hevc_) {
            config.encodeCodecConfig.hevcConfig.idrPeriod = fps_;
            config.encodeCodecConfig.hevcConfig.repeatSPSPPS = 1;
        } else {
            config.encodeCodecConfig.h264Config.idrPeriod = fps_;
            config.encodeCodecConfig.h264Config.repeatSPSPPS = 1;
        }

        NV_ENC_INITIALIZE_PARAMS init{};
        init.version = NV_ENC_INITIALIZE_PARAMS_VER;
        init.encodeGUID = codec_guid;
        init.presetGUID = NV_ENC_PRESET_P1_GUID;
        init.encodeWidth = width_;
        init.encodeHeight = height_;
        init.darWidth = width_;
        init.darHeight = height_;
        init.frameRateNum = fps_;
        init.frameRateDen = 1;
        init.enablePTD = 1;
        init.enableEncodeAsync = 0;
        init.enableWeightedPrediction = 0;
        init.maxEncodeWidth = width_;
        init.maxEncodeHeight = height_;
        init.tuningInfo = NV_ENC_TUNING_INFO_ULTRA_LOW_LATENCY;
        init.encodeConfig = &config;
        require_status(
            api_.nvEncInitializeEncoder(encoder_, &init),
            "nvEncInitializeEncoder"
        );

        NV_ENC_CREATE_BITSTREAM_BUFFER create_bitstream{};
        create_bitstream.version = NV_ENC_CREATE_BITSTREAM_BUFFER_VER;
        require_status(
            api_.nvEncCreateBitstreamBuffer(encoder_, &create_bitstream),
            "nvEncCreateBitstreamBuffer"
        );
        bitstream_ = create_bitstream.bitstreamBuffer;

        NV_ENC_REGISTER_RESOURCE resource{};
        resource.version = NV_ENC_REGISTER_RESOURCE_VER;
        resource.resourceType = NV_ENC_INPUT_RESOURCE_TYPE_CUDAARRAY;
        resource.width = width_;
        resource.height = height_;
        resource.pitch = 0;
        resource.resourceToRegister = cuda_array_;
        resource.bufferFormat = NV_ENC_BUFFER_FORMAT_ABGR;
        resource.bufferUsage = NV_ENC_INPUT_IMAGE;
        require_status(
            api_.nvEncRegisterResource(encoder_, &resource),
            "nvEncRegisterResource(CUDAARRAY)"
        );
        registered_ = resource.registeredResource;
    }

    void collect_packet() {
        NV_ENC_LOCK_BITSTREAM lock{};
        lock.version = NV_ENC_LOCK_BITSTREAM_VER;
        lock.outputBitstream = bitstream_;
        lock.doNotWait = 0;
        require_status(
            api_.nvEncLockBitstream(encoder_, &lock),
            "nvEncLockBitstream"
        );
        try {
            const auto* begin = static_cast<const uint8_t*>(lock.bitstreamBufferPtr);
            packets_.emplace_back(begin, begin + lock.bitstreamSizeInBytes);
        } catch (...) {
            api_.nvEncUnlockBitstream(encoder_, bitstream_);
            throw;
        }
        require_status(
            api_.nvEncUnlockBitstream(encoder_, bitstream_),
            "nvEncUnlockBitstream"
        );
    }

    void close_noexcept() noexcept {
        try {
            flush();
        } catch (...) {
        }
        if (encoder_ && registered_) {
            api_.nvEncUnregisterResource(encoder_, registered_);
            registered_ = nullptr;
        }
        if (encoder_ && bitstream_) {
            api_.nvEncDestroyBitstreamBuffer(encoder_, bitstream_);
            bitstream_ = nullptr;
        }
        if (encoder_) {
            api_.nvEncDestroyEncoder(encoder_);
            encoder_ = nullptr;
        }
        if (cuda_module_) {
            FreeLibrary(cuda_module_);
            cuda_module_ = nullptr;
        }
        if (nvenc_module_) {
            FreeLibrary(nvenc_module_);
            nvenc_module_ = nullptr;
        }
    }

    uint32_t width_;
    uint32_t height_;
    uint32_t fps_;
    uint32_t bitrate_;
    bool hevc_;
    void* cuda_array_;
    HMODULE nvenc_module_ = nullptr;
    HMODULE cuda_module_ = nullptr;
    NV_ENCODE_API_FUNCTION_LIST api_{};
    void* encoder_ = nullptr;
    NV_ENC_REGISTERED_PTR registered_ = nullptr;
    NV_ENC_OUTPUT_PTR bitstream_ = nullptr;
    bool flushed_ = false;
    std::deque<std::vector<uint8_t>> packets_;
};

template <typename Fn>
int32_t protect(Fn&& fn) {
    try {
        fn();
        g_last_error.clear();
        return 0;
    } catch (const std::exception& exc) {
        g_last_error = exc.what();
        return -1;
    } catch (...) {
        g_last_error = "unknown native NVENC CUDAARRAY error";
        return -1;
    }
}

}  // namespace

extern "C" {

uint32_t d2s_nvenc_cudaarray_abi_version(void) {
    return 1;
}

int32_t d2s_nvenc_cudaarray_probe(void) {
    return protect([] {
        HMODULE nvenc = LoadLibraryW(L"nvEncodeAPI64.dll");
        if (!nvenc) {
            throw NvencError("nvEncodeAPI64.dll is unavailable");
        }
        FreeLibrary(nvenc);
        HMODULE cuda = LoadLibraryW(L"nvcuda.dll");
        if (!cuda) {
            throw NvencError("nvcuda.dll is unavailable");
        }
        FreeLibrary(cuda);
    });
}

d2s_nvenc_cudaarray_handle d2s_nvenc_cudaarray_create(
    uint32_t width,
    uint32_t height,
    uint32_t fps,
    uint32_t bitrate,
    int32_t hevc,
    uint64_t cuda_array
) {
    try {
        auto encoder = std::make_unique<NvencCudaArrayEncoder>(
            width,
            height,
            fps,
            bitrate,
            hevc != 0,
            reinterpret_cast<void*>(static_cast<uintptr_t>(cuda_array))
        );
        g_last_error.clear();
        return encoder.release();
    } catch (const std::exception& exc) {
        g_last_error = exc.what();
        return nullptr;
    } catch (...) {
        g_last_error = "unknown native NVENC CUDAARRAY create error";
        return nullptr;
    }
}

int32_t d2s_nvenc_cudaarray_submit(
    d2s_nvenc_cudaarray_handle handle,
    int64_t timestamp
) {
    return protect([&] {
        if (!handle) {
            throw NvencError("NVENC CUDAARRAY handle is null");
        }
        static_cast<NvencCudaArrayEncoder*>(handle)->submit(timestamp);
    });
}

int32_t d2s_nvenc_cudaarray_read_packet(
    d2s_nvenc_cudaarray_handle handle,
    uint8_t* destination,
    size_t capacity,
    size_t* packet_size
) {
    try {
        if (!handle) {
            throw NvencError("NVENC CUDAARRAY handle is null");
        }
        const int result = static_cast<NvencCudaArrayEncoder*>(handle)->read_packet(
            destination,
            capacity,
            packet_size
        );
        g_last_error.clear();
        return result;
    } catch (const std::exception& exc) {
        g_last_error = exc.what();
        return -1;
    } catch (...) {
        g_last_error = "unknown native NVENC CUDAARRAY packet error";
        return -1;
    }
}

int32_t d2s_nvenc_cudaarray_flush(d2s_nvenc_cudaarray_handle handle) {
    return protect([&] {
        if (!handle) {
            throw NvencError("NVENC CUDAARRAY handle is null");
        }
        static_cast<NvencCudaArrayEncoder*>(handle)->flush();
    });
}

void d2s_nvenc_cudaarray_destroy(d2s_nvenc_cudaarray_handle handle) {
    delete static_cast<NvencCudaArrayEncoder*>(handle);
}

const char* d2s_nvenc_cudaarray_last_error(void) {
    return g_last_error.c_str();
}

}  // extern "C"
