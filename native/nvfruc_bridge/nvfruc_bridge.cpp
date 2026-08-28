#include "nvfruc_bridge.h"

#include <NvOFFRUC.h>
#include <cuda_runtime.h>
#include <windows.h>

#include <array>
#include <cstring>
#include <mutex>
#include <string>

namespace {

thread_local std::string g_last_error;
std::mutex g_loader_mutex;
HMODULE g_library = nullptr;

using CreateFn = NvOFFRUC_STATUS(CALLBACK*)(const NvOFFRUC_CREATE_PARAM*, NvOFFRUCHandle*);
using RegisterFn = NvOFFRUC_STATUS(CALLBACK*)(NvOFFRUCHandle, const NvOFFRUC_REGISTER_RESOURCE_PARAM*);
using UnregisterFn = NvOFFRUC_STATUS(CALLBACK*)(NvOFFRUCHandle, const NvOFFRUC_UNREGISTER_RESOURCE_PARAM*);
using ProcessFn = NvOFFRUC_STATUS(CALLBACK*)(NvOFFRUCHandle, const NvOFFRUC_PROCESS_IN_PARAMS*, const NvOFFRUC_PROCESS_OUT_PARAMS*);
using DestroyFn = NvOFFRUC_STATUS(CALLBACK*)(NvOFFRUCHandle);

CreateFn g_create = nullptr;
RegisterFn g_register = nullptr;
UnregisterFn g_unregister = nullptr;
ProcessFn g_process = nullptr;
DestroyFn g_destroy = nullptr;

struct Session {
    uint32_t width = 0;
    uint32_t height = 0;
    int device_index = 0;
    cudaStream_t stream = nullptr;
    std::array<cudaArray_t, 3> arrays{};
    NvOFFRUCHandle fruc = nullptr;
};

int fail(const std::string& message) {
    g_last_error = message;
    return -1;
}

int fail_cuda(cudaError_t status, const char* operation) {
    return fail(std::string(operation) + ": " + cudaGetErrorString(status));
}

bool load_library() {
    std::lock_guard<std::mutex> lock(g_loader_mutex);
    if (g_library && g_create && g_register && g_process && g_destroy) {
        return true;
    }
    const wchar_t* names[] = {L"NvOFFRUC.dll", L"NvFRUC.dll"};
    for (const auto* name : names) {
        g_library = LoadLibraryW(name);
        if (g_library) {
            break;
        }
    }
    if (!g_library) {
        fail("NvOFFRUC.dll/NvFRUC.dll was not found in the DLL search path");
        return false;
    }
    g_create = reinterpret_cast<CreateFn>(GetProcAddress(g_library, "NvOFFRUCCreate"));
    g_register = reinterpret_cast<RegisterFn>(GetProcAddress(g_library, "NvOFFRUCRegisterResource"));
    g_unregister = reinterpret_cast<UnregisterFn>(GetProcAddress(g_library, "NvOFFRUCUnregisterResource"));
    g_process = reinterpret_cast<ProcessFn>(GetProcAddress(g_library, "NvOFFRUCProcess"));
    g_destroy = reinterpret_cast<DestroyFn>(GetProcAddress(g_library, "NvOFFRUCDestroy"));
    if (!g_create || !g_register || !g_process || !g_destroy) {
        fail("NvOFFRUC runtime is missing one or more required exports");
        return false;
    }
    return true;
}

void release_arrays(Session* session) {
    for (auto& array : session->arrays) {
        if (array) {
            cudaFreeArray(array);
            array = nullptr;
        }
    }
}

void release_session(Session* session) {
    if (!session) {
        return;
    }
    if (session->fruc && g_destroy) {
        g_destroy(session->fruc);
        session->fruc = nullptr;
    }
    release_arrays(session);
    delete session;
}

void reset_session(Session* session) {
    if (session->fruc && g_destroy) {
        g_destroy(session->fruc);
        session->fruc = nullptr;
    }
    release_arrays(session);
}

int initialize_session(Session* session) {
    if (!load_library()) {
        return -1;
    }
    if (cudaSetDevice(session->device_index) != cudaSuccess) {
        return fail_cuda(cudaGetLastError(), "cudaSetDevice");
    }
    cudaChannelFormatDesc format = cudaCreateChannelDesc<uchar4>();
    const cudaExtent extent = make_cudaExtent(session->width, session->height, 1);
    for (auto& array : session->arrays) {
        const cudaError_t status = cudaMalloc3DArray(&array, &format, extent);
        if (status != cudaSuccess) {
            release_arrays(session);
            return fail_cuda(status, "cudaMalloc3DArray");
        }
    }

    NvOFFRUC_CREATE_PARAM create_params{};
    create_params.uiWidth = session->width;
    create_params.uiHeight = session->height;
    create_params.pDevice = nullptr;
    create_params.eResourceType = CudaResource;
    create_params.eSurfaceFormat = ARGBSurface;
    create_params.eCUDAResourceType = CudaResourceCuArray;
    NvOFFRUC_STATUS status = g_create(&create_params, &session->fruc);
    if (status != NvOFFRUC_SUCCESS || !session->fruc) {
        release_arrays(session);
        return fail("NvOFFRUCCreate failed with status " + std::to_string(static_cast<int>(status)));
    }

    NvOFFRUC_REGISTER_RESOURCE_PARAM register_params{};
    register_params.uiCount = 3;
    for (size_t i = 0; i < session->arrays.size(); ++i) {
        register_params.pArrResource[i] = session->arrays[i];
    }
    status = g_register(session->fruc, &register_params);
    if (status != NvOFFRUC_SUCCESS) {
        g_destroy(session->fruc);
        session->fruc = nullptr;
        release_arrays(session);
        return fail("NvOFFRUCRegisterResource failed with status " +
                    std::to_string(static_cast<int>(status)));
    }
    return 0;
}

int copy_to_array(
    cudaArray_t destination,
    uint64_t source,
    size_t source_pitch,
    cudaStream_t stream,
    uint32_t width,
    uint32_t height
) {
    const cudaError_t status = cudaMemcpy2DToArrayAsync(
        destination,
        0,
        0,
        reinterpret_cast<const void*>(source),
        source_pitch,
        static_cast<size_t>(width) * 4,
        height,
        cudaMemcpyDeviceToDevice,
        stream
    );
    return status == cudaSuccess ? 0 : fail_cuda(status, "cudaMemcpy2DToArrayAsync");
}

int copy_from_array(
    uint64_t destination,
    size_t destination_pitch,
    cudaArray_t source,
    cudaStream_t stream,
    uint32_t width,
    uint32_t height
) {
    const cudaError_t status = cudaMemcpy2DFromArrayAsync(
        reinterpret_cast<void*>(destination),
        destination_pitch,
        source,
        0,
        0,
        static_cast<size_t>(width) * 4,
        height,
        cudaMemcpyDeviceToDevice,
        stream
    );
    return status == cudaSuccess ? 0 : fail_cuda(status, "cudaMemcpy2DFromArrayAsync");
}

}  // namespace

extern "C" {

uint32_t d2s_nvfruc_abi_version(void) {
    return 1;
}

int32_t d2s_nvfruc_probe(void) {
    if (!load_library()) {
        return -1;
    }
    int device = 0;
    const cudaError_t status = cudaGetDevice(&device);
    if (status != cudaSuccess) {
        return fail_cuda(status, "cudaGetDevice");
    }
    return 0;
}

const char* d2s_nvfruc_last_error(void) {
    return g_last_error.c_str();
}

d2s_nvfruc_handle d2s_nvfruc_create(
    uint32_t width,
    uint32_t height,
    int32_t device_index,
    uint64_t cuda_stream
) {
    if (!width || !height || device_index < 0) {
        fail("NvFRUC dimensions and device index must be positive");
        return nullptr;
    }
    auto* session = new Session();
    session->width = width;
    session->height = height;
    session->device_index = device_index;
    session->stream = reinterpret_cast<cudaStream_t>(cuda_stream);
    if (initialize_session(session) != 0) {
        delete session;
        return nullptr;
    }
    return session;
}

int32_t d2s_nvfruc_process(
    d2s_nvfruc_handle handle,
    uint64_t previous_ptr,
    size_t previous_pitch,
    double previous_timestamp,
    uint64_t next_ptr,
    size_t next_pitch,
    double next_timestamp,
    uint64_t output_ptr,
    size_t output_pitch,
    double output_timestamp,
    uint64_t cuda_stream
) {
    auto* session = static_cast<Session*>(handle);
    if (!session || !previous_ptr || !next_ptr || !output_ptr) {
        return fail("NvFRUC process received an invalid session or CUDA pointer");
    }
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);
    if (cudaSetDevice(session->device_index) != cudaSuccess) {
        return fail_cuda(cudaGetLastError(), "cudaSetDevice");
    }
    if (copy_to_array(session->arrays[0], previous_ptr, previous_pitch, stream, session->width, session->height) ||
        copy_to_array(session->arrays[1], next_ptr, next_pitch, stream, session->width, session->height)) {
        return -1;
    }

    uint64_t repeated = 0;
    NvOFFRUC_PROCESS_IN_PARAMS input{};
    NvOFFRUC_PROCESS_OUT_PARAMS output{};
    input.stFrameDataInput.pFrame = session->arrays[0];
    input.stFrameDataInput.nTimeStamp = previous_timestamp;
    input.bSkipWarp = 1;
    output.stFrameDataOutput.pFrame = session->arrays[2];
    output.stFrameDataOutput.nTimeStamp = previous_timestamp;
    output.stFrameDataOutput.bHasFrameRepetitionOccurred = &repeated;
    NvOFFRUC_STATUS status = g_process(session->fruc, &input, &output);
    if (status != NvOFFRUC_SUCCESS) {
        return fail("NvOFFRUCProcess state update failed with status " +
                    std::to_string(static_cast<int>(status)));
    }

    input = {};
    output = {};
    input.stFrameDataInput.pFrame = session->arrays[1];
    input.stFrameDataInput.nTimeStamp = next_timestamp;
    input.bSkipWarp = 0;
    output.stFrameDataOutput.pFrame = session->arrays[2];
    output.stFrameDataOutput.nTimeStamp = output_timestamp;
    output.stFrameDataOutput.bHasFrameRepetitionOccurred = &repeated;
    status = g_process(session->fruc, &input, &output);
    if (status != NvOFFRUC_SUCCESS) {
        return fail("NvOFFRUCProcess interpolation failed with status " +
                    std::to_string(static_cast<int>(status)));
    }
    return copy_from_array(
        output_ptr,
        output_pitch,
        session->arrays[2],
        stream,
        session->width,
        session->height
    );
}

int32_t d2s_nvfruc_reset(d2s_nvfruc_handle handle) {
    auto* session = static_cast<Session*>(handle);
    if (!session) {
        return fail("NvFRUC reset received an invalid session");
    }
    reset_session(session);
    return initialize_session(session);
}

void d2s_nvfruc_destroy(d2s_nvfruc_handle handle) {
    release_session(static_cast<Session*>(handle));
}

}  // extern "C"
