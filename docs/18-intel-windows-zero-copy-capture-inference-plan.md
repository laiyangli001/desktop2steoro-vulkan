# Intel Windows 零拷贝捕捉—推理支持计划书

## 1. 目标

为 Desktop2Stereo 增加 Windows Intel GPU 的原生图像路径：

```text
DXGI Desktop Duplication
    -> ID3D11Texture2D
    -> OpenVINO GPU RemoteTensor
    -> Intel GPU 推理
    -> GPU SBS/颜色转换
    -> oneVPL/QSV Surface
    -> H.264/H.265
    -> MediaMTX/WebRTC
```

本计划同时新增 GUI 的 `DesktopDuplication` 捕捉模式，保留现有 WindowsCaptureCUDA、WindowsCaptureROCm、WindowsCapture 和 DXCamera。

Desktop Duplication 是 Windows DXGI 1.2 的通用桌面捕捉接口，适用于 WDDM 体系下的 Intel、NVIDIA 和 AMD 显卡；但严格零拷贝仍取决于驱动、GPU 适配器、资源句柄和同步能力。

## 2. 当前实现结论

- `windows_capture_event.py` 使用 `windows_capture` 回调接口，并不能证明已经实现 DXGI Desktop Duplication。
- 已新增 `windows_desktop_duplication.py` 与 `native/desktop_duplication` 原生桥；原生 DLL 不可用时仍回退到 `DXCamera`。
- Desktop Duplication 原生路径现在优先让同一借用 D3D11 texture 同时进入 OpenVINO 推理，并在保持 `AcquireNextFrame` 生命周期期间通过 staging readback 生成兼容 BGR CPU 帧；不再默认“DXCamera 捕捉一次、原生推理再捕捉一次”。该兼容边界仍是 `gpu_to_cpu=true`、`zero_copy=false`、`gpu_copy_count=1`。
- 已建立 D3D11 Texture 到 OpenVINO RemoteTensor 的原生桥；PyTorch XPU 仍作为兼容后端。
- Vulkan 已有 Adapter 身份、外部内存、同步和 GPU 图像路径，Intel 实现应复用这些公共能力。
- “无 CPU 回读”和“严格无 GPU 复制”必须分开记录。

## 3. 技术路线

### 3.1 捕捉

新增 `WindowsDesktopDuplicationRunner`：

- 枚举 DXGI Adapter/Output。
- 使用目标 Adapter 创建 D3D11 Device。
- 通过 `DuplicateOutput` 获取 `IDXGIOutputDuplication`。
- 使用 `AcquireNextFrame` 获取 `ID3D11Texture2D`。
- 处理鼠标指针、旋转、多显示器和 dirty/move rect。
- 使用 `ReleaseFrame` 完成生命周期管理。
- 处理 `DXGI_ERROR_ACCESS_LOST`、桌面切换和分辨率变化。
- 使用固定大小的 Texture Ring Buffer。
- 严格零拷贝目标路径禁止 Map/Readback 到 CPU；当前为保证既有 CPU 兼容消费者，同帧实现保留受控 staging readback，并明确记录其非零拷贝状态。

统一帧契约至少包含：

```text
resource_kind = d3d11_texture
format        = BGRA8
adapter_luid  = ...
gpu_to_cpu    = false
capture_backend = desktop_duplication
```

### 3.2 Intel 推理

OpenVINO GPU RemoteTensor 作为正式零拷贝主路线：

```text
D3D11 BGRA8 Texture
    -> GPU BGRA8 -> NV12 conversion
    -> D3D11 NV12 surface
    -> OpenVINO RemoteContext
    -> NV12 RemoteTensor
    -> GPU preprocess
    -> GPU inference
```

需要新增 C++ 原生桥，负责：

- D3D11 Device/Texture 句柄管理。
- 使用 D3D11 VideoProcessor 或等价 GPU kernel 将 Desktop Duplication 的 BGRA8 转为 NV12。
- OpenVINO GPU RemoteContext 和 NV12 surface RemoteTensor 包装。
- GPU resize、normalize、layout 转换。
- 推理输入输出生命周期。
- Fence/同步和异常恢复。

注意：OpenVINO 的 D3D11 surface 互操作不能直接把 Desktop Duplication 的 BGRA8 texture 当作已验证的 NV12 RemoteTensor。转换器未完成前，能力探针必须保持 `zero_copy_ready=false`。

当前实现接入方式：设置 `D2S_INTEL_NATIVE_OPENVINO=1`，并提供 `D2S_OPENVINO_MODEL` 模型路径；只有 Desktop Duplication 原生桥能力探测通过时才启用原生 provider，否则继续使用 DXCamera 兼容 RGB 帧和既有深度路径。原生 provider 的深度结果会随 `CapturedFrame.metadata.native_depth_profile` 进入运行时，现有立体合成仍使用兼容 RGB 帧。由于当前 provider 的输出 ABI 会回读 CPU，运行时必须记录 `native_depth_zero_copy=0`。

现有 PyTorch XPU 保留为兼容后端，但如果发生 CPU 映射或普通 Tensor 上传，必须标记：

```text
zero_copy = false
```

### 3.3 编码

Intel 编码优先使用 oneVPL/QSV：

- 输入优先 NV12，后续支持 P010。
- 使用 D3D11/oneVPL Surface。
- 复用现有 GPU 颜色转换、帧池和同步逻辑。
- 输出只返回压缩视频包。
- 如果无法直接导入 D3D11 Surface，可允许 GPU 内复制，但必须记录 `gpu_copy_count`，不能误报严格零拷贝。
- 已接入 Intel QSV/D3D11 最终 SBS 后端，GUI 可显式选择 `Intel QSV (D3D11)`；它创建 D3D11 子设备和派生 QSV 设备，并使用 `hwupload=extra_hw_frames=16,hwmap=derive_device=qsv,format=qsv` 让 QSV 消费硬件帧。由于当前最终 SBS 仍来自 RGB24 stdin，该路径明确记录 `gpu_to_cpu=True zero_copy=False gpu_copy_count=1`。通用 FFmpeg 后端仍可通过 `D2S_QSV_D3D11_UPLOAD=1` 启用同一边界；它不冒充捕捉到编码的严格零拷贝。
- 新增可选 `native/onevpl_d3d11_encoder` C ABI：在配置 oneVPL SDK 时接收借用的 NV12 `ID3D11Texture2D`，执行 D3D11 加速编码并返回压缩包；未配置 SDK、DLL 或 Intel 驱动时探针返回不可用，不能误启用。
- 新增 `native/d3d11_sbs_surface` 最终 SBS surface bridge：接收已经完成左右眼合成的 BGRA8 CPU 帧，上传到 D3D11 staging/default texture，经 VideoProcessor 在 GPU 上转换为 NV12，并向同一 D3D11 device 的 oneVPL encoder 暴露借用 surface。该过渡路径的 CPU→GPU 上传计为 `gpu_copy_count=1`，不能报告严格零拷贝。

### 3.4 最终 SBS 输出边界

当前运行时的 OpenVINO D3D11 provider 只产生单目深度结果，不能直接作为网络视频源；网络编码必须消费已经完成左右眼合成的最终 SBS 帧。现阶段 Intel 输出契约为：

```text
stereo runtime SBS RGB8 (CPU)
    -> FFmpeg rawvideo stdin
    -> D3D11 NV12 upload
    -> QSV surface
    -> H.264/H.265 packet
```

因此这一步是 Intel GPU 编码加速和 GPU surface 复用的第一阶段，不是端到端零拷贝。后续只有当立体合成器导出最终 SBS 的 D3D11/Vulkan surface，并完成 Adapter LUID、同步和生命周期验证后，才能消除当前 RGB→D3D11 上传并切换到真正的 oneVPL native surface ABI。当前已实现可选 `D2S_ONEVPL_FINAL_SBS=1` 链路：最终 SBS → D3D11 NV12 surface → oneVPL → FFmpeg packet mux；若任一 DLL/SDK/驱动能力不可用，自动回退到 Intel QSV/D3D11 FFmpeg 路径。

### 3.5 Vulkan 公共层

复用 Vulkan 已有能力：

- Adapter LUID/UUID 匹配。
- 外部内存句柄。
- Fence/semaphore。
- 图像格式和编码能力探测。
- GPU copy 计数。
- 资源 lease 和环形缓冲区。

D3D11 到 Vulkan 若需要 GPU copy，则记录：

```text
gpu_to_cpu = false
zero_copy = false
gpu_copy_count = 1
```

## 4. GUI 和回退

GUI 新增独立选项：

```text
Desktop Duplication
```

内部键：

```text
DesktopDuplication
```

保留：

```text
WindowsCaptureCUDA
WindowsCaptureROCm
WindowsCapture
DXCamera
```

自动模式推荐顺序：

```text
Intel       -> DesktopDuplication
NVIDIA      -> WindowsCaptureCUDA
AMD         -> WindowsCaptureROCm
未知/失败   -> DesktopDuplication -> WindowsCapture -> DXCamera
```

用户显式选择的模式不得静默替换；初始化失败时必须输出明确原因并进入稳定回退。

## 5. 统一日志

每次启动记录：

- 捕捉后端。
- 推理后端。
- 编码后端。
- Adapter LUID/UUID。
- 资源格式和分辨率。
- `capture_gpu`。
- `gpu_to_cpu`。
- `gpu_copy_count`。
- `zero_copy`。
- 回退原因。

只有满足以下条件才允许记录 `zero_copy=True`：

- 捕捉帧没有 CPU 映射。
- 推理使用原生 RemoteTensor/共享 GPU 资源。
- 原始帧没有 `.cpu()`、`.numpy()` 或 Host staging。
- 编码输入使用同一资源或可验证共享句柄。
- 捕捉、推理和编码 Adapter 一致。
- GPU 同步和资源生命周期通过验证。

## 6. 实施阶段

### 阶段 1：Desktop Duplication 捕捉

- 原生 D3D11/DXGI 捕捉桥。
- 统一 `CapturedFrame` 元数据。
- GUI 模式和配置归一化。
- 初始化失败和 `ACCESS_LOST` 回退。
- 单元测试和资源生命周期测试。

### 阶段 2：OpenVINO RemoteTensor

- 模型转换验证。
- D3D11 Texture 包装为 RemoteTensor。
- GPU 预处理和推理。
- 原生桥 Python API。
- PyTorch XPU 兼容回退。

### 阶段 3：Intel QSV/oneVPL

- D3D11/oneVPL Surface。
- GPU NV12 转换。
- 硬件编码和压缩包输出。
- MediaMTX/WebRTC 共享发布。

### 阶段 4：Vulkan 联动

- 公共 Adapter/Handle/Fence 层。
- D3D11/Vulkan 资源互操作。
- GPU copy 和 zero-copy 可观测性。
- 多 GPU 和驱动差异处理。

## 7. 验收

### 自动化测试

- GUI 能显示并保存 `DesktopDuplication`。
- 非 Windows 环境不会加载 DXGI 模块。
- `DXGI_ERROR_ACCESS_LOST` 能触发重建。
- Adapter 不匹配时拒绝零拷贝导入。
- 旧配置继续加载。
- NVIDIA、AMD、CPU 路径不回归。

### 真机测试

- Intel UHD/Iris Xe。
- Intel Arc。
- Intel 核显与独显共存。
- 多显示器、4K 30 FPS、4K 60 FPS。
- 窗口移动、最小化、桌面切换、分辨率变化。
- 受保护视频内容。
- 连续运行至少 30 分钟。

## 8. 当前实现状态

- [x] 新增 `DesktopDuplication` GUI/工厂模式。
- [x] 新增 `native/desktop_duplication` 的 DXGI/D3D11 C ABI 与 CMake 工程。
- [x] 新增 Python ctypes 能力探测和原生会话封装。
- [x] `DXGI_ERROR_ACCESS_LOST` 时重建 Desktop Duplication 输出并由 Python 自动重试一次。
- [x] 新增 `NativeD3D11TextureFrame` 借用帧对象及 `CapturedFrame` GPU 资源契约。
- [x] 新增 OpenVINO RemoteTensor 能力层，明确区分 Python runtime 与原生 D3D11 bridge。
- [x] 新增 `native/openvino_d3d11_bridge` 的 C ABI/CMake 接口，以及 Python ctypes session facade。
- [x] 校正 RemoteTensor 能力边界：桥接器明确暴露 NV12 surface 与 BGRA8→NV12 conversion capability，未具备二者时不报告零拷贝。
- [x] 原生桥不可用时明确回退到 DXCamera，并保持 `gpu_to_cpu=True zero_copy=False`。
- [x] Desktop Duplication 探针合并捕捉能力、OpenVINO RemoteTensor 能力、`zero_copy_ready` 和回退原因日志。
- [x] 实现 D3D11 VideoProcessor 的 BGRA8→NV12 GPU 转换器、NV12 RemoteTensor 接线，以及输出 shape/float buffer ABI。
- [x] 暴露 bridge 内部借用的 NV12 D3D11 surface 契约，供后续 oneVPL/QSV 复用；直接编码提交仍需 native encoder 实现和实机验证。
- [x] 新增 `OpenVINOD3D11DepthProvider` 适配器，将原生输出转换为现有 `DepthProfileResult`，并在桥接能力不完整时拒绝初始化。
- [x] Desktop Duplication C ABI 暴露同一捕捉适配器的 `ID3D11Device`，provider 创建默认复用该 device，并通过 DXGI Adapter LUID 拒绝跨适配器推理。
- [x] 新增 `infer_native_frame()` 生命周期封装，provider 异常时也会释放借用的 Desktop Duplication frame。
- [x] 新增同帧 `copy_frame` readback C ABI；原生 Desktop Duplication monitor 路径在同一借用帧内完成推理和兼容 BGR 输出，GitHub workflow 校验该导出符号。
- [x] 通过 GitHub-hosted Windows runner 使用官方 OpenVINO Windows C++ archive 构建并验证原生 bridge DLL；输出 surface ABI 已完成导出校验，真实设备运行仍待验证。
- [x] 将原生 `ID3D11Texture2D` 接入 OpenVINO RemoteTensor 的代码链路；仍需在真实 DLL/Intel 驱动环境验证。
- [x] 接入可选 FFmpeg D3D11/QSV Surface upload 边界；仍需将 OpenVINO/Vulkan 原生输出在真实环境交给 oneVPL/QSV。
- [x] 新增可选 `native/onevpl_d3d11_encoder` bridge 和 Python 能力探测/Surface 提交 API；已由 GitHub Actions 使用官方 oneVPL dispatcher 远程编译并完成导出/链接校验，Intel 真机仍待验证。
- [x] 新增 `native/d3d11_sbs_surface` 最终 SBS BGRA8→NV12 bridge、Python surface owner 和 oneVPL final-SBS 输出接线；已由 GitHub Actions 远程编译，真实 oneVPL 硬件编码仍待 Intel 真机验证。
- [ ] 将最终 SBS 原生 D3D11/Vulkan surface 接入 oneVPL/QSV，消除当前 RGB24 stdin 边界。
- [ ] 在 Intel 真机完成 4K 长时间验证，并确认 Desktop Duplication、OpenVINO 与编码设备的 Adapter LUID 一致。

目标机诊断命令：

```powershell
src\python3\python.exe src\tools\intel_native_runtime_probe.py
src\python3\python.exe src\tools\intel_native_runtime_probe.py --strict
```

默认模式用于收集 JSON 能力报告；`--strict` 只有在 Windows Intel 驱动、OpenVINO runtime、oneVPL 和四个发布 DLL 都可用时才返回成功。

新增 `.github/workflows/intel-windows-native.yml`，以 GitHub-hosted `windows-2022` runner 作为 C++ 编译依据：workflow 从官方 `oneapi-src/oneVPL`、OpenVINO Windows C++ archive、Khronos OpenCL-Headers/CLHPP/ICD-Loader 拉取依赖，编译 Desktop Duplication、D3D11 SBS surface、oneVPL 和 OpenVINO bridge，并收集必要 OpenVINO runtime DLL，上传扁平化 artifact、`manifest.json` 和仅覆盖二进制文件的 SHA-256 清单；后续 job 会按功能自动提交到 `src/desktop2stereo/capture/native/desktop_duplication/`、`src/desktop2stereo/stereo_runtime/providers/intel/native/d3d11_sbs_surface/`、`onevpl_d3d11_encoder/` 和 `openvino_d3d11_bridge/`，使对外发布版无需手工设置 DLL 路径。workflow run `32644145650` 已成功完成全部配置、编译、DLL C ABI 导出及 oneVPL `libvpl.dll` 链接校验；本地不作为 native C++ 编译验证环境。下载 artifact 后，将目录设置到 `D2S_INTEL_NATIVE_ARTIFACT_DIR`，四个 Python 适配器会共享该目录；单个 `D2S_*_DLL` 环境变量仍可覆盖。OpenVINO runtime DLL 随发布包提供，Intel 驱动和 OpenCL ICD 仍需安装在目标机。Python 侧只有 DLL 可加载且导出 ABI 完整时才会报告 `directx_remote_tensor=True`。本轮新增 Intel surface/oneVPL/QSV 及同帧 readback 回归测试；全量 Python 测试此前为 `1202 passed`。当前剩余工作是下载 artifact 到 Intel 目标机完成驱动、OpenVINO GPU RemoteTensor、oneVPL 硬件编码、Adapter LUID、4K 长时间和最终 SBS 原生 surface 验证。仓库中未发现 `docs/00-api-handoff-progress.md`，因此未修改不存在的交接文档。

## 9. 交付文件

计划执行过程中更新：

- `docs/18-intel-windows-zero-copy-capture-inference-plan.md`
- `docs/16-network-streaming-specification.md`
- `native/d3d11_sbs_surface/`
- `native/onevpl_d3d11_encoder/`
- `.github/workflows/intel-windows-native.yml`
- `changelog.md`

参考：

- [Microsoft Desktop Duplication API](https://learn.microsoft.com/en-us/windows/win32/direct3ddxgi/desktop-dup-api)
- [OpenVINO GPU RemoteTensor API](https://docs.openvino.ai/nightly/openvino-workflow/running-inference/inference-devices-and-modes/gpu-device/remote-tensor-api-gpu-plugin.html)
- [Intel oneVPL](https://www.intel.com/content/www/us/en/developer/tools/vpl/overview.html)
