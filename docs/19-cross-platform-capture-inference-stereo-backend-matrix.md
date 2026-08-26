# 跨平台捕获、深度推理与立体合成后端组合矩阵

## 1. 文档目的与适用范围

本文基于当前项目代码、[macOS ScreenCaptureKit 到 MPS/CoreML 零拷贝实现调查报告](17-macos-zero-copy-capture-inference-survey.md) 和 [Intel Windows 零拷贝捕捉—推理支持计划书](18-intel-windows-zero-copy-capture-inference-plan.md)，统一定义不同操作系统、GPU 厂商下的捕获、深度推理和立体合成组合。

本文既描述当前实现，也定义目标架构。表中的“目标”或“规划”不代表已经交付；没有当前代码或真机结果支撑的能力统一标记为“待验证”。

状态含义：

- **已实现**：已有可执行代码路径；仍应在目标驱动和发布环境回归。
- **部分实现**：已有独立组件或桥接基础，但捕获、推理、合成尚未形成完整闭环。
- **规划**：设计目标，当前没有完整实现。
- **兼容回退**：以可用性优先，允许 CPU 回读、CPU 上传或较低性能。
- **待验证**：代码存在或技术上可行，但缺少对应硬件、驱动或端到端运行证据。

## 2. 三个独立配置维度

三个维度必须分别配置，不能相互混用：

1. **捕获模式**：仅表示捕获对象，取值为“显示器”或“窗口”。
2. **捕获工具/API**：负责取得桌面或窗口图像，例如 DesktopDuplication、WindowsCapture、WindowsCaptureCUDA、WindowsCaptureROCm、DXCamera、ScreenCaptureKit、CoreGraphics 和 Linux MSS。
3. **推理与立体合成后端**：负责深度推理和左右眼图像生成，例如 TensorRT、MIGraphX、OpenVINO、DirectML、CoreML、MPSGraph、Vulkan、Metal、OpenGL 和 CPU。

DirectML、OpenVINO、CoreML、MPSGraph、Vulkan 和 Metal 都不是捕获模式。

Windows 捕获工具边界：

- **DesktopDuplication**：Windows 跨厂商 DXGI 1.2 / D3D11 显示器捕获路径；本身不原生支持单窗口捕获。当前项目在用户要求窗口捕获时会进入兼容路径，不能把该行为记作 Desktop Duplication 的原生窗口能力。
- **WindowsCapture**：基于 Windows Graphics Capture 的通用路径，支持显示器和窗口；当前通用实现会把帧复制到 CPU。
- **WindowsCaptureCUDA**：支持显示器和窗口，捕获帧可进入 CUDA GPU Tensor 路径。
- **WindowsCaptureROCm**：支持显示器和窗口，捕获帧可进入 ROCm/HIP GPU Tensor 路径；依赖和代码入口已存在，AMD 真机兼容性仍需持续验证。
- **DXCamera**：基于项目使用的 Windows 相机/桌面捕获封装，支持显示器和按窗口区域捕获，当前输出为 CPU NumPy 图像，主要作为兼容回退。

## 3. 跨平台后端组合主矩阵

> “预计 GPU 复制次数”只统计捕获资源进入推理并交给立体合成这一段。CPU 映射、CPU 图像转换和最终编码另行记录。区间表示是否可共享资源取决于驱动、格式、适配器一致性和同步机制。

| 操作系统 | GPU/硬件类型 | 捕获模式 | 首选捕获工具/API | 捕获输出资源 | 首选推理后端 | 推理备用后端 | 首选立体合成后端 | 合成备用后端 | 是否支持零 CPU 回读 | 预计 GPU 复制次数 | 当前实现状态 | 自动选择及回退顺序 | 备注与限制 |
|---|---|---|---|---|---|---|---|---|---|---:|---|---|---|
| Windows | NVIDIA | 显示器、窗口 | WindowsCaptureCUDA；显示器也可用 DesktopDuplication | 首选 CUDA GPU Tensor；DesktopDuplication 原生资源为 D3D11 Texture，当前兼容输出仍会 staging readback | 原生 TensorRT | PyTorch CUDA → DirectML → CPU | 目标为 Vulkan | OpenGL → Torch/CPU | 捕获到推理可避免 CPU 回读；推理到 Vulkan 的完整共享链路需按输出模式验证 | 0–1 | 厂商捕获和 NVIDIA 推理已实现；完整跨 API 零回读为部分实现/待验证 | 捕获：WindowsCaptureCUDA → DesktopDuplication → WindowsCapture → DXCamera；推理：原生 TensorRT → PyTorch CUDA → DirectML → CPU；合成：Vulkan → OpenGL → Torch/CPU | OpenGL 当前仅有输出/流媒体兼容实现，真正立体合成能力仍待补齐；完整跨 API 零回读待验证 |
| Windows | AMD | 显示器、窗口 | WindowsCaptureROCm；显示器也可用 DesktopDuplication | 首选 ROCm/HIP GPU Tensor；DesktopDuplication 为 D3D11 Texture，兼容路径可回读 CPU | MIGraphX/ROCm | PyTorch ROCm → DirectML → CPU | 目标为 Vulkan | OpenGL → Torch/CPU | 厂商捕获到 ROCm 推理具备无 CPU 回读代码基础；端到端仍待 AMD 真机验证 | 0–1 | ROCm 捕获依赖、MIGraphX 和 PyTorch ROCm provider 已实现；硬件闭环待验证 | 捕获：WindowsCaptureROCm → DesktopDuplication → WindowsCapture → DXCamera；推理：MIGraphX → PyTorch ROCm → DirectML → CPU；合成目标：Vulkan → OpenGL → CPU | MIGraphX 当前主要面向 Distill-Any-Depth；模型、驱动和 `wc-rocm` 兼容性必须真机验证 |
| Windows | Intel | 显示器；窗口使用 WindowsCapture | 显示器首选 DesktopDuplication；窗口首选 WindowsCapture | D3D11 BGRA Texture；当前同帧兼容输出仍保留 staging readback | OpenVINO GPU RemoteTensor / D3D11 原生桥 | PyTorch XPU → DirectML → CPU | Vulkan；Intel 专用最终输出可使用 D3D11/oneVPL 路径 | OpenGL → Torch/CPU | 原生输入侧具备零 CPU 回读基础；当前深度输出和部分兼容消费者仍会回读，不能宣称端到端零回读 | 目标 0–1；当前兼容路径至少 1 | D3D11/OpenVINO/oneVPL/Vulkan 互操作组件已实现并由 CI 编译；完整 Intel 真机长跑待验证 | 捕获：DesktopDuplication（显示器）或 WindowsCapture（窗口）→ DXCamera；推理：OpenVINO → XPU → DirectML → CPU；合成：Vulkan或专用 D3D11 → OpenGL → CPU | 必须校验 Desktop Duplication、OpenVINO、Vulkan和 oneVPL 的 Adapter LUID；OpenVINO 输出 ABI 当前仍可能回读 CPU |
| Windows | 其他支持 DX12/DirectML 的 GPU，包括国产显卡 | 显示器、窗口 | WindowsCapture；显示器可选 DesktopDuplication | 当前通用 WindowsCapture 为 CPU NumPy；目标为 D3D11 Texture，经共享资源或一次 GPU 复制进入 D3D12 | DirectML | CPU | Vulkan | OpenGL → CPU | 当前不支持完整零 CPU 回读；目标支持，但需逐厂商、逐驱动验证 | 目标 0–1；当前通常 1 次 CPU→GPU 上传且伴随 CPU 回读 | DirectML 设备发现和 PyTorch DirectML 计算路径已有基础；D3D11/D3D12/Vulkan 共享闭环为规划 | 捕获：WindowsCapture → DesktopDuplication（仅显示器）→ DXCamera；推理：DirectML → CPU；合成：Vulkan → OpenGL → CPU | “支持 DX12”不等于已经通过 DirectML 模型算子、共享句柄和 Vulkan 外部内存验证；国产显卡均标记待验证 |
| macOS | Apple Silicon | 显示器、窗口 | ScreenCaptureKit | 当前：CVPixelBuffer 锁定后复制为 CPU NumPy；目标：CVPixelBuffer/IOSurface → MTLTexture | 目标：CoreML；可选 MPSGraph | 当前 PyTorch MPS → CPU | 目标：Metal | CPU | 当前不支持；目标路径支持零 CPU 图像回读 | 目标 0–1；当前至少 1 次 CPU→GPU 上传 | ScreenCaptureKit 和 PyTorch MPS 已实现；CoreML/MPSGraph/Metal 原生桥为规划 | 捕获：ScreenCaptureKit → CoreGraphics；推理目标：CoreML → MPSGraph → PyTorch MPS → CPU；合成：Metal → CPU | 不将 Vulkan/MoltenVK或已废弃的 OpenGL作为 macOS 零拷贝主路径；CoreML 模型转换和 ANE/GPU 调度待验证 |
| macOS | Intel Mac | 显示器、窗口 | ScreenCaptureKit；旧系统回退 CoreGraphics | CPU BGRA/NumPy；IOSurface/Metal 目标能力依机型和系统版本验证 | CoreML（待验证） | PyTorch MPS（若系统和 GPU 支持）→ CPU | Metal（待验证） | CPU | 当前不支持；目标能力待具体 Intel Mac/AMD GPU 组合验证 | 目标 0–1；当前至少 1 次 CPU→GPU 上传 | 捕获已实现；厂商推理与 Metal 闭环未验证，CPU 是可靠回退 | 捕获：ScreenCaptureKit → CoreGraphics；推理：CoreML/MPS → CPU；合成：Metal → CPU | Intel Mac 不应套用 Apple Silicon 的 ANE 和统一内存假设；不同机型差异较大 |
| Linux | NVIDIA | 显示器、窗口区域 | 当前实际实现：MSS + Xlib | CPU BGRA NumPy | 原生 TensorRT | PyTorch CUDA → CPU | Vulkan | OpenGL → Torch/CPU | 当前不支持，因为捕获阶段已回读 CPU | 至少 1 次 CPU→GPU 上传 | MSS 捕获和 NVIDIA 推理/Vulkan 组件已实现；Linux GPU 原生捕获未实现 | 捕获：MSS；推理：原生 TensorRT → PyTorch CUDA → CPU；合成目标：Vulkan → OpenGL → CPU | DirectML仅Windows可用，Linux自动跳过；窗口路径按X11坐标截取，不是窗口surface零拷贝；Wayland/PipeWire/DMABUF尚无项目实现 |
| Linux | AMD | 显示器、窗口区域 | 当前实际实现：MSS + Xlib | CPU BGRA NumPy | MIGraphX/ROCm | PyTorch ROCm → CPU | Vulkan | OpenGL → Torch/CPU | 当前不支持 | 至少 1 次 CPU→GPU 上传 | MSS、MIGraphX/ROCm 和 Vulkan 代码存在；AMD Linux 端到端真机待验证 | 捕获：MSS；推理：MIGraphX → PyTorch ROCm → CPU；合成目标：Vulkan → OpenGL → CPU | Wayland、PipeWire、DMA-BUF 和跨设备同步均为缺口；MIGraphX 模型覆盖有限 |
| Linux | Intel或其他显卡 | 显示器、窗口区域 | 当前实际实现：MSS + Xlib | CPU BGRA NumPy | Intel 首选 OpenVINO/PyTorch XPU；其他 GPU 的厂商后端待验证 | 通用 PyTorch设备（可用时）→ CPU | Vulkan | OpenGL → Torch/CPU | 当前不支持 | 至少 1 次 CPU→GPU 上传 | MSS 已实现；Intel XPU provider 有代码基础；Linux OpenVINO、其他厂商 GPU 和完整链路待验证 | 捕获：MSS；推理：OpenVINO/XPU（Intel）或可用厂商后端 → CPU；合成目标：Vulkan → OpenGL → CPU | 当前没有 Linux 原生 GPU 捕获工具选项；不能仅凭 Vulkan 可用推断深度模型可在该 GPU 上运行 |
| Windows / macOS / Linux | GPU 后端不可用或初始化失败 | 显示器、窗口（按平台捕获能力） | 平台通用 CPU 捕获：WindowsCapture/DXCamera、CoreGraphics、MSS | CPU BGRA/BGR NumPy | CPU PyTorch | 无 | Torch/CPU | 平台显示/输出的最低兼容路径 | 不适用；该路径明确允许 CPU 内存 | 0 次 GPU复制；若仍使用 GPU显示则另有 1 次上传 | 已实现的兼容回退 | 捕获工具失败时按平台回退 → CPU推理 → CPU合成/输出 | 应输出明确的回退原因；CPU 路径以兼容性为目标，不承诺实时 4K 性能 |

## 4. Windows 通用 DX12/DirectML 目标路径

Windows 其他 DX12/DirectML GPU 的目标路径为：

```text
Desktop Duplication / Windows Graphics Capture
→ D3D11 纹理
→ D3D11/D3D12 共享资源或一次 GPU 复制
→ DirectML 推理
→ Vulkan 立体合成
```

约束如下：

- Desktop Duplication 只负责显示器捕获；窗口捕获必须使用 Windows Graphics Capture。
- 当前 `WindowsCapture` 若未暴露原生资源仍使用 CPU 兼容帧；暴露原生资源时原生帧是主输出，DirectML 输入准备器只有在共享句柄和同 Adapter LUID 的桥接方法实际返回输入后才允许 GPU 路径。
- DirectML 是推理后端，不是捕获工具。
- D3D11 到 D3D12 应优先使用同 Adapter 共享资源；驱动或格式不支持时允许一次 GPU 内复制。
- DirectML 输出到 Vulkan 还需要可验证的共享句柄、格式转换和同步；未验证前不得记录 `zero_copy=true`。
- 若 DirectML 模型出现不支持算子、隐式 CPU fallback 或设备创建失败，必须回退 CPU，并记录具体原因。

## 5. macOS 零 CPU 回读目标路径

macOS 原生目标路径为：

```text
ScreenCaptureKit
→ CVPixelBuffer / IOSurface
→ MTLTexture
→ Metal/MPS 预处理
→ CoreML或MPSGraph推理
→ Metal立体合成
```

当前代码仍通过 `CVPixelBufferGetBaseAddress` 将画面复制到 NumPy，再上传到 PyTorch MPS；因此当前路径不是零 CPU 回读。目标实现必须保持 IOSurface、MTLTexture、模型输入和立体合成资源在同一 `MTLDevice` 上，并用 Metal command buffer 明确同步。

Apple Silicon 首选 CoreML，MPSGraph用于需要更细控制或 CoreML 算子受限的场景，PyTorch MPS保留为现有兼容路径。Intel Mac 的 CoreML、MPS和 Metal 能力必须按真实机型验证，不得假定与 Apple Silicon 等价。

## 6. 各平台“自动”模式完整选择顺序

以下是自动选择顺序；仍未闭环或仅为目标的部分列在第 10 节。

### 6.1 Windows

- **NVIDIA**
  - 捕获：WindowsCaptureCUDA → DesktopDuplication（显示器）/ WindowsCapture（窗口）→ DXCamera。
  - 推理：原生 TensorRT → PyTorch CUDA → DirectML → CPU。
  - 合成：Vulkan → OpenGL → Torch/CPU。
- **AMD**
  - 捕获：WindowsCaptureROCm → DesktopDuplication（显示器）/ WindowsCapture（窗口）→ DXCamera。
  - 推理：MIGraphX → PyTorch ROCm → DirectML → CPU。
  - 合成：Vulkan → OpenGL → Torch/CPU。
- **Intel**
  - 捕获：DesktopDuplication（显示器）或 WindowsCapture（窗口）→ DXCamera。
  - 推理：OpenVINO D3D11 RemoteTensor → PyTorch XPU → DirectML → CPU。
  - 合成：Vulkan；Intel专用输出可选 D3D11/oneVPL → OpenGL → Torch/CPU。
- **其他 DX12/DirectML GPU**
  - 捕获：WindowsCapture → DesktopDuplication（仅显示器）→ DXCamera。
  - 推理：DirectML → CPU。
  - 合成：Vulkan → OpenGL → CPU。

任何显式选择都应先尝试用户指定项；失败时必须显示失败原因和实际回退项，不能静默伪装成原后端。

### 6.2 macOS

- 捕获：ScreenCaptureKit → CoreGraphics。
- Apple Silicon推理目标：CoreML → MPSGraph → PyTorch MPS → CPU。
- Intel Mac推理：CoreML或PyTorch MPS（能力探测通过时）→ CPU。
- 合成：Metal → CPU。
- MoltenVK不进入零 CPU 回读主链；OpenGL不作为新架构主路径。

### 6.3 Linux

- 捕获：当前固定为 MSS + Xlib；未来原生 GPU 捕获后端必须单独设计和探测。
- NVIDIA推理：原生 TensorRT → PyTorch CUDA → CPU；DirectML仅Windows可用，在Linux自动跳过。
- AMD推理：MIGraphX → PyTorch ROCm → CPU。
- Intel推理：OpenVINO/PyTorch XPU → CPU。
- 其他 GPU推理：已验证的厂商后端 → CPU；不得因 Vulkan 可用而自动选择 Vulkan 推理。
- 合成：Vulkan → OpenGL → Torch/CPU。

## 7. GUI 配置与后端自动选择建议

GUI 只保留用户容易理解且确实需要主动选择的“捕获模式”和“捕获工具”。深度推理与立体合成后端不提供下拉框，由程序按照第 6 节定义的平台顺序自动探测、依次尝试和回退。

### 7.1 捕获模式控件

只提供两个与平台无关的选项：

- 显示器
- 窗口

捕获模式不能包含“自动”或任何捕获工具、推理后端、合成后端名称。

### 7.2 捕获工具下拉框

按当前操作系统过滤不可用项：

- 自动
- Windows：DesktopDuplication、WindowsCapture、WindowsCaptureCUDA、WindowsCaptureROCm、DXCamera。
- macOS：ScreenCaptureKit、CoreGraphics。
- Linux：MSS（当前实际实现）；未来新增 PipeWire/DMABUF 等实现时再加入，不能提前显示为可用。

当捕获模式为“窗口”时，应禁用 DesktopDuplication或给出明确提示，并自动建议 WindowsCapture、WindowsCaptureCUDA或WindowsCaptureROCm。

### 7.3 深度推理与立体合成后端

深度推理和立体合成后端不放入 GUI 下拉框。程序必须先识别操作系统、GPU厂商、可用运行时和资源互操作能力，再按照第 6 节的完整顺序逐项尝试；当前候选初始化失败后才进入下一项。

GUI只显示以下只读状态：

- 实际启用的深度推理后端。
- 实际启用的立体合成后端。
- 是否发生回退。
- 回退前后端名称及失败原因。
- 当前是否存在 CPU回读，以及 `gpu_copy_count`、`zero_copy` 状态。
- 当前捕获资源的类型、像素格式和 DirectML 资源决策（shared、gpu_copy 或 cpu_compat）。

规划中、未打包或能力探测未通过的后端不能进入自动候选链。用户显式选择捕获工具时，只固定捕获工具，不应同时强制指定推理或立体合成后端。

Vulkan主要承担 Windows/Linux 的立体合成与输出，不默认承担深度推理；Metal是 macOS 原生立体合成目标。TensorRT、MIGraphX、OpenVINO、DirectML、CoreML和MPSGraph只参与内部推理后端选择。

## 8. GPU 资源跨 API 传递与设备一致性

### 8.1 Windows Adapter LUID

以下对象必须属于同一物理适配器：

- Desktop Duplication或Windows Graphics Capture使用的 D3D11 Device。
- DirectML使用的 D3D12 Device。
- OpenVINO D3D11 RemoteContext。
- Vulkan PhysicalDevice。
- CUDA、ROCm或Intel XPU推理设备。
- D3D11/oneVPL编码或最终输出设备。

Windows上应以 Adapter LUID作为首要身份，不得只比较厂商 ID、设备名称或显存大小。LUID为空、不一致或无法验证时，禁止共享资源导入，转入一次 GPU复制或CPU兼容回退。

### 8.2 资源与同步要求

跨 API 契约至少记录：

- 资源类型、宽高、像素格式、色彩空间和行跨度。
- 生产者与消费者 API。
- Adapter LUID或等价设备身份。
- 外部内存/共享句柄类型及所有权。
- producer-ready与consumer-done同步对象。
- 资源租约、环形缓冲区槽位和释放时机。
- `gpu_to_cpu`、`gpu_copy_count`和`zero_copy`。

D3D11、D3D12和Vulkan之间不能只传裸指针；必须验证共享句柄类型、格式能力和 fence/semaphore顺序。共享资源失败时不得跨 Adapter 猜测或静默复制。

### 8.3 macOS 与 Linux 设备身份

- macOS：ScreenCaptureKit的 IOSurface、CVMetalTextureCache、CoreML/MPSGraph和Metal合成应绑定同一 `MTLDevice`；必要时记录 registry ID。
- Linux：未来引入 DMA-BUF时应记录 DRM render node、PCI BDF、Vulkan device UUID以及显式/隐式同步方式。当前 MSS 路径没有 GPU 资源身份契约。

## 9. 术语定义

### 9.1 零 CPU 回读

捕获帧从 GPU或共享图像资源直接进入预处理、推理和立体合成，期间没有 `Map`、staging readback、`.cpu()`、`.numpy()`或等价的 GPU→CPU 图像复制。

零 CPU 回读不等于零 GPU复制。

### 9.2 一次 GPU复制

图像始终留在 GPU内存，但因 API、格式、usage flags或共享句柄限制，从生产者资源复制一次到消费者可用资源。例如 D3D11 Texture复制到可供 D3D12/DirectML或Vulkan导入的共享 Texture。此时：

```text
gpu_to_cpu = false
gpu_copy_count = 1
zero_copy = false
```

### 9.3 CPU兼容回退

捕获图像进入 CPU NumPy/内存，必要时再上传到 GPU推理或合成，或直接在 CPU完成。该路径强调可运行性，不属于零 CPU 回读，通常有额外延迟和带宽开销。

## 10. 当前实现与目标架构的主要缺口

本节只记录仍未闭环的事项；“已增加探针/契约”不等于“已完成零拷贝或真机验证”。macOS 原生 CoreML/MPSGraph/Metal 链路本轮不实现，且不修改 `17-macos-zero-copy-capture-inference-survey.md`。

| 缺口 | 当前状态 | 已交付证据 | 未完成/验证边界 |
|---|---|---|---|
| Windows DirectML闭环 | 部分实现 | 设备创建、代表性算子探针、自动回退接入 | D3D11/D3D12共享输入、模型级算子覆盖、输出到 Vulkan 的同步与真机验证 |
| WGC 原生资源契约 | 部分实现 | 原生资源作为 `CapturedFrame.frame`，CPU 兼容帧单独保留；Adapter LUID/格式/尺寸校验与显式 CPU 回退决策 | `windows_capture` 原生对象暴露、DirectML消费者桥和跨 API 同步待验证 |
| 立体合成自动顺序 | 已调整；OpenGL缺口 | 自动顺序 Vulkan → OpenGL → Torch/CPU；OpenGL 能力探针与回退日志 | 项目现有 OpenGL 仅输出/流媒体兼容，不是立体合成实现 |
| Intel 跨 API 闭环 | 部分实现 | D3D11/OpenVINO/Vulkan/oneVPL 入口与 LUID 校验 | Intel 真机句柄、格式、同步、长跑和零回读待验证 |
| macOS 原生链路 | 本轮排除 | 文档明确不实现、不把目标写成现状 | 由其他项目处理；本项目不修改文档 17 |
| Linux 原生 GPU 捕获 | 探针已补齐 | MSS/Xlib、Wayland/PipeWire/DMA-BUF 状态探针 | 原生 PipeWire/DMA-BUF、窗口 surface、显式同步待实现 |
| 其他/国产 GPU | 探针部分实现 | DirectML 设备与代表性算子探针、资源契约报告 | 模型覆盖、驱动白名单、共享句柄和性能仍待逐卡验证 |
| 统一可观测性 | 部分实现 | `[D2S_BACKEND_STATUS]` 日志、GUI 只读状态、资源计数 | 真机端到端计数一致性待验证 |
| GUI 与自动路由 | 代码侧已保持 | 未新增推理/合成下拉框，显示实际后端和回退原因 | 仅允许展示已实际选择的后端，不能据此证明后端闭环 |
| 硬件回归矩阵 | 测试骨架已增加 | `hardware_regression_matrix()`、WGC 原生帧/DirectML 输入/Intel LUID 单元测试入口 | NVIDIA/AMD/Intel/国产 GPU、Linux 和 macOS 原生链路仍需真机回归 |

1. **Windows DirectML闭环（部分实现）**：已增加 DirectML 设备创建、代表性张量算子探针、模型输出设备检查并纳入 Windows 自动推理回退；WGC 原生帧通过 DirectML 输入准备器按共享句柄/同 LUID/桥接方法实际结果选择 shared、gpu_copy 或显式 CPU compatibility。尚未实现通用 D3D11/D3D12 纹理到 DirectML 的生产桥、模型级完整算子覆盖、DirectML 输出到 Vulkan 的外部内存与同步，因此不得宣称端到端零 CPU 回读。
2. **Windows Graphics Capture资源契约（部分实现）**：`WindowsCapture` 回调在暴露 D3D11 资源时将原生资源作为 `CapturedFrame.frame`，CPU 兼容帧单独放在 `cpu_compat_frame`，并传递格式、尺寸、Adapter LUID/UUID/PCI BDF 和生命周期；DirectML 输入准备器只在共享句柄/桥接方法完成后采用 GPU 资源，否则显式选择兼容帧并记录 `gpu_to_cpu`、`gpu_copy_count` 和回退原因；`D2S_WGC_NATIVE_RESOURCE_REQUIRED=1` 可用于资源缺失时快速失败。`windows_capture` 包的实际原生对象暴露、跨 API 同步和真机资源导入仍待验证。
3. **立体合成自动顺序（已调整但 OpenGL 仍是缺口）**：自动解析器已按 Vulkan → OpenGL → Torch/CPU 传递可用性；NVIDIA/AMD 不再在自动模式优先 Triton，Triton 保留为显式/兼容优化路径。当前 OpenGL 探针只识别“立体合成后端”能力；项目已有的 OpenGL 代码是输出/流媒体 fallback，不能当作 GPU 立体合成实现，因此实际自动链仍通常为 Vulkan → Torch/CPU。
4. **Intel真机闭环（部分实现/待验证）**：Desktop Duplication、OpenVINO D3D11、Vulkan/D3D11 资源入口、oneVPL 和 Adapter LUID 校验已具备代码基础；新增通用资源共享决策与遥测字段，但尚未完成 Intel 真机的跨 API 句柄、格式、同步、严格零回读和 4K 长时间稳定性验证。
5. **macOS原生链路（本轮明确排除）**：CoreML depth provider、MPSGraph/CVMetalTextureCache 输入桥和 Metal 立体合成仍由其他项目处理；本项目继续把现有 macOS 路径标记为待验证/兼容回退，不把设计目标写成已实现。
6. **Linux原生GPU捕获（探针已补齐，原生路径仍缺失）**：新增 Linux 能力探针，明确当前只有 MSS + Xlib CPU 捕获；X11 窗口仅按坐标裁剪，不是窗口 surface 零拷贝。Wayland、PipeWire、DMA-BUF、显式同步、DRM/Vulkan 设备身份和厂商原生捕获均仍未实现，自动模式保留 CPU 兼容回退。后续实施入口应先增加 PipeWire screen-capture consumer，再将 DMA-BUF fd、DRM render node、PCI BDF、Vulkan device UUID 和 fence ownership 纳入同一 `CapturedFrame` 资源契约，完成真实 X11/Wayland 硬件回归后才允许进入自动链。
7. **其他及国产GPU能力验证（探针部分实现）**：Windows DirectML 探针可验证运行时、设备创建和基础算子，但模型算子覆盖、D3D11 共享资源、Vulkan 外部内存、Adapter 匹配和性能白名单/黑名单仍分别标记为未探测或待验证，不能仅凭“支持 DX12”推断可用。
8. **统一可观测性（部分实现）**：运行时报告已纳入实际深度后端、合成后端及选择原因、provider 尝试记录；捕获调试字段已纳入资源类型/格式/分辨率、Adapter LUID/UUID/PCI BDF、生命周期、`gpu_to_cpu`、`gpu_copy_count`、DirectML 资源模式、`zero_copy_ready` 和每级回退原因，并由 `[D2S_BACKEND_STATUS]` 与 GUI 只读状态输出。真实硬件的跨 API 计数和端到端帧级一致性仍需验证。
9. **GUI简化与内部自动路由（代码侧已保持）**：本轮没有新增推理或立体合成下拉框；这些后端由能力探测和自动链内部选择，GUI 只应展示只读实际后端、回退原因和资源遥测。若现有界面尚未展示全部只读字段，属于显示层缺口，不得作为后端已实现的证明。
10. **硬件回归矩阵（测试骨架已增加）**：已增加 LUID/资源契约、队列借用资源释放、Linux 能力探针、DirectML 安全探针和 OpenGL 自动选择的单元测试；NVIDIA、AMD、Intel、其他 DX12/国产 GPU、Linux 原生捕获及 macOS 原生链路仍缺对应真机/驱动回归，不应以单一开发机结果推断全平台支持。

## 11. 当前代码核对基线

本矩阵核对了以下当前实现入口：

- `src/desktop2stereo/capture/capture_select.py`
- `src/desktop2stereo/capture/factory.py`
- `src/desktop2stereo/capture/backends/windows_capture_event.py`
- `src/desktop2stereo/capture/backends/windows_desktop_duplication.py`
- `src/desktop2stereo/capture/backends/windows_dxcamera.py`
- `src/desktop2stereo/capture/backends/macos_screencapturekit.py`
- `src/desktop2stereo/capture/backends/macos_coregraphics.py`
- `src/desktop2stereo/capture/backends/linux_mss.py`
- `src/desktop2stereo/stereo_runtime/depth_provider.py`
- `src/desktop2stereo/stereo_runtime/providers/`
- `src/desktop2stereo/stereo_runtime/compute_backend.py`
- `docs/17-macos-zero-copy-capture-inference-survey.md`
- `docs/18-intel-windows-zero-copy-capture-inference-plan.md`

如代码、依赖打包或真机验证状态发生变化，应同步更新本矩阵中的“当前实现状态”“零CPU回读”和“预计GPU复制次数”，不能只更新自动选择顺序。
