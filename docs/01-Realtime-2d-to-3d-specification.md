# D2S Vulkan 实时 2D 转 3D 系统规格书

**文档版本**：4.0
**发布日期**：2026 年 7 月
**规范状态**：Vulkan 原生架构正式规格
**编制依据**：`01.D2S_Vulkan_Migration_Technical_Report.md`
**适用范围**：Desktop2Stereo 新一代实时运行时、OpenXR 查看器与跨平台 GPU 后端

---

## 1. 文档目标

本规格定义 D2S 从单目 RGB 帧到立体画面、三维场景渲染和 OpenXR 提交的新架构。Vulkan 是默认且完整功能的主图形 API；OpenGL 仅作为隔离的兼容 Fallback，用于 Vulkan 不可用或用户显式选择兼容模式的环境。两个后端不得在同一会话中并行管理同一组图形资源。

本规范取代此前以OpenGL、D3D11、CUDA-GL/WGL互操作和多输出上传器为中心的图形运行时描述。经过验证的Python GPU tensor、Capture、Inference和调度实现继续作为新架构基础；旧文档中的其他产品语义只有在本规范重新定义后才继续有效。

### 1.1 建设目标

1. 使用 Vulkan 统一屏幕纹理、立体合成、后处理、三维场景渲染和 OpenXR 交换链。
2. 使用 Filament Vulkan 后端渲染房间、虚拟屏幕、手柄和其他 glTF 2.0 场景资产。
3. 使用 Vulkan Compute 完成 RGB 缩放、深度后处理、视差生成、双目变形、空洞修补、时域稳定和环境光效。
4. 保留各 GPU 平台性能最优的 AI 推理后端，通过外部内存或一次 GPU 内拷贝接入 Vulkan。
5. 采用 latest-frame 调度和有界帧上下文，保证负载升高时延迟不持续累积。
6. 首先交付 Windows/Linux Vulkan + OpenXR，平台具备有效 OpenXR Vulkan Runtime 时再启用对应 XR 输出。
7. 提供功能分级明确的 OpenGL Fallback，使老旧硬件、虚拟机、远程桌面和 MoltenVK 不可用环境仍可进行基本渲染。

### 1.2 非目标

- 不把 OpenGL Fallback 作为与 Vulkan 同等级的性能或功能主路径。
- 不保留 D3D11 OpenXR Session、WGL 互操作、CUDA-GL PBO 或 GL texture uploader。
- 不把已经验证的 Python Capture、Inference Provider 和 latest-frame 调度机械改写为 C++；迁移时保留有效实现，但不保留废弃类名、旧环境变量和旧配置字段的长期兼容适配层。
- 不把 Vulkan Compute 当作神经网络推理框架；深度模型继续由平台推理后端执行。
- 不在实时数据面中进行 CPU 图像回读、NumPy 往返或进程间原始帧复制。
- 不承诺 macOS XR 输出；macOS Vulkan/MoltenVK 能力与具体 OpenXR Runtime 可用性必须分别验证。

---

## 2. 术语和系统边界

| 术语 | 定义 |
|------|------|
| `capture_size` | 输入源的原始像素尺寸。 |
| `render_size` | 深度、视差、双目合成和时域状态使用的唯一工作尺寸。 |
| `scene_size` | OpenXR Runtime 推荐的单眼交换链尺寸。 |
| Relative Depth | 单目模型输出的归一化相对深度，不等同于米制距离。 |
| Parallax Budget | 按 `render_size` 和用户档位计算的左右眼总视差像素预算。 |
| Frame Context | 一帧独占的命令池、命令缓冲、描述符、查询和同步状态。 |
| Latest Frame | 消费者只处理当前最新可用输入，过期输入允许被覆盖。 |
| External Image | 由 Vulkan 导出并由 CUDA/HIP 等推理后端访问的 GPU 图像或缓冲。 |
| Graphics Queue | 执行 Filament 场景渲染、图像合成和交换链写入的 Vulkan 队列。 |
| Compute Queue | 执行立体合成与异步光效计算的 Vulkan 队列。 |
| Graphics Backend | 启动阶段选定的 Vulkan 主后端或 OpenGL Fallback；运行中不可热切换。 |
| Compatibility Mode | 用户显式启用 OpenGL Fallback 的运行模式。 |

系统边界如下：

```text
Desktop / Window / Video / API Frame
                  |
                  v
        Capture and GPU Import
                  |
                  v
       Vulkan Frame Processing Graph
   Resize -> Inference Bridge -> Depth Postprocess
   -> Parallax -> Stereo Warp -> Hole Fill -> Temporal
                  |
                  v
         Vulkan Scene Composition
  Filament scene + virtual screen + asynchronous effects
                  |
                  v
        OpenXR Vulkan Swapchain / Vulkan Output
```

---

## 3. 总体架构

### 3.1 全 Vulkan 流程图

```text
┌────────────────────────────────────────────────────────────────┐
│                       推理层 (厂商专用)                          │
│                                                                 │
│  Windows NVIDIA: CUDA + TensorRT                                │
│  Windows AMD:    ROCm + MIOpen (DirectML 备用)                  │
│  Linux AMD:      ROCm + MIOpen                                  │
│  macOS:          MPSGraph / CoreML                              │
│                                                                 │
│  输出：SBS 立体画面 (厂商私有纹理)                               │
└───────────────────────┬────────────────────────────────────────┘
                        │ external memory / 单次 GPU 拷贝
                        ▼
┌────────────────────────────────────────────────────────────────┐
│                    Vulkan 统一渲染与计算层                        │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Filament (Vulkan 后端)                                  │   │
│  │  - 加载房间.glb / 手柄.glb                               │   │
│  │  - 渲染 3D 场景 + 虚拟屏幕四边形                         │   │
│  │  - 直接输出到 OpenXR Vulkan 交换链                       │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  异步计算队列 (Vulkan 原生)                              │   │
│  │  - 降采样 + 模糊 → Glow 纹理                            │   │
│  │  - 取平均色 + 预计算掩码 → 墙面反射光斑纹理             │   │
│  │  - 消费旧帧，零阻塞                                      │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  OpenXR 提交                                             │   │
│  │  - Projection Layer: 3D 场景 + 虚拟屏幕                 │   │
│  │  - Quad Layer: Glow 特效 / 文字面板 / 虚拟键盘          │   │
│  │  - xrEndFrame 直接提交 VkImage                          │   │
│  └─────────────────────────────────────────────────────────┘   │
└────────────────────────────────────────────────────────────────┘
```

### 3.2 推荐的 Fallback 策略

```text
主路径：Vulkan
    ├── Windows: 原生 Vulkan
    ├── Linux: 原生 Vulkan
    └── macOS: MoltenVK (Vulkan → Metal)

Fallback 路径：OpenGL
    ├── Windows: 原生 OpenGL
    ├── Linux: 原生 OpenGL
    └── macOS: OpenGL 4.1 (功能受限，但可用于基本渲染)
```

触发 Fallback 的条件：

1. Vulkan 驱动不可用，例如老旧硬件、虚拟机或远程桌面环境。
2. macOS 上 MoltenVK 初始化失败，或基准测试确认其性能不可接受。
3. 用户显式选择兼容模式。

Fallback 只能在启动探测阶段选择。Vulkan 已经创建 Device、OpenXR Session 或帧资源后发生错误时，不得在原进程内静默切换到 OpenGL；运行时应结束当前图形会话，向用户报告原因，并通过受控重启进入兼容模式。

OpenGL Fallback 不承诺 Vulkan Compute、异步 Compute Queue、Timeline Semaphore、外部内存零拷贝和完整高级光效。macOS OpenGL 4.1 只保证基本场景与虚拟屏幕渲染。Fallback 若无法从推理/合成后端取得 GPU 驻留的 Left/Right Eye 或 SBS 结果，应明确失败，不得改走 CPU 实时像素链路。

### 3.3 组件划分

| 组件 | 职责 | 实现约束 |
|------|------|----------|
| D2S Host | 生命周期、配置、设备选择、状态与诊断 | Python；不得执行 CPU 逐像素计算 |
| Capture Adapter | 捕获桌面、窗口、视频或 API 帧 | Python Adapter；优先复用现有 WindowsCaptureCUDA/ROCm 等实现，输出 GPU 资源和单调时间戳 |
| Vulkan Device Context | Instance、Device、队列、内存、Pipeline Cache | 全进程唯一设备上下文 |
| Inference Adapter | 运行深度模型并写入 Vulkan 可消费资源 | Python Provider；允许内部使用 GPU tensor，但不得把 CPU 像素往返带入实时主路径 |
| Stereo Compute Graph | 深度后处理到双目时域输出 | Vulkan Compute Pipeline |
| Scene Renderer | glTF 场景、虚拟屏幕、手柄和灯光 | Python 调用 Filament DLL Bridge；这是唯一允许的自有原生代码边界 |
| Effects Compute Graph | Glow、平均色、墙面反射与色彩处理 | 异步 Compute，允许滞后 1 至 3 帧 |
| OpenXR Presenter | 帧预测、视图、交换链和 Layer 提交 | 延续Python OpenXR实现/API；Vulkan Session为主，兼容模式可创建OpenGL Graphics Binding |
| Control Plane | GUI/API、配置快照、日志和遥测 | Python；不读取逐帧像素 |

### 3.4 进程与语言边界

正式应用运行时以 Python 进程为主进程。Capture、Inference、Vulkan Compute、OpenXR Presenter、资源调度和控制面均由 Python 源码实现并在同一运行时内协作。WindowsCaptureCUDA/ROCm、PyTorch、TensorRT 和 OpenXR 延续当前 Python 包/API 的直接调用方式，不为它们新增项目自有绑定层。GPU 密集工作通过这些 Python API 和 SPIR-V Shader 提交，Python 不进行 CPU 逐像素循环。

Filament DLL Bridge 是唯一允许由本项目维护的 C/C++ 原生桥接层。它与 Python 主进程同进程加载，只暴露场景创建、资产加载、状态更新、Vulkan Render Target 绑定和渲染提交所需的窄 C ABI。禁止为 Capture、Inference、Stereo、OpenXR 或 Output 再建立自有 C++ Runtime，也禁止通过 Filament 子进程或共享内存传递整帧图像。

### 3.5 单一资源所有权

- Vulkan Device Context 是所有可渲染图像的最终所有者。
- 推理后端只在明确的 external-memory 生命周期内访问资源。
- OpenXR 交换链图像由 Runtime 创建，应用 acquire 后临时取得写入权，release 后不得继续访问。
- Filament 导入交换链或中间图像时，不得销毁其不拥有的 `VkImage`、内存或同步对象。

---

## 4. 平台和后端矩阵

### 4.1 基线平台

| 平台 | 图形与计算 | 深度推理 | 资源连接 |
|------|------------|----------|----------|
| Windows NVIDIA | 默认请求 Vulkan 1.4，协商最低 Vulkan 1.2 | TensorRT/CUDA | Win32 external memory + external semaphore，目标为零拷贝 |
| Linux NVIDIA | 默认请求 Vulkan 1.4，协商最低 Vulkan 1.2 | TensorRT/CUDA | FD external memory + external semaphore，目标为零拷贝 |
| Windows AMD | 默认请求 Vulkan 1.4，协商最低 Vulkan 1.2 | ROCm/HIP | 优先 external memory；不可用时一次 GPU 内拷贝 |
| Linux AMD | 默认请求 Vulkan 1.4，协商最低 Vulkan 1.2 | ROCm/HIP | FD external memory；失败时一次 GPU 内拷贝 |
| Windows 通用 | 默认请求 Vulkan 1.4，协商最低 Vulkan 1.2 | ONNX Runtime/DirectML 可选适配器 | D3D12 resource 仅存在于推理适配器边界，导入 Vulkan 后不进入渲染架构 |
| macOS Apple Silicon | 默认请求 Vulkan 1.4，按 MoltenVK/Runtime 协商，最低 Vulkan 1.2 | CoreML/MPSGraph | 平台纹理桥或一次 GPU 拷贝；独立验收 |

Intel 与其他 Vulkan 设备只有在对应推理适配器通过本规范的 GPU 驻留和性能验收后，才能列为受支持平台。

OpenGL Fallback 的平台基线为：Windows/Linux 使用驱动可提供的原生 OpenGL，目标版本不低于 4.3；macOS 使用系统 OpenGL 4.1，仅启用基本图形能力。具体功能必须由 capability probe 决定，不能仅按版本字符串推断。

### 4.2 Vulkan 最低能力

正式运行时默认请求 Vulkan 1.4，但最终版本必须服从 `xrGetVulkanGraphicsRequirements2KHR` 返回的 Runtime 最大版本；Vulkan 1.2 是主路径最低版本。Vulkan 1.4 不可用时，只要 Runtime 协商结果满足本节能力要求，不得因为版本低于请求值而直接失败。

正式运行时要求：

- Vulkan 1.2 或更高版本。
- Timeline Semaphore，必须查询 `timelineSemaphore` 支持状态，并在 `VkDeviceCreateInfo.pNext` 链中显式启用 `VkPhysicalDeviceTimelineSemaphoreFeatures`；仅请求 Vulkan 1.2/1.4 不等于已启用该 Feature。
- Synchronization2。
- Descriptor Indexing。
- Dynamic Rendering，若 Filament 集成路径要求 Render Pass，则允许由 Filament 内部管理。
- `VK_KHR_swapchain`，用于非 XR Vulkan 窗口输出。
- 平台对应的 external memory 与 external semaphore 扩展，供零拷贝推理路径使用。
- OpenXR 输出要求 `XR_KHR_vulkan_enable2` 及 Runtime 声明的 Vulkan 版本和扩展。

缺失 Vulkan 必需能力时，默认启动必须给出缺失项。配置允许 Fallback 时，启动器可在尚未创建正式图形会话前选择 OpenGL；否则启动失败。任何情况下都不得静默切换到 D3D11 或 CPU 实时渲染。

---

## 5. Vulkan 设备与资源规格

### 5.1 Instance 和 Device 创建

1. OpenXR 模式必须先创建 `XrInstance`，再通过 `xrGetVulkanGraphicsRequirements2KHR` 获取要求。
2. Vulkan Instance 与 Physical Device 必须满足 OpenXR Runtime 要求；OpenXR 指定设备时不得自行替换。
3. Device 扩展集合由 OpenXR、Filament、外部内存和 D2S Compute Graph 合并生成。
4. 正式构建启用 Vulkan validation 的开关必须可配置；开发和 CI 验证默认启用。
5. Device UUID、驱动版本、队列族和扩展清单必须写入启动日志。
6. 默认请求 Vulkan API 版本 1.4；实际创建版本取请求版本、Runtime 最大版本和 Runtime 最低版本的交集，最低保证 Vulkan 1.2。
7. 创建设备前必须通过 `vkGetPhysicalDeviceFeatures2` 查询 Timeline Semaphore；支持时将 `VkPhysicalDeviceTimelineSemaphoreFeatures.timelineSemaphore` 设为 `VK_TRUE` 并挂入 `VkDeviceCreateInfo.pNext`。
8. Timeline Semaphore 不支持或无法启用时，Vulkan OpenXR 主路径必须在正式 Device/Session 建立前报告原因；允许 Fallback 时通过受控重启进入 OpenGL，不得在已运行会话中静默改用其他 API。

### 5.2 队列模型

系统至少申请一个 Graphics Queue。存在独立 Compute Queue 时必须启用异步光效；否则光效计算提交到 Graphics Queue，但仍维持独立依赖关系。

优先队列拓扑：

```text
Graphics Queue : scene render + virtual screen + OpenXR swapchain
Compute Queue  : stereo synthesis + asynchronous effects
Transfer Queue : capture upload / staging copy, only when useful
```

跨队列共享图像必须使用 Synchronization2 barrier。队列族不同则执行明确的 queue-family ownership transfer；禁止依赖隐式布局和隐式所有权变化。

### 5.3 帧上下文

默认使用 3 个 Frame Context，最大值由 `max_frames_in_flight` 限定为 2 至 4。每个上下文至少包含：

- Graphics/Compute command pool 与 command buffer。
- 每帧 descriptor allocator。
- timestamp query pool 区间。
- 推理完成值、合成完成值和呈现完成值。
- 对应的输入、深度、左右眼、mask 和 temporal 索引。

帧上下文重新使用前必须确认其 timeline 值已完成。系统不得无限创建 command buffer、CUDA work 或待提交帧。

### 5.4 图像格式

| 资源 | 推荐格式 | 必需用途 |
|------|----------|----------|
| Capture RGB | `VK_FORMAT_B8G8R8A8_UNORM` 或捕获原生格式 | sampled、transfer src/dst |
| Linear RGB | `VK_FORMAT_R16G16B16A16_SFLOAT` | storage、sampled |
| Relative Depth | `VK_FORMAT_R16_SFLOAT` | storage、sampled |
| Disparity | `VK_FORMAT_R16_SFLOAT` 或 `R32_SFLOAT` | storage、sampled |
| Occlusion Mask | `VK_FORMAT_R8_UNORM` | storage、sampled |
| Left/Right Eye | `VK_FORMAT_R16G16B16A16_SFLOAT` | storage、sampled |
| Packed SBS | `VK_FORMAT_R8G8B8A8_UNORM` | storage、sampled、transfer src |
| OpenXR Color | Runtime 支持的 sRGB/UNORM 格式 | color attachment、sampled |
| Glow/Reflection | `VK_FORMAT_R16G16B16A16_SFLOAT` | storage、sampled |

格式不支持所需 usage 时，初始化失败或选择同语义的 Vulkan 格式。不得通过 CPU 格式转换规避设备格式能力。

### 5.5 Descriptor 和 Pipeline

- 使用固定 descriptor set layout 描述每个 Compute Pass 的输入、输出和常量。
- 帧间图像采用数组描述符或按 Frame Context 分配的 descriptor set，禁止更新仍在 GPU 使用的 descriptor。
- Compute pipeline 在启动或质量配置切换时创建，不得每帧编译 shader。
- Pipeline Cache 必须按设备 UUID、驱动版本和 shader 版本持久化；不匹配时废弃重建。
- Shader 使用离线编译的 SPIR-V，并在构建阶段执行反射和绑定校验。

---

## 6. 实时帧处理流程

### 6.1 阶段 1：捕获和导入

输入支持显示器、窗口、视频解码器和 API 图像。Capture Adapter 必须产生：

```text
CaptureFrame {
  frame_id
  timestamp_ns
  width, height
  pixel_format
  color_space
  image_origin
  gpu_resource
}
```

实时捕获队列容量为 1。新帧到达且旧帧未消费时覆盖旧帧并增加 `capture_overwrite_count`。不得为了不丢帧而扩大队列并累积延迟。

捕获资源进入 Vulkan 的优先级为：原生 Vulkan 图像、外部内存导入、GPU copy。CPU staging 只允许用于文件、测试图和明确标记的非实时输入。

### 6.2 阶段 2：解析工作尺寸

`render_size` 是立体合成的唯一工作坐标系。所有 Depth、Disparity、Mask、Left/Right Eye 和 Temporal 资源必须与其一致。

固定缩放档位：

| 用户档位 | 内部比例 |
|----------|----------|
| 4K | 1.00 |
| 3K | 0.85 |
| 2K | 0.75 |
| 1K | 0.50 |

只有 4K 级输入应用缩放档位；其他输入默认保持原尺寸。4K 级判断为：长边不小于 3840 且短边不小于 1600，或像素数不小于 UHD 4K 的 85%、长边不小于 3200 且短边不小于 1600。

尺寸必须对齐 Compute Pipeline 的 workgroup 要求并保持输入宽高比。`render_size` 改变时，在帧边界停止接收新工作，等待在途 Frame Context，重建尺寸相关图像并清空时域状态。

### 6.3 阶段 3：RGB 预处理

Vulkan Compute 完成格式转换、色彩空间线性化和尺寸调整，输出 `Linear RGB`。当输入尺寸和格式已满足要求时允许消除无意义 pass。

HDR 输入必须依据捕获 metadata 执行明确的 transfer function 和 tone mapping。任何未知色彩空间必须记录告警，不得默认把 HDR 当作 sRGB。

### 6.3.1 颜色空间与显示编码规范

颜色处理必须遵循“声明语义、只转换一次、禁止静默映射”的原则。所有 Capture、Inference、Vulkan、Filament 和 OpenXR 边界必须携带并验证 `pixel_format`、`transfer_function`、`primaries`、`range` 和 `color_space`；缺失或未知信息不得猜测为 HDR 或 sRGB。

本项目当前产品输出为 SDR sRGB，规范要求如下：

1. 桌面 SDR 捕捉以 sRGB/BT.709 显示参考 RGB 字节进入运行时。通道排列、尺寸调整和张量类型转换不等同于 gamma 或 tone mapping；不得在捕捉、CUDA/Vulkan copy 或 RGBA 打包阶段私自执行 gamma、ACES、Reinhard、曝光或对比度映射。
2. 运行时显示输出契约固定为 `color_space=srgb`、RGBA8、`image_origin=top_left`。浮点 `[0,1]` 到 8-bit 的量化只能执行一次，并且不得把显示参考 sRGB 值再次当作线性 HDR 值处理。
3. OpenXR Projection Layer 的颜色交换链必须使用 API 对应的 sRGB 格式（Vulkan 优先 `VK_FORMAT_R8G8B8A8_SRGB`，允许运行时支持的等价 BGRA sRGB 格式）。不得以 UNORM 作为静默回退；运行时没有可用 sRGB 格式时必须失败并报告原因。
4. Filament 场景、PBR 材质、灯光和环境在 Rec.709/D65 线性工作空间计算。场景曝光只能通过 Filament ColorGrading/相机显示变换实现，不得修改 glTF `baseColorFactor` 伪装成曝光。
5. Tone mapping 只允许作用于明确声明为场景 HDR 的主场景 View；ACES 不是通用的颜色校正，也不得作用于显示参考的屏幕、UI、激光或其他 LDR 内容。
6. 显示参考屏幕纹理必须以 sRGB Vulkan 图像导入并以 `SRGB8_A8` 采样，完成一次 sRGB 解码；屏幕和激光使用无后处理 View 直接写入 sRGB 目标。激光必须不透明，alpha 不得被用作颜色变淡机制。
7. 任何手动亮度、对比度、饱和度、Gamma、色温和色调控制都必须有明确配置来源；中性值必须是恒等变换，默认运行路径不得隐式启用。

规范依据：Khronos [OpenXR XrSwapchain 颜色格式规则](https://registry.khronos.org/OpenXR/specs/1.0/man/html/XrSwapchain.html)、[Vulkan 格式语义](https://registry.khronos.org/vulkan/specs/latest/registry.html)、[glTF 2.0](https://github.com/KhronosGroup/glTF/tree/main/specification/2.0) 和 [Filament HDR/线性渲染及色调管理](https://github.com/google/filament)。

### 6.4 阶段 4：深度推理

Inference Adapter 接收 GPU 驻留的 RGB 输入，输出 `Relative Depth`。适配器合同为：

```text
submit(frame_id, input_resource, output_resource, wait_value, signal_value)
```

约束如下：

- 输出最终必须对齐 `render_size`；模型内部可使用独立尺寸。
- 输出必须声明 near/far 方向、归一化范围、模型和后端版本。
- CUDA/HIP 路径使用 Vulkan 导出的内存和 semaphore；推理完成后通过 external semaphore 交回 Vulkan。
- 不支持可靠 external image 写入时，允许写入后端 GPU buffer，再执行一次 GPU 内拷贝。
- 实时模式禁止 GPU 到 CPU 回读后再上传 Vulkan。
- 推理失败不得复用无标记的陈旧深度；可在限定帧数内复用上一张深度，但必须设置 `depth_stale=true`。

### 6.5 阶段 5：深度后处理

深度后处理由 Vulkan Compute 完成，至少包括：

1. near/far 方向规范化。
2. 非有限值修复和范围裁剪。
3. 对齐 `render_size` 的边缘感知上采样。
4. 可选的轻量双边或引导滤波。
5. 场景切换统计，输出时域重置信号。

Relative Depth 不得作为真实米制 Z 值代入相机模型或 IPD 公式。

### 6.6 阶段 6：视差预算

`max_disparity_px` 表示左右眼总视差预算，不表示单眼位移。默认预算表：

| 短边等级 | comfort | standard | strong | extreme |
|----------|--------:|---------:|-------:|--------:|
| 720 | 24 px | 36 px | 48 px | 64 px |
| 1080 | 32 px | 48 px | 64 px | 80 px |
| 1440 | 48 px | 64 px | 88 px | 112 px |
| 2160 | 64 px | 96 px | 128 px | 160 px |

短边位于等级之间时线性插值。最终宽高比超过 2:1 时应用：

```text
aspect_factor = clamp(2.0 / aspect, 0.70, 1.0)
max_disparity_px = base_budget * aspect_factor
```

视差预算只在尺寸、宽高比阈值或 preset 改变时重算，不得随每帧内容抖动。

### 6.7 阶段 7：视差场

核心语义为：

```text
disparity_px = depth_response(relative_depth, convergence)
             * max_disparity_px
             * depth_strength

left_shift_px  = +disparity_px / 2
right_shift_px = -disparity_px / 2
```

`depth_response` 输出建议限制在 `[-1, 1]`。`convergence` 只定义零视差平面，`depth_strength` 只定义连续强度增益，二者不得合并为旧式 IPD 经验乘法链。

Compute Shader 必须对 NaN、越界采样和极端视差进行限幅，并生成供 Warp 与 Hole Fill 使用的边缘风险数据。

### 6.8 阶段 8：双目变形

Stereo Warp 使用反向采样生成 Left Eye 和 Right Eye，并写出原始 disocclusion mask。必须满足：

- 两眼使用同一份不可变视差场。
- 位移单位为 `render_size` 坐标中的像素。
- 边界采样策略显式定义，不得读取图像外内存。
- 前后景冲突按深度优先级处理。
- Warp 只负责几何变形，不以模糊掩盖超预算视差。

### 6.9 阶段 9：遮挡与空洞修补

实时默认采用方向性、边缘感知的多轮 Compute Pass：mask 膨胀、背景方向搜索、候选颜色选择和羽化合成。修补不得跨越明确的前景深度边缘，也不得改变未被 mask 标记的有效区域。

大面积生成式补图不进入默认实时管线。若未来提供离线质量模式，必须使用独立产品模式和资源预算。

### 6.10 阶段 10：时域稳定

时域稳定以左右眼、深度、mask 和历史置信度为输入。静态区域增加历史权重，运动、遮挡和场景切换区域降低历史权重。

以下事件必须清空历史资源：

- `render_size` 改变。
- 捕获源改变或时间戳回退。
- 模型、depth direction 或 depth response 改变。
- `temporal_enabled` 状态改变。
- 场景切换检测触发。
- Device lost 或交换链重建。

时域处理不得修改 Parallax Budget 的定义，也不得因历史帧滞后扩大有效视差。

### 6.11 阶段 11：打包和场景合成

Stereo Compute Graph 的标准输出是完整分辨率 Left Eye 和 Right Eye。SBS 仅作为虚拟屏幕纹理或非 XR 输出的打包形式：

- Full-SBS：`2 * render_width` × `render_height`。
- Half-SBS：`render_width` × `render_height`，每眼横向缩放为一半。
- Eye Pair：两个独立 Vulkan Image，供 OpenXR 场景直接采样。

Filament Scene Renderer 必须：

1. 加载 glTF/GLB 房间、手柄和场景资产。
2. 使用 Left/Right Eye 或 SBS 对应区域更新虚拟屏幕材质。
3. 按 OpenXR 每眼 pose 和 FOV 渲染 Projection Layer。
4. 将 Glow、墙面反射和 UI Layer 作为 Vulkan 资源采样。
5. 直接写入 acquire 的 OpenXR `VkImage`，不经过中间 D3D11/GL 交换链。

---

## 7. 异步光效计算

Effects Compute Graph 消费最近完成的屏幕颜色，但不得阻塞当前场景渲染。它包含：

```text
Screen Color N
  -> Downsample
  -> Separable Blur
  -> Glow Texture N
  -> Average Color / Histogram
  -> Reflection Mask Composite
  -> Reflection Texture N
```

场景渲染使用 `latest_effects_ready_index` 指向的最近完成纹理。没有新结果时继续使用上一结果。允许光效落后屏幕 1 至 3 帧；超过上限时丢弃最旧待处理任务。

主图形队列不得等待光效 Compute Queue。只有资源首次创建、尺寸重建和关闭阶段允许全局等待。

---

## 8. OpenXR Vulkan 提交规格

### 8.1 Session 初始化

OpenXR Presenter 必须使用 Vulkan Graphics Binding 创建 Session。禁止创建 D3D11 Session 后再桥接 Vulkan 图像。

初始化顺序：

```text
xrCreateInstance
-> xrGetSystem
-> xrGetVulkanGraphicsRequirements2KHR
-> create Vulkan instance/device for the XR system
-> xrCreateSession with XrGraphicsBindingVulkan2KHR
-> enumerate view configuration and swapchain formats
-> create color swapchains
-> xrBeginSession
```

### 8.2 帧循环

```text
xrWaitFrame
-> xrBeginFrame
-> xrLocateViews
-> acquire/wait swapchain image per view
-> record Filament/Vulkan rendering
-> submit graphics work
-> wait only for required swapchain completion
-> release swapchain images
-> xrEndFrame
```

每眼 Projection View 必须使用本帧 `xrLocateViews` 返回的 pose 和 FOV。场景相机 near/far clip 应由配置显式给出，并适配大型 glTF 场景；不得依赖旧查看器默认值。

### 8.2.1 Vulkan 输出图像环与异步纹理管理

运行时屏幕图像不得使用单个可变纹理作为跨线程共享缓冲。Vulkan 主路径必须为左右眼分别创建有界的输出图像环，默认容量为 3，可在 2 至 4 之间配置：

```text
Left  VkImage[0..N-1]  <- CUDA external memory import once per slot
Right VkImage[0..N-1]  <- CUDA external memory import once per slot
                         |
                         +-> Filament Texture cache, one import per VkImage
```

每个环槽的 Vulkan image、image view、CUDA external-memory mapping 和 Filament Texture 必须保持稳定。这里的“屏幕源图像”是推理/合成输出、由 Filament 屏幕材质采样的图像，不是 OpenXR 眼睛输出交换链图像；两者必须使用不同的同步契约。正常帧不得执行以下操作：

- 重新创建或销毁运行时屏幕纹理；
- 为每帧分配新的 external-memory handle；
- 将 GPU 图像回读到 CPU；
- 用 `vkDeviceWaitIdle`、`vkQueueWaitIdle` 或 `flushAndWait` 作为每眼同步。

每张屏幕源图像的持久化记录至少包含：`VkImage`、格式、extent、当前 `VkImageLayout`、producer queue family、consumer queue family、ring slot、producer-ready point 和 consumer-release point。帧生命周期固定为：选择可复用槽位 -> CUDA 写入 -> producer signal -> Vulkan barrier 将图像转换为 `VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL` 并完成 queue ownership transfer -> Filament 采样对应的持久化外部纹理 -> graphics completion signal -> release queue ownership/barrier -> 生产端回收槽位。槽位在 graphics completion 前不得被 CUDA 重写。`set_screen_ready_semaphore`（或后续等价 ABI）只表示屏幕源图像 producer-ready，不得被用作 OpenXR 输出目标完成信号；Filament 输出目标必须使用独立的 render-finished/present 同步。

直接把 Vulkan 外部图像交给 Filament 采样是 Vulkan 主路径的目标优化，不得因为当前 Bridge 不完整而永久关闭。每个源 `VkImage` 只允许创建一次对应的 Filament 外部纹理；正常帧只更新材质绑定和槽位状态，不重复 import、复制或回读。若平台不支持安全的 external memory、external semaphore/timeline、layout transition、queue ownership transfer 或 Filament 外部纹理接入，运行时必须在能力探测阶段或首次绑定失败时记录原因，并自动退回一次 GPU copy 的屏幕路径/Quad Layer；不得把不完整的同步契约直接提交给 Filament，也不得崩溃。

平台、CUDA Runtime 或旧 Bridge ABI 不支持完整源图像同步时，才允许退回 `cudaStreamSynchronize` 兼容安全栅栏。该降级必须显式标记为兼容路径，不能伪装成零拷贝；输出槽位仍须通过 producer lease 和 consumer completion 保护，直到 Filament 完成采样后才可复用。

Filament Bridge 必须缓存每眼所有已导入的环槽纹理，并只更新当前材质绑定。Projection Layer 两眼应先完成 acquire/wait，再批量提交，整帧只执行一次必要的完成等待，随后统一 release 两眼 OpenXR 图像。

### 8.2.2 Filament 显示层与场景层

每眼至少维护两个颜色语义明确的 View：

```text
HDR scene View: environment + PBR + controllers -> linear Rec.709 -> scene tone mapping -> sRGB swapchain
LDR display View: virtual screen + UI/laser -> no post-processing/tone mapping -> sRGB swapchain
```

主场景 View 不得看见 LDR 显示层图元；LDR View 不得启用场景 ColorGrading。两个 View 必须在同一帧先提交主场景、再提交 LDR 显示层，并共享相机、深度和目标 viewport。

### 8.3 Composition Layer

- 主场景使用 `XrCompositionLayerProjection`。
- 文本面板、虚拟键盘或确需独立采样率的 UI 可使用 Quad Layer。
- Glow 和墙面反射默认合成在 Projection Layer 场景中；只有经过延迟和视觉验证后才拆为独立 Layer。
- `xrEndFrame` 的 layer 列表只引用已完成并 release 的交换链图像。

### 8.4 非 XR 输出

非 XR 桌面预览使用 Vulkan Swapchain；无窗口输出使用 Vulkan Image 导出给编码器或 API Consumer。两者复用 Stereo Compute Graph，但不复用 OpenXR 帧循环。

非 XR 输出默认使用 Vulkan。兼容模式使用独立 OpenGL Presenter，但不得加载旧 viewer 模块或共享 Vulkan 资源所有权。

---

## 9. 配置与热更新

控制面向运行时提交不可变 `RuntimeConfigSnapshot`：

```text
RuntimeConfigSnapshot {
  version
  graphics_backend
  capture_source
  output_target
  render_scale_tier
  depth_backend
  depth_model
  parallax_preset
  depth_strength
  convergence
  depth_response
  hole_fill_quality
  temporal_enabled
  temporal_strength
  scene_asset
  scene_near_clip
  scene_far_clip
  color_space
  debug_flags
}
```

| 参数组 | 应用方式 | 资源动作 |
|--------|----------|----------|
| Graphics Backend (`auto/vulkan/opengl`) | 不可热更新 | 返回 restart-required，重启后重新探测并创建图形会话 |
| 视差、汇聚、时域强度 | 下一帧边界热更新 | 更新 uniform；必要时 reset temporal |
| Hole Fill 参数 | 下一帧边界热更新 | 更新 uniform |
| Render Scale | 受控重配置 | drain Frame Context、重建尺寸资源 |
| Depth Model/Backend | 受控重配置 | 停止提交、重建 Inference Adapter、reset temporal |
| Capture Source | 受控重配置 | 重建 Capture Adapter 和尺寸资源 |
| Scene Asset | 异步加载后切换 | 创建当前 Backend 的新场景资产，帧边界交换 |
| OpenXR Swapchain Format | Session 重建 | 重建 OpenXR Session 或 Swapchain |

旧配置字段不得在运行时静默映射。遇到已删除字段时，配置加载应报告明确错误和新字段名称，由配置迁移工具一次性转换文件，而不是在核心运行时长期兼容。

---

## 10. 调度、同步与延迟

### 10.1 Timeline 约定

每个帧使用单调递增的 timeline 值表达依赖：

```text
capture_ready(frame_id)
-> inference_done(frame_id)
-> stereo_done(frame_id)
-> scene_done(frame_id)
-> present_released(frame_id)
```

Vulkan 内部同步优先使用 Timeline Semaphore + Synchronization2。与 CUDA/HIP 交界使用平台支持的 external semaphore。只在无法表达 timeline 的外部 API 上使用 binary semaphore。

图像槽位的同步点必须随输出契约传递，至少包含 `frame_id`、`ring_slot`、producer ready point、consumer completion point 和 image ownership。`ready_timeline=None` 只能表示当前兼容路径已在发布前完成 CUDA stream 同步，不得被解释为 GPU 工作已经没有依赖。

Timeline Semaphore 是设备能力，不由 Vulkan API 版本号自动开启。所有 NVIDIA、AMD、Intel 和 MoltenVK 目标都必须执行运行时 capability probe；Vulkan 1.4 请求失败或被 Runtime 裁剪到 1.2 不构成错误，只要 Timeline Semaphore 等最低能力满足即可。

禁止在正常帧循环中调用 `vkDeviceWaitIdle`、`vkQueueWaitIdle` 或等价的全设备同步。

### 10.2 反压策略

- Capture Queue：容量 1，覆盖旧帧。
- Inference：最多 1 个当前任务和 1 个候选最新帧。
- Stereo/Scene：受 `max_frames_in_flight` 限制。
- Effects：最多保留 1 个待处理任务，旧任务可被覆盖。
- Presenter：严格跟随 OpenXR `xrWaitFrame`，不得把预测帧无限排队。

GPU 落后时优先丢弃未开始处理的旧输入，不取消已提交 GPU 工作，不继续提交无界任务。

### 10.3 帧时间预算

90 Hz XR 目标的参考预算：

| 阶段 | 目标时间 |
|------|---------:|
| Capture import + RGB preprocess | <= 0.8 ms |
| Depth inference | <= 5.0 ms |
| Depth postprocess + stereo synthesis | <= 2.5 ms |
| Filament scene render | <= 2.5 ms |
| Queue/synchronization/application overhead | <= 0.8 ms |
| 总应用 GPU 关键路径 | <= 10.0 ms |

光效计算不计入主关键路径，但必须在平均 3 帧内产出。不同硬件和模型应分别记录 P50、P95、P99，不能用 CPU enqueue 时间代替 GPU 时间。

---

## 11. 错误处理与恢复

| 故障 | 处理要求 |
|------|----------|
| Vulkan 初始化失败 | 输出缺失版本、扩展、格式或队列能力；满足 Fallback 条件时请求以 OpenGL 兼容模式受控重启，否则退出 |
| External memory 导入失败 | 当前平台规范允许时切到一次 GPU copy，并永久标记本次会话路径；不切 CPU |
| 推理提交失败 | 丢弃当前帧；在限定次数内重试，连续失败则停止运行时 |
| Shader/Pipeline 创建失败 | 输出 shader hash、stage 和 validation 信息并停止启动 |
| OpenXR Session loss | 停止提交、释放 Session 资源并按 Runtime 状态重建 |
| Swapchain out of date | 在安全帧边界重建对应资源 |
| `VK_ERROR_DEVICE_LOST` | 收集 device fault/validation 信息，终止当前 Device；不得在未知状态继续渲染 |
| glTF 资产失败 | 阻止场景切换并保留当前有效场景；首次场景失败则停止启动 |

所有降级必须是本规范明确定义的 Vulkan/GPU 路径。禁止以“保证能显示”为由静默启用旧 API 或 CPU 像素链路。

---

## 12. 可观测性

### 12.1 每帧指标

运行时至少记录：

```text
frame_id
capture_timestamp_ns
capture_to_present_ms
capture_overwrite_count
depth_stale
gpu_preprocess_ms
gpu_inference_ms
gpu_stereo_ms
gpu_scene_ms
gpu_effects_ms
xr_wait_ms
xr_submit_ms
present_interval_ms
frames_in_flight
active_config_version
```

GPU 阶段时间必须来自 Vulkan timestamp query 或推理后端 GPU event。`xr_wait_ms`、`xr_submit_ms` 和显示间隔必须与算法 GPU 时间分开统计。

### 12.2 启动报告

启动日志必须包含：

- OS、CPU、GPU、驱动和 Vulkan API 版本。
- Device UUID、队列族和启用扩展。
- OpenXR Runtime、System、交换链尺寸和格式。
- 推理模型、后端、精度和互操作模式。
- Scene Renderer 版本、选定 Graphics Backend 和 Fallback 原因。
- Shader/Pipeline Cache 命中状态。
- 最终 `render_size`、`scene_size` 和帧上下文数量。

### 12.3 用户可见状态

GUI显示启动、模型准备、运行、重配置、Session恢复和失败状态。技术日志通过Python Runtime的结构化事件传给控制面；GUI不解析任意stdout文本来推断运行状态。

---

## 13. 测试与验收

### 13.1 Vulkan 正确性

| 测试 | 通过标准 |
|------|----------|
| Validation 全流程 | 30 分钟压力运行无 error；warning 有明确豁免记录 |
| Resource lifetime | 无提前释放、重复释放、在途 descriptor 更新 |
| Image layout | 每个 pass 的读写布局和 barrier 可由 capture 验证 |
| Queue ownership | 独立 Compute/Transfer 队列设备上无 ownership 错误 |
| Device lost 注入 | 运行时停止并生成完整诊断，不继续提交 |

### 13.2 算法正确性

| 测试 | 通过标准 |
|------|----------|
| 尺寸一致性 | 所有合成中间图像严格等于 `render_size` |
| 深度方向 | 测试场景近景和远景视差方向符合 metadata |
| 视差公式 | GPU 结果与参考实现误差 <= 0.01 px 或 FP16 合理误差 |
| 双眼对称性 | `left_shift = +d/2`，`right_shift = -d/2` |
| 空洞填充 | 未越过强深度边缘，无未初始化像素 |
| Temporal reset | 所有规定事件后历史状态被清空 |
| 色彩一致性 | sRGB/HDR 测试图在场景和输出端无重复 gamma |

### 13.3 OpenXR 验收

- 使用至少两个 Windows OpenXR Runtime 完成启动、运行、Session loss 和退出测试。
- 左右眼 pose、FOV、eye order 和虚拟屏幕采样区域正确。
- 交换链 acquire/wait/release 顺序符合规范。
- 大型 glTF 场景 near/far clip 正确，无截断或深度精度明显异常。
- 90 Hz 目标下 30 分钟无持续帧队列增长，P95 capture-to-present 满足产品阈值。

### 13.4 性能验收

| 项目 | 通过标准 |
|------|----------|
| CPU readback | 正式实时路径为 0 次/帧 |
| Vulkan 主路径 API 隔离 | D3D11/OpenGL/WGL 调用为 0 |
| 主路径 GPU copy | NVIDIA 零拷贝；其他平台不超过一次明确 GPU 内拷贝 |
| Frame Context | 数量固定，无运行时持续增长 |
| Capture queue | 始终有界，丢帧发生在 latest-frame 覆盖点 |
| Effects | 不阻塞 Graphics Queue，平均滞后 <= 3 帧 |
| Pipeline creation | 稳态帧循环中为 0 次 |

OpenGL Fallback 必须单独验收，证明会话内没有 Vulkan/D3D11/WGL 混合调用、没有 CPU 实时像素回读，并正确标记受限功能。

### 13.5 画质与舒适度

- 标准测试集覆盖 UI、字幕、人物、快速运动、细线条、透明和镜面场景。
- 专家评审不得出现持续双影、明显左右眼不一致、空洞闪烁和边缘撕裂。
- 用户舒适性测试中无不适报告率目标不低于 90%。
- comfort、standard、strong、extreme 的实际视差必须单调递增并保持限幅。

---

## 14. 交付阶段

### 阶段 A：Vulkan/OpenXR 骨架

- 建立 OpenXR Vulkan Session、交换链和每眼清屏提交。
- 建立 Device Context、Frame Context、Timeline 和 GPU timing。
- Filament Vulkan 后端渲染最小 glTF 场景到 OpenXR 交换链。

退出条件：Validation 无 error，头显中稳定显示正确双眼场景。

### 阶段 B：立体 Compute Graph

- 完成 RGB preprocess、Depth postprocess、Parallax、Warp、Hole Fill 和 Temporal shader。
- 建立固定资源图、descriptor 和 pipeline cache。
- 使用离线/测试深度输入验证左右眼结果。

退出条件：算法正确性测试和 Vulkan resource 测试全部通过。

### 阶段 C：推理互操作

- 接入 NVIDIA CUDA/TensorRT external memory 路径。
- 接入 AMD ROCm/HIP 路径及一次 GPU copy 方案。
- 建立 backend capability probe 和明确错误报告。

退出条件：正式实时路径无 CPU readback，互操作稳定性和性能达标。

### 阶段 D：场景与光效

- 接入正式房间、手柄、虚拟屏幕材质。
- 完成异步 Glow、平均色和墙面反射。
- 验证 compute 滞后不影响主队列帧节奏。

退出条件：完整场景和光效连续运行 30 分钟无资源与同步错误。

### 阶段 E：产品化

- 接入控制面、配置快照、结构化日志和故障报告。
- 完成 OpenXR Runtime、GPU、分辨率和画质矩阵测试。
- 完成 Vulkan 主路径与 OpenGL Fallback 的启动探测、功能分级和独立发布验证。
- 删除旧运行时代码、旧桥接依赖和废弃配置字段。

退出条件：本规范全部验收项通过；发布包只包含 Vulkan 主路径和本规范定义的隔离 OpenGL Fallback，不包含旧图形桥接后端。

---

## 15. 架构清理要求

Vulkan 主路径和新 OpenGL Fallback 进入正式主线前，必须从运行时和构建系统中移除：

- 与新 `GraphicsBackend` 接口无关的旧 OpenGL viewer、场景渲染和资源封装。
- D3D11 OpenXR Graphics Binding 和交换链实现。
- WGL_NV_DX_interop2、CUDA-GL interop、PBO uploader 和 GL mipmap 实时逻辑。
- 旧 OpenGL/D3D11/CPU fallback 分支及其配置开关；只保留本规范定义的新 OpenGL 兼容模式。
- 旧 viewer shader 对 IPD、stereo scale、max shift ratio 的强度解释。
- CPU NumPy/PIL 逐帧 RGB、Depth、SBS 往返路径；Python 中的 GPU tensor、GPU handle 和 Vulkan绑定调用属于正式路径。
- 仅为旧模块名、旧类接口和旧环境变量存在的适配器。

删除应通过代码、依赖、构建产物和配置 schema 四个层面完成。文档或注释不得继续把已删除路径描述为可用方案，也不得把归档中的 OpenGL 旧架构等同于新 Fallback。

---

## 15.1 Filament Bridge 构建与交付规则

Filament Bridge 是本项目唯一需要预编译的自有原生组件。正式构建必须由 GitHub Actions 按平台矩阵完成，不要求用户在本地编译 Filament SDK 或 Bridge。

```text
Bridge 源码变更
    -> GitHub Actions 三平台 CI
    -> Windows x86_64: filament_bridge.dll
    -> Linux x86_64:   libfilament_bridge.so
    -> macOS arm64:    libfilament_bridge.dylib
    -> 上传 Actions Artifact 或 GitHub Release
    -> 下载对应平台产物到 src/xr_viewer/native/<platform>/
    -> Python ctypes 加载并运行
```

构建和运行约束：

1. Windows、Linux、macOS 的 Bridge 必须在对应平台 runner 上编译，不得用交叉编译结果替代平台验证。
2. CI 产物必须包含 Bridge、所需 Filament 运行库、版本信息和 SHA-256；产物上传到 Actions Artifact 或 Release，不以提交编译目录代替发布。
3. 本地默认只下载并使用 CI 已验证的对应平台二进制。日常修改 Python 代码、配置、GUI 或资源时，不得触发 Filament 本地重编译。
4. 只有 Bridge 源码、C ABI、Filament 版本或平台构建配置发生变化时，才重新触发三平台 Bridge CI 构建。
5. 本地编译仅用于 CI 不可用时的故障诊断、调试器定位或尚未提交的 Bridge 原生代码验证；本地编译结果不得作为正式跨平台发布依据。
6. 二进制下载后必须放入 `src/xr_viewer/native/` 的平台目录，由 Python 根据当前操作系统加载对应文件；缺少或校验失败时必须在启动探测阶段报告明确错误。
7. Bridge 的 C ABI 必须保持窄接口和版本化。新增 Filament 能力应先扩展 Bridge 接口并通过 CI 矩阵验证，不得让 Python 直接依赖未封装的 Filament C++ ABI。

---

## 15.2 Filament Bridge 接口完整性清单

窄 C ABI 不是只保留当前已有函数，而是覆盖本项目实际使用的 Filament 能力。接口按业务功能维护，禁止按 Filament 全部类和方法自动导出。以下清单是 Bridge 的完整功能边界；新增功能必须先归入现有功能域，或明确新增功能域后再实现。

| 功能域 | 必须覆盖的能力 | 接口要求 |
|--------|----------------|----------|
| ABI 与版本 | ABI 版本、Filament 版本、平台和架构查询 | Python 加载后先校验版本，不匹配立即失败 |
| 生命周期 | Vulkan Bridge/Preview 创建、初始化、销毁、资源释放 | C++ 持有所有 Filament 对象，Python 只持有不透明 handle |
| 错误处理 | 最近错误、错误码、失败阶段、可诊断描述 | 所有返回值必须可判断成功或失败，错误字符串由 Bridge 管理 |
| Vulkan/OpenXR | Python 传入 Instance、Physical Device、Device、Queue、Queue Family；导入 OpenXR swapchain image；绑定 acquired image | 不接管 Python/OpenXR 所有权，不创建第二套 Vulkan Device |
| Swapchain 与帧 | swapchain image 注册、acquired image 选择、尺寸/格式更新、begin/end frame、重建 | 明确 image 生命周期和 GPU 同步边界，不在 Bridge 内隐式等待无界时间 |
| 场景资产 | GLB 加载、资源上传完成、场景切换、场景卸载、last-good scene 保留 | 支持静态 GLB、PBR 纹理、透明材质和大型场景；加载失败可诊断 |
| 场景对象 | 场景根节点、对象可见性、局部/世界变换、虚拟屏幕和手柄状态 | 只暴露项目需要的对象句柄或命名操作，不暴露 Filament Entity 内部结构 |
| 相机 | look-at、视图矩阵、每眼投影、frustum、near/far、viewport | 支持预览单视图和 OpenXR 左右眼视图，坐标系和单位必须固定 |
| 输出渲染 | Preview window、Vulkan Render Target、OpenXR Projection Layer 目标、render/present | Bridge 只提交已绑定目标，不管理 `xrEndFrame` 和 Layer 生命周期 |
| 动画 | 动画数量、名称/时长、播放时间、循环、暂停、动画选择和逐帧应用 | 时间由 Python 驱动；动画状态不得依赖 Bridge 内部墙钟 |
| 光照与天空盒 | 场景曝光/亮度、天空盒亮度、方向光/填充光、颜色和方向 | 天空盒与场景主体必须能独立控制，不得通过无效的反向曝光补偿实现 |
| 材质与颜色 | 材质参数覆盖、亮度、对比度、饱和度、Gamma、色温、色调 | 统一在渲染前或材质/后处理明确阶段处理，参数范围和中性值由规格定义 |
| 后处理与特效 | Glow、星光闪烁、平均色、墙面反射、色彩后处理、时间参数 | 每项特效都要有启用/禁用、参数更新、时间推进和资源失败处理接口 |
| 资源与窗口 | Preview viewport resize、surface/window 变化、纹理尺寸变化 | Resize 必须在帧边界处理，不得遗留旧 Render Target 或纹理引用 |
| 统计与诊断 | 资源状态、加载耗时、渲染耗时、GPU 错误、能力报告 | 支持日志和 capability report，但不从 C ABI 返回逐帧大块像素数据 |

当前接口实现、待实现接口和测试状态必须在 Bridge 接口清单中逐项标记。任何功能只有同时具备 C ABI 声明、Python wrapper、CI 编译验证和至少一个运行测试，才能标记为完成。不得因为某个 Filament C++ 类未导出而视为遗漏；判断标准是本表定义的产品能力是否完整。

---

## 16. 规范性引用

1. Khronos Vulkan 1.3 Specification。
2. Khronos OpenXR 1.1 Specification。
3. `XR_KHR_vulkan_enable2` Extension Specification。
4. Vulkan Synchronization2、Timeline Semaphore、External Memory 与 External Semaphore 相关扩展规范。
5. glTF 2.0 Specification。
6. Filament Documentation and Native API Reference。
7. ISO/IEC TR 23090-27:2025, render-based immersive media architectures。
8. T/UWA 035-2025，基于双目视差的裸眼 3D 系统参考架构与通用技术要求。

---

## 附录 A：缩略语

| 缩略语 | 全称 |
|--------|------|
| D2S | Desktop to Stereo |
| DIBR | Depth-Image-Based Rendering |
| SBS | Side-by-Side |
| XR | Extended Reality |
| GPU | Graphics Processing Unit |
| SPIR-V | Standard Portable Intermediate Representation - V |
| GLB | Binary glTF |
| HDR | High Dynamic Range |
| P50/P95/P99 | 延迟或帧时间分位数 |

---

本规格自 Vulkan 主线开发启动之日起生效。任何与本规范冲突的旧运行时设计，不再作为新架构的实现依据。

## 17. 全量符合性追踪

本规格的每一项架构、平台、推理、渲染、OpenXR、配置、性能、安全和测试要求，统一登记在
[`docs/requirements-matrix.md`](requirements-matrix.md)。矩阵中的每个需求必须关联实现位置、测试或人工验收方式和状态；未登记、无映射或无验收条件的实现不得视为完成。

发布候选版本必须通过 `src/tools/check_compliance.py --strict`、自动化测试、三平台 Bridge CI 和必要的 GPU/OpenXR 实机验收。
