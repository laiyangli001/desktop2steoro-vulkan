# Desktop2Stereo Python Vulkan 工程设计规范

**文档版本**：4.0
**发布日期**：2026 年 7 月 18 日
**规范状态**：目标态工程设计
**上位规格**：`01-Realtime-2d-to-3d-specification.md`
**技术依据**：`01.D2S_Vulkan_Migration_Technical_Report.md`

---

## 1. 文档定位

本文规定 Desktop2Stereo Python Vulkan 运行时的工程实现方式，包括代码组织、模块接口、线程和队列、GPU资源、推理互操作、Filament DLL Bridge、OpenXR提交、配置、诊断、测试和交付。

`docs/01` 定义系统必须表现出的行为和验收结果；本文定义工程如何实现这些行为。二者冲突时以 `docs/01` 为准，并同步修订本文。

本文不是旧OpenGL/D3D11运行时的兼容模块地图，也不要求为迁移而复制旧架构。经过验证的Python Capture、Inference Provider、latest-frame调度和诊断代码属于可复用实现；图形与OpenXR路径按Vulkan目标重新组织。本文定义的OpenGL Fallback是隔离兼容后端，不等同于归档中的旧viewer、Panda3D、WGL或D3D11桥接路径。

### 1.1 工程目标

1. 应用、实时调度、Capture、Inference、Vulkan、OpenXR和Output统一使用Python源码实现。
2. Vulkan 负责默认主路径的图形、通用计算、资源管理和显示提交；OpenGL 只承担受限 Fallback。
3. OpenXR Session 默认使用 Vulkan Graphics Binding；兼容模式允许使用原生 OpenGL Graphics Binding。
4. AI 推理保留厂商最优后端，通过明确的 GPU 互操作接口接入。
5. 捕获、推理、立体合成、场景渲染和呈现使用固定容量资源池，不产生无界队列。
6. 正常帧循环不进行 CPU 像素回读，不调用全设备空闲等待。
7. 平台差异只存在于 Capture Adapter、Inference Adapter 和系统句柄层。
8. 所有关键 GPU 工作可测量、可验证、可故障定位。
9. 图形后端在启动探测阶段确定，运行中不静默切换；Fallback 必须通过受控重启进入。
10. 项目自有原生代码仅允许存在于Filament DLL Bridge；其他模块不得新增C/C++、Rust或自定义扩展模块。

### 1.2 明确排除

- 不把现有Python Capture、Inference Provider和OpenXR行为机械改写为C++。
- 不把 OpenGL 嵌入 Vulkan 资源图；OpenGL Fallback 必须作为独立 Graphics Backend 实现。
- 不恢复 D3D11、WGL、CUDA-GL、PBO uploader 或旧 OpenGL viewer。
- 不通过CPU NumPy数组、PIL Image或共享内存传递实时整帧像素；Python中的GPU tensor和GPU handle属于正式数据路径。
- 不保留旧类名、旧环境变量、旧 YAML 字段的长期运行时适配器。
- 不把 Filament 封装为通过 CPU 图像传输工作的独立渲染子进程。
- 不允许推理后端直接管理 OpenXR Session 或场景资源。
- 不为CUDA、ROCm、TensorRT、WindowsCapture或OpenXR新增项目自有Binding；继续直接使用其现有Python包/API。

---

## 2. 技术基线

### 2.1 语言与构建

| 项目 | 规定 |
|------|------|
| 核心语言 | Python；版本按平台锁定文件确定 |
| Python运行方式 | 源码直接执行，正式入口保持为 `python src/main.py` |
| 包管理 | 固定版本清单；依赖必须可校验 SHA-256 |
| 图形 API | 默认请求 Vulkan 1.4，按 OpenXR Runtime 协商，最低 Vulkan 1.2；OpenGL 4.3+ Fallback，macOS 为 OpenGL 4.1 受限模式 |
| XR API | OpenXR 1.1；主路径使用 `XR_KHR_vulkan_enable2`，Fallback 使用 Runtime 支持的 OpenGL Graphics Binding |
| 场景引擎 | Filament Vulkan Backend 为主；OpenGL 使用隔离的兼容 Scene Renderer |
| Shader | GLSL/HLSL 离线编译为 SPIR-V |
| 测试 | pytest；GPU/OpenXR集成测试使用独立marker和实机矩阵 |
| 自有原生代码 | 仅Filament DLL Bridge；由独立CMake工程预编译 |

Python产品代码必须直接支持Windows x86_64、Linux x86_64和macOS arm64。只有Filament DLL Bridge按平台预编译；不得因为MoltenVK或Bridge能够加载就宣称macOS OpenXR可用。

### 2.2 依赖版本管理

所有第三方二进制依赖必须在版本清单中记录：版本、下载来源、平台、架构和 SHA-256。至少包括 Vulkan SDK、OpenXR Loader、Filament SDK、TensorRT/CUDA、ROCm/MIOpen、DirectML/ONNX Runtime、MoltenVK 和 CoreML 相关构建要求。

升级 Filament 或 OpenXR Loader 时，必须同时运行三类验证：ABI/链接验证、最小场景渲染验证、真实 OpenXR 交换链验证。只通过资产加载测试不能视为升级完成。

### 2.3 Filament DLL Bridge定位

当前 `native/filament/bridge` 已验证Filament gltfio加载和动画接口，但仍依赖调用方提供OpenGL context，尚未实现目标OpenXR/Vulkan swapchain绑定。

新工程允许重新实现该Bridge，使其支持Filament Vulkan Backend和Python提供的Vulkan/OpenXR目标。Bridge是唯一自有原生边界，只管理Filament对象和渲染调用；Capture、Inference、Vulkan资源图、OpenXR生命周期和产品状态机必须保留在Python。Bridge使用窄C ABI并由Python通过`ctypes`或`cffi`加载。

### 2.4 推荐的 Fallback 策略

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
2. macOS 上 MoltenVK 初始化失败，或能力探测/基准测试确认性能不可接受。
3. 用户显式选择兼容模式。

后端选择发生在正式 Runtime 初始化前。`auto` 模式先探测 Vulkan，失败后返回带原因的 OpenGL 重启建议；启动器根据策略创建新的兼容会话。已经创建 Vulkan Device、Filament Engine 或 OpenXR Session 后不得在原会话中切换 API。

OpenGL Fallback 只保证基本场景、虚拟屏幕和必要的 OpenXR/窗口呈现。Vulkan Compute、异步 Compute Queue、高级 Glow/Reflection、Timeline Semaphore 和 external-memory 零拷贝不属于其必需能力。Fallback 必须消费推理/合成后端已经生成的 GPU Left/Right Eye 或 SBS；无法建立 GPU 路径时明确失败，不启用 CPU 实时像素链路。

---

## 3. 目标代码布局

新工程保持当前项目已经使用并验证过的`src/`顶层目录和模块名称，使日志路径、异常栈、测试定位和开发习惯保持一致。除Filament Bridge源码和预编译库外，所有项目代码均为Python；Vulkan Shader位于`src/`并由Python工具按需编译。新工程不得为了架构形式重新命名已稳定的模块，目标目录如下：

```text
Desktop2Stereo/
  src/
    main.py                      # Keep the current program entrypoint
    main.bat
    settings.yaml
    requirements.txt
    requirements-cuda.txt
    requirements-rocm7.txt
    requirements-mps.txt
    app_runtime/                 # Lifecycle, queues and runtime assembly
    capture/
      backends/
      dxgi/
    gui/                         # Flet GUI
    stereo_runtime/
      providers/
        nvidia/
        amd/
        apple/
        intel/
        cpu/
      model_impl/
      vulkan_graph.py            # Python Vulkan stereo/compute graph
      vulkan_resources.py
    viewer/
      viewer.py
      viewer_runtime.py
      vulkan_renderer.py         # Default desktop renderer
      opengl_renderer.py         # Isolated Fallback
    xr_viewer/
      openxr_runtime.py
      openxr_frame_pipeline.py
      core_openxr_vulkan.py      # Python OpenXR Vulkan session
      core_openxr_opengl.py      # Python OpenXR OpenGL Fallback
      controllers/
      environments/
      gltf/
      native/
        filament_bridge/         # The only project-owned native source
        windows-x64/
        linux-x64/
        macos-arm64/
    streaming/
    tools/
      model_tooling/
      benchmarks/
      probe.py
    utils/
    shaders/
      preprocess/
      depth/
      stereo/
      effects/
      output/
  tests/
    test_*.py
    fixtures/
```

`src/`是产品发布边界。发行流程整体复制或打包`src/`；Filament对应平台的预编译库必须已放入`src/xr_viewer/native/<platform>/`，不得在用户启动时编译。产品启动方式继续保持当前习惯，只要求建立Python环境并执行`src/main.py`或`src/main.bat`，不要求CMake或C++编译器。

新项目不建立`migration_reference`运行目录。可以整文件复用的实现直接复制到与当前项目相同的模块路径；尚未确认可用的代码留在旧项目或文档归档，不进入新项目`src/`。目录一致用于降低排查成本，不代表复制旧兼容分支和废弃实现。

### 3.1 Python模块与唯一原生目标

| 模块/目标 | 类型 | 职责 |
|----------|------|------|
| `app_runtime` | Python package | 状态机、配置快照、队列、生命周期和运行时装配 |
| `capture` | Python package | 复用WindowsCaptureCUDA/ROCm及跨平台Capture实现 |
| `stereo_runtime` | Python package | 推理Provider、模型、深度、立体合成和Vulkan Compute编排 |
| `viewer.vulkan_renderer` | Python module | 默认Vulkan窗口输出、资源和同步 |
| `viewer.opengl_renderer` | Python module | 隔离的OpenGL Fallback |
| `xr_viewer.core_openxr_vulkan` | Python module | OpenXR Vulkan生命周期、交换链和帧提交 |
| `xr_viewer.core_openxr_opengl` | Python module | OpenXR OpenGL Fallback |
| `xr_viewer.gltf` | Python package | 场景contract、资产状态和虚拟屏幕 |
| `streaming` | Python package | 编码和网络输出 |
| `gui` | Python package | Flet控制界面 |
| `tools` / `utils` | Python package | 模型工具、探测、benchmark和通用辅助 |
| `filament_bridge` | shared library | 唯一CMake目标；Filament Vulkan/OpenGL调用 |

厂商后端通过Python环境锁定文件和运行时capability probe控制。OpenGL Fallback由配置控制，推理Provider不得隐式改变图形后端选择。只有Filament Bridge使用CMake feature option。

### 3.2 依赖方向

```text
main.py
 `- app_runtime
     |- capture
     |- stereo_runtime.providers
     |- stereo_runtime.vulkan_graph
     |- viewer.vulkan_renderer
     |- xr_viewer.core_openxr_vulkan
     |- xr_viewer.gltf -> xr_viewer.native.filament_bridge
     |- streaming
     `- utils / telemetry

viewer.opengl_renderer and xr_viewer.core_openxr_opengl
  <- compatibility mode only
```

禁止依赖反转：`stereo_runtime`不得引用`xr_viewer`场景对象，`xr_viewer`不得控制capture session，`gui`不得取得底层图形资源所有权。Vulkan与OpenGL实现不得相互导入或共享资源对象；公共契约放在对应现有包的backend-neutral模块中。

### 3.3 目录兼容映射

| 当前项目路径 | 新项目路径 | 处理方式 |
|-------------|-----------|----------|
| `src/main.py` / `src/main.bat` | 原路径保留 | 重写装配逻辑，启动习惯不变 |
| `src/app_runtime/` | 原路径保留 | 保留状态、队列和生命周期职责 |
| `src/capture/` | 原路径保留 | 优先整文件迁入已验证Capture实现 |
| `src/gui/` | 原路径保留 | 保留Flet界面和配置职责 |
| `src/stereo_runtime/` | 原路径保留 | 保留Provider、模型和立体算法，新增Vulkan Graph模块 |
| `src/viewer/` | 原路径保留 | 删除旧图形实现，在原目录加入Vulkan主路径和OpenGL Fallback |
| `src/xr_viewer/` | 原路径保留 | 用Python重写Vulkan OpenXR路径，保留控制器、环境和帧调度职责 |
| `src/xr_viewer/native/` | 原路径保留 | 仅存放Filament Bridge源码和平台预编译库 |
| `src/streaming/` | 原路径保留 | 接入新的GPU输出契约 |
| `src/tools/` / `src/utils/` | 原路径保留 | 放置probe、模型工具、benchmark和公共辅助 |

目录兼容只保证定位和职责连续性，不保证旧模块内部API兼容。新项目禁止通过`sys.path`指向旧仓库，也禁止从旧仓库动态导入模块；所有正式依赖必须实际存在于新项目同名路径中。

---

## 4. 运行时总体结构

### 4.1 进程模型

正式数据面运行在单个Python进程中，一次进程会话只加载一个Graphics Backend。Flet GUI、Capture、Inference、Vulkan、OpenXR和Telemetry可以作为同一Python应用内的模块协作；也允许GUI通过`multiprocessing`启动独立Python Runtime进程以提高故障隔离，但进程之间只传控制消息，不传整帧像素。

```text
python src/main.py
   |- Flet Control UI / CLI
   |- Python Capture Adapter
   |- Python Inference Provider
   |- Python Vulkan Stereo Graph
   |- Python OpenXR Presenter
   |- Python Filament Bridge wrapper -> filament_bridge DLL
   `- Python Telemetry
```

采用独立GUI/Runtime进程时，Windows使用Named Pipe，Linux/macOS使用Unix Domain Socket；单进程模式可直接使用有界Python队列。两种模式使用同一带版本号的结构化消息，图像和GPU handle不得进入跨进程控制通道。

### 4.2 生命周期状态机

```text
Created
  -> Probing
  -> Initializing
  -> Ready
  -> Running
  -> Reconfiguring -> Running
  -> Recovering    -> Running
  -> Stopping
  -> Stopped

Any state -> Failed
```

状态转换由 `RuntimeController` 串行执行。模块不得自行把全局状态改为 Running 或 Stopped。每次转换生成带原因、时间戳和配置版本的事件。

### 4.3 初始化顺序

```text
1. Parse and validate RuntimeConfig.
2. Probe OpenXR runtime and platform capabilities.
3. Select Vulkan main path or OpenGL Fallback before creating graphics resources.
4. Vulkan path creates XrInstance requirements, Vulkan Instance/Device, queues and allocators.
5. OpenGL path creates the platform OpenGL context and reduced capability set.
6. Create the matching OpenXR Session/swapchains or non-XR presenter.
7. Initialize the selected Scene Renderer and scene resources.
8. Vulkan path creates Stereo/Effects Compute Graph resources; OpenGL path validates a GPU stereo input source.
9. Initialize Inference Adapter and backend-specific interop slots.
10. Initialize Capture Adapter.
11. Warm up the selected inference and graphics pipelines.
12. Enter Ready, then start capture and presentation.
```

推理 engine 构建或模型下载必须发生在 Capture 启动前。运行时不得边捕获边编译 TensorRT engine。

### 4.5 Graphics Backend 选择

```python
class GraphicsBackendKind(StrEnum):
    AUTO = "auto"
    VULKAN = "vulkan"
    OPENGL = "opengl"

@dataclass(frozen=True, slots=True)
class GraphicsBackendSelection:
    selected: GraphicsBackendKind
    reason: str
    capabilities: GraphicsCapabilities
```

`GraphicsBackendProbe` 在启动前生成选择结果。`Auto` 优先 Vulkan；只有符合第 2.4 节条件时才能选择 OpenGL。选择结果、触发原因和禁用功能写入启动报告和 GUI 状态。

Vulkan 与 OpenGL 实现共同遵循 `IGraphicsBackend` 生命周期接口，但资源类型保持后端私有。公共层只能传递 backend-neutral handle、Frame ID、尺寸、颜色空间和同步状态，不得用 `void*` 在两个 API 之间偷渡原生资源。

### 4.4 关闭顺序

停止接收新 capture frame 后，依次停止推理提交、等待有限数量在途 Frame Context、结束 OpenXR Session、释放场景、推理槽和 Vulkan 资源。正常关闭最多等待配置定义的超时；超时必须报告具体未完成 timeline 值。

---

## 5. 线程与队列设计

### 5.1 线程职责

| 线程 | 职责 | 禁止事项 |
|------|------|----------|
| Control Thread | 状态机、配置、命令和恢复协调 | 不提交逐帧 GPU 工作 |
| XR/Render Thread | OpenXR frame loop、视图定位、Graphics Queue 提交 | 不等待新 capture frame |
| Capture Thread | 平台捕获回调和 latest-frame 发布 | 不执行推理和场景渲染 |
| Inference Thread | 厂商后端 enqueue、external semaphore 协调 | 不操作 OpenXR swapchain |
| Asset Thread | 文件读取、GLB 解码准备、shader/cache IO | 不销毁在用 GPU 资源 |
| Telemetry Thread | 日志落盘、统计聚合、控制面事件 | 不读取 GPU 图像内容 |

Vulkan queue submit 由 `GpuScheduler` 统一序列化或按队列外部同步规则保护。不得让多个模块无约束地并发调用同一个 `VkQueue`。

### 5.2 有界队列

| 队列 | 容量 | 满载策略 |
|------|-----:|----------|
| Capture latest slot | 1 | 新帧覆盖旧帧 |
| Inference pending | 1 | 保留最新尚未开始的帧 |
| Frame Context pool | 3，配置范围 2 至 4 | 无可用上下文时跳过旧输入 |
| Effects pending | 1 | 新任务替换旧任务 |
| Control commands | 64 | 拒绝并报告过载，不丢停止命令 |
| Telemetry events | 固定 ring | 丢弃低级别采样，保留 warning/error |

任何实时队列都不得按运行时间增长。丢帧发生在未提交 GPU 工作之前，已提交工作不得通过破坏性方式取消。

### 5.3 帧调度

Render Thread 按 OpenXR 预测节奏持续更新头部和手柄姿态。没有新立体纹理时复用 last-good screen image，不能暂停 `xrWaitFrame/xrBeginFrame/xrEndFrame`。

Capture 与 Inference 以 latest-frame 推进；新 screen image 完成后以原子方式更新 `latest_screen_slot`。场景渲染只等待所选 screen slot 的完成信号，不等待下一张输入。

Effects Graph 消费最近完成的 screen slot，并发布 `latest_effect_slot`。Graphics Queue 从不等待指定 frame ID 的光效结果。

---

## 6. 核心数据契约

### 6.1 基础类型

```python
FrameId = int

@dataclass(frozen=True, slots=True)
class Extent2D:
    width: int
    height: int

@dataclass(frozen=True, slots=True)
class SyncPoint:
    semaphore: object
    value: int
    stage_mask: int

@dataclass(frozen=True, slots=True)
class GpuImageView:
    image: object
    view: object
    format: int
    extent: Extent2D
    layout: int
    queue_family: int
```

`GpuImageView`是非拥有视图。拥有资源由`GpuImage`显式管理，并提供幂等`close()`和context manager；不得依赖Python垃圾回收时机释放Vulkan对象。

### 6.2 CaptureFrame

```python
@dataclass(frozen=True, slots=True)
class CaptureFrame:
    id: FrameId
    timestamp_ns: int
    capture_size: Extent2D
    pixel_format: PixelFormat
    color_space: ColorSpace
    image: ExternalImageHandle
    ready: SyncPoint
```

`ExternalImageHandle` 是 tagged union，平台实现可以承载 Vulkan image、Win32 handle、DMA-BUF/FD、IOSurface 或受控 CPU 测试帧。实时模式下不接受普通 CPU pointer。

### 6.3 InferenceResult

```python
@dataclass(frozen=True, slots=True)
class InferenceResult:
    frame_id: FrameId
    relative_depth: object
    metadata: DepthMetadata
    ready: SyncPoint
    stale: bool
```

`DepthMetadata` 必须包含模型 ID、backend、precision、near/far direction、normalization、模型输入尺寸和 GPU timing。下游不得猜测深度方向。

### 6.4 StereoFrame

```python
@dataclass(frozen=True, slots=True)
class StereoFrame:
    frame_id: FrameId
    left_eye: GpuImageView
    right_eye: GpuImageView
    packed_sbs: GpuImageView | None
    ready: SyncPoint
    config_version: int
    color_space: str = "srgb"
    image_origin: str = "top_left"
```

Left/Right Eye 是标准输出；只有虚拟屏幕材质或输出目标需要时才生成 `packed_sbs`。未生成时使用空 view，不分配无意义图像。输出帧契约明确声明 `color_space=srgb` 和 `image_origin=top_left`；任何 OpenXR、Preview 或编码后端需要改变目标坐标原点时，必须在边界处显式适配，不能修改源图像语义。

### 6.5 FrameContext

每个 Frame Context 独占 command pool、command buffer、descriptor arena、timestamp query 范围和中间图像索引。Frame Context 只能在其最终 timeline 值完成后复用。

---

## 7. Vulkan 基础层

### 7.1 VulkanContext

`VulkanContext` 负责 Instance、Device、Physical Device、queue family、allocator、pipeline cache 和 debug messenger。它不管理 OpenXR frame loop，也不加载 glTF。

必须提供：

```python
class VulkanContext:
    @classmethod
    def create(cls, requirements: VulkanRequirements) -> "VulkanContext": ...
    def close(self) -> None: ...
    @property
    def instance(self) -> object: ...
    @property
    def physical_device(self) -> object: ...
    @property
    def device(self) -> object: ...
    @property
    def graphics_queue(self) -> QueueHandle: ...
    @property
    def compute_queue(self) -> QueueHandle: ...
    @property
    def transfer_queue(self) -> QueueHandle: ...
```

`VulkanRequirements` 由 OpenXR、Filament、Compute Graph 和推理互操作能力合并生成。缺失必需扩展时在创建 Device 前失败。

Vulkan 版本和设备 Feature 规则：

- OpenXR Vulkan 初始化默认请求 Vulkan 1.4；实际版本必须限制在 `xrGetVulkanGraphicsRequirements2KHR` 返回的最小/最大范围内，主路径最低保证 Vulkan 1.2。
- Vulkan 1.4 只是默认请求上限，不要求所有 Runtime、驱动或 GPU 都实际创建 1.4 Device；Runtime 协商到 1.2 或 1.3 时，只要能力探测通过即可继续。
- 创建设备前必须使用 `vkGetPhysicalDeviceFeatures2` 查询 `timelineSemaphore`。
- 当 Timeline Semaphore 可用时，必须将 `VkPhysicalDeviceTimelineSemaphoreFeatures` 追加到 `VkDeviceCreateInfo.pNext`，并将 `timelineSemaphore` 设置为 `VK_TRUE`。请求 Vulkan 版本不会自动启用该 Feature。
- Timeline Semaphore 不支持或启用失败时，Vulkan OpenXR 路径必须在创建正式 Device/Session 前失败并给出原因；由启动器按策略受控重启进入 OpenGL Fallback。

### 7.2 资源分配

- 所有长期图像通过统一 `GpuAllocator` 创建。
- transient image 按尺寸、格式和 usage 建立可复用池。
- external-memory image 单独分配，不与普通 transient memory alias。
- OpenXR swapchain image 不绑定应用内存，不进入 allocator 销毁流程。
- 内存预算来自 `VK_EXT_memory_budget` 时必须采集并上报。

### 7.3 图像状态追踪

`ImageStateTracker` 记录每个受管图像的 layout、stage、access 和 queue family。Compute Graph 编译阶段产生 barrier plan，执行阶段使用 Synchronization2 提交。

不得使用 `VK_IMAGE_LAYOUT_GENERAL` 规避全部状态管理。External-memory 交界可按后端要求使用 GENERAL，但进入场景采样前必须转换为正确只读布局。

### 7.4 Descriptor

- Descriptor layout 在 shader 构建时反射并生成稳定 binding metadata。
- 静态资源使用 persistent set；帧资源使用 Frame Context arena。
- 不更新仍可能被 GPU 读取的 descriptor。
- Descriptor Indexing 只用于资源数组，不允许动态越界。

### 7.5 Pipeline 与 Shader

Shader 源码进入版本控制，SPIR-V 由构建系统生成。每个 shader 必须有：入口、workgroup size、descriptor binding、push constant 大小和精度要求的机器可读 manifest。

Pipeline Cache key 至少包含 GPU UUID、驱动版本、Filament 版本、shader hash 和 build type。缓存无效时重建，不把 pipeline 创建放入稳态帧循环。

### 7.6 同步规则

内部依赖使用 Timeline Semaphore 和 `vkQueueSubmit2`。正常帧循环禁止调用：

```text
vkDeviceWaitIdle
vkQueueWaitIdle
CPU polling loop on fence status
implicit queue ownership assumptions
```

只有初始化失败清理、Session 销毁和进程关闭可执行受控全局等待。

Timeline 值必须由统一的 `FrameContext` 同步协议管理。NVIDIA、AMD、Intel 和 MoltenVK 路径均不得假设该 Feature 默认开启；能力探测、Device 创建 pNext 链和启动诊断必须记录实际支持与启用状态。

---

## 8. Capture 工程设计

### 8.1 接口

```python
class CaptureAdapter(Protocol):
    def start(self, config: CaptureConfig, sink: CaptureSink) -> None: ...
    def stop(self) -> None: ...
    def capabilities(self) -> CaptureCapabilities: ...
```

`CaptureSink::publish()` 只发布句柄和 metadata，不进行图像转换。Capture Adapter 对平台捕获对象负责，Gpu Importer 对 Vulkan 导入负责。

### 8.2 平台路径

| 平台 | 首选捕获 | Vulkan 接入目标 |
|------|----------|-----------------|
| Windows | Windows Graphics Capture / DXGI | 共享 handle 导入或 GPU copy 到 Vulkan image |
| Linux | PipeWire/DMABUF 或 DRM 路径 | DMA-BUF/FD 导入 Vulkan |
| macOS | ScreenCaptureKit | IOSurface/Metal 纹理桥接到 MoltenVK 路径 |

平台路径必须在运行报告中标明`native_import`、`gpu_copy`或`cpu_test_input`。Windows首先复用现有`WindowsCaptureCUDA`、`WindowsCaptureROCm`和对应测试，不因Vulkan迁移重写捕获算法。正式实时运行不得把CPU路径命名为零拷贝。

### 8.3 尺寸与颜色

Capture Adapter 报告真实 `capture_size`、pixel format、transfer function、primaries 和 range。RGB preprocess Compute Pass 负责统一到线性工作色彩空间和 `render_size`。

窗口尺寸或 HDR 状态改变时发布 format-change event。资源重建在 Frame Boundary 进行，capture callback 不直接重建 Vulkan 资源。

---

## 9. 推理适配器设计

### 9.1 接口

```python
class InferenceProvider(Protocol):
    def initialize(
        self, config: InferenceConfig, interop: InteropContext
    ) -> InferenceCapabilities: ...
    def submit(self, submission: InferenceSubmission) -> SyncPoint: ...
    def reset(self) -> None: ...
```

`InferenceSubmission`引用预分配input/output slot。现有TensorRT、PyTorch CUDA/ROCm、MIGraphX和其他Python Provider应保留优化实现，只调整统一接口和Vulkan输出契约；不得每帧创建engine、分配大块device memory或重新注册external resource。

### 9.2 Slot 注册

初始化阶段由 Vulkan 创建 N 组可导出资源，Inference Adapter 一次性导入并保存后端对象。每帧只选择 slot、等待 timeline、enqueue 和 signal。

```text
Vulkan creates exportable input/depth slots
-> exports OS memory handles
-> inference adapter imports handles once
-> frame loop reuses registered slots
```

Win32/FD handle 在成功导入后按对应 API 所有权规则关闭。句柄泄漏测试必须覆盖重复初始化和 Session 恢复。

### 9.3 NVIDIA

NVIDIA 后端使用 TensorRT + CUDA。优先由 CUDA 直接访问 Vulkan 导出的 input/output memory，并使用 external semaphore 同步。

验收要求：主路径无 host memcpy、无 `.cpu()`/NumPy 往返、无每帧 external-memory import。TensorRT enqueue 时间和 GPU 完成时间分别记录。

### 9.4 AMD

Windows/Linux 优先 ROCm/HIP + MIOpen。可靠 external memory 可用时使用零拷贝；否则允许一次明确的 GPU 内拷贝。

Windows ROCm 不可用时，DirectML Adapter 可作为独立推理后端。D3D12 resource 仅存在于该 Adapter 内部，导出到 Vulkan 后不扩散为 D3D 渲染架构。

### 9.5 Apple Silicon

CoreML/MPSGraph Adapter 管理 MTLTexture/IOSurface 和模型执行。MoltenVK 与 Metal 的资源连接必须通过原型验证确定；不能假设 MoltenVK 自动消除所有拷贝。

### 9.6 失败策略

Adapter 初始化失败时终止本次启动并返回 capability report。运行中单帧失败可丢弃；连续失败达到阈值后进入 Failed 或受控重建。不得静默切换 CPU 推理。

---

## 10. Stereo Compute Graph

### 10.1 Graph 结构

```text
Capture Import
-> RGB Normalize/Resize
-> Inference Bridge
-> Depth Normalize
-> Edge-aware Depth Filter
-> Parallax Field
-> Left/Right Warp
-> Occlusion Mask
-> Directional Hole Fill
-> Temporal Stabilization
-> Optional SBS Pack
```

Graph 在 `render_size` 和质量配置确定后编译。Pass、资源和 barrier 在编译时固定，执行时只更新 descriptor、push constant 和 slot index。

### 10.2 Pass 接口

```python
class ComputePass(Protocol):
    def declare(self) -> PassDeclaration: ...
    def create_pipeline(self, shaders: ShaderLibrary) -> None: ...
    def record(
        self,
        command_buffer: object,
        resources: PassResources,
        parameters: FrameParameters,
    ) -> None: ...
```

Pass 不得自行提交 queue，不得持有 OpenXR swapchain，不得执行文件 IO。

### 10.3 参数上传

每帧参数写入 host-visible ring buffer 或 push constant。参数快照带 `config_version`，一帧所有 Pass 必须使用同一版本。

视差核心语义保持：

```text
disparity_px = depth_response(relative_depth, convergence)
             * max_disparity_px
             * depth_strength
```

IPD、Stereo Scale、Max Shift Ratio 不进入 normalized-depth shader contract。

### 10.4 Temporal 资源

Temporal history 每个 eye 至少包含 color、confidence 和有效性标记。资源按 ping-pong 方式复用。尺寸、捕获源、模型、深度方向或关键曲线改变时，`TemporalController` 在下一帧前清除有效性。

### 10.5 Shader 测试

每个 Compute Pass 必须有 CPU reference 或固定 golden fixture。GPU 测试比较误差、边界像素、NaN 输入、极端尺寸和非 16:9 输入。

---

## 11. Filament 场景系统

### 11.1 Python SceneRenderer接口

```python
class FilamentSceneRenderer:
    def initialize(self, vulkan: VulkanContext, config: SceneConfig) -> None: ...
    def load_glb(self, path: Path) -> SceneHandle: ...
    def activate_scene(self, handle: SceneHandle) -> None: ...
    def update(self, state: SceneFrameState) -> None: ...
    def render(self, target: SceneRenderTarget, eye: EyeView) -> None: ...
```

`FilamentSceneRenderer`是Python封装，内部通过`ctypes`/`cffi`调用Filament DLL Bridge。DLL管理Filament Engine、Renderer、Scene、View、Camera、AssetLoader、ResourceLoader和Animator；Python只持有不透明整数handle，不直接持有Filament entity或C++对象地址。

### 11.2 Graphics Backend

Filament DLL Bridge默认创建Filament Vulkan Backend，并实现经过版本控制的Vulkan Render Target接入。OpenGL Fallback可由同一Bridge创建独立Filament OpenGL Backend或使用Python兼容Renderer，但不得复用Vulkan会话资源。旧Bridge的glTF加载和动画经验可以复用，旧OpenGL-only ABI必须由新版窄C ABI取代。

OpenXR swapchain 的接入必须通过经过验证的 Filament external render target/Vulkan platform integration 实现。若特定 Filament 版本不能安全直接绑定 acquired `VkImage`，允许在 Vulkan 内渲染到应用 color image 后执行一次 Vulkan copy/blit 到 swapchain；不得改用 D3D11/GL 桥接。

OpenGL Fallback 直接渲染到 OpenGL OpenXR swapchain 或 OpenGL window framebuffer，不经过 Vulkan、D3D11 或 WGL 跨 API 桥接。Windows/Linux 目标 OpenGL 4.3 及以上；macOS OpenGL 4.1 仅提供基本网格、材质、虚拟屏幕和 UI，不启用依赖 Compute Shader 的效果。

### 11.2.1 Bridge 接口边界与完整性清单

Bridge 采用窄 C ABI，按产品功能补充接口，不做 Filament 全量 API 的 Python 化。C++ 内部管理 Engine、Renderer、Scene、View、Camera、AssetLoader、ResourceLoader、Animator、Material 和 Texture 等对象；Python 通过 `ctypes` 使用不透明 handle、标量、矩阵、资源字节和结构化状态。

Bridge 接口必须覆盖以下功能域：

1. ABI/Filament 版本、平台能力、错误码和最近错误。
2. Vulkan Bridge 与 Preview 的创建、初始化、销毁和资源释放。
3. Python/OpenXR Vulkan 对象借用、swapchain image 注册、acquired image 绑定、帧开始/结束和 swapchain 重建。
4. GLB 加载、纹理/PBR/透明材质资源准备、场景切换、卸载和 last-good scene 保留。
5. 场景对象的可见性、变换、虚拟屏幕、手柄和场景根节点状态更新。
6. 预览相机、左右眼相机、look-at、视图矩阵、投影/frustum、near/far 和 viewport。
7. Preview window、Vulkan Render Target、OpenXR Projection Layer 目标的渲染提交。
8. 动画枚举、名称/时长、选择、播放、暂停、循环和由 Python 驱动的动画时间应用。
9. 场景主体亮度/曝光、天空盒亮度、方向光、填充光及其颜色和方向的独立控制。
10. 材质参数、亮度、对比度、饱和度、Gamma、色温和色调控制。
11. Glow、星光闪烁、平均色、墙面反射及其他后处理/特效的资源设置、启停、参数更新和时间推进。
12. Preview resize、surface/window 变化、渲染目标尺寸变化和资源重建。
13. 加载、上传、渲染、同步和 GPU 资源统计，以及 capability report 所需的诊断数据。

每个功能域必须完成以下闭环才算实现：

```text
功能需求确认
    -> filament_bridge.h C ABI 声明
    -> filament_bridge.cpp 实现
    -> Python ctypes wrapper
    -> Windows/Linux/macOS CI 编译
    -> 单元/集成/头显或预览运行验证
    -> 接口清单标记完成
```

接口清单以本节为功能基线，另在实现任务中记录“已实现、待实现、暂不需要、验证证据”。新增能力不得只修改 C++ 而遗漏 Python wrapper、错误处理、CI 或测试；也不得为了所谓完整性导出 Filament 未使用的模板、内部类型、Entity 管理器或逐资源底层 API。

### 11.3 资产加载

- GLB 文件 IO 和解析准备在 Asset Thread 完成。
- GPU 上传由 Render Thread 在受控阶段执行。
- 新场景全部资源 ready 后，在帧边界原子切换。
- 加载失败保留当前有效场景，首次加载失败则启动失败。
- 资产路径必须位于允许的资源根目录，禁止从配置执行任意脚本。

### 11.4 虚拟屏幕

虚拟屏幕材质采样 Left/Right Eye 或 SBS 对应区域。每眼 View 必须选择正确 eye texture/UV，不通过 cross-eyed 旧兼容参数猜测顺序。

屏幕变换、距离、曲率和可见性由 `SceneFrameState` 提供。交互状态与渲染资源分离，手柄拖动只更新下一帧 transform snapshot。

### 11.5 相机和裁剪面

每帧使用 OpenXR `pose` 和 `fov` 更新 Filament Camera。Near/Far Clip 来自场景配置，并满足 `far > near`。大型 GLB 场景必须通过实际包围盒和头显视觉验证，不能依赖固定 100 m 默认值。

---

## 12. 异步光效系统

Effects Graph 使用 Vulkan Compute 处理最近完成的 screen image：

```text
Downsample -> Horizontal Blur -> Vertical Blur -> Glow
           -> Color Reduction -> Reflection Mask Composite
```

预计算掩码在场景或尺寸变化时更新，不得每帧从 CPU 生成。颜色归约结果若仅供 GPU 材质使用，不回读 CPU。

效果 slot 发布采用 timeline + 原子索引。SceneRenderer 使用最新 ready slot；效果缺帧或失败时继续使用上一结果。Graphics Queue 不等待与当前 screen frame 同 ID 的效果任务。

---

## 13. OpenXR Presenter

### 13.1 职责

`OpenXrPresenter` 负责 XrInstance、System、Session、Space、Action、Swapchain、事件和 Composition Layer。它不执行深度推理或立体算法。

### 13.2 Vulkan Session 创建

必须通过 `xrGetVulkanGraphicsRequirements2KHR`、`xrCreateVulkanInstanceKHR`/等效规范路径和 `XrGraphicsBindingVulkan2KHR` 建立 Session。Physical Device 选择服从 OpenXR Runtime 要求。

禁止创建 D3D11 Session 后导入 Vulkan 结果。

### 13.3 Frame Loop

```text
xrPollEvent
-> xrWaitFrame
-> xrBeginFrame
-> xrLocateViews / xrSyncActions
-> acquire and wait swapchain images
-> update scene snapshots
-> record and submit scene render
-> release swapchain images
-> xrEndFrame
```

即使没有新 capture frame，也继续更新 pose、controller 和场景，并复用 last-good screen image。只有 Runtime 指示 `shouldRender=false` 时跳过图像渲染，但仍按 OpenXR 规则结束本帧。

### 13.4 Swapchain

- 格式从 Runtime 支持列表按色彩语义选择。
- 每眼 swapchain image 建立非拥有 `GpuImageView`。
- acquire/wait/release 必须成对，异常路径也要归还已 acquire 图像。
- swapchain 重建前等待引用它的 Frame Context 完成。
- 应用不得销毁 OpenXR 创建的 `VkImage`。

### 13.5 Layers

主房间、虚拟屏幕、手柄和默认 Glow 使用 Projection Layer。文字面板、虚拟键盘等独立 UI 可使用 Quad Layer。Layer 数量和顺序由单一 `CompositionBuilder` 生成，模块不能各自调用 `xrEndFrame`。

---

## 14. 非 XR 输出

### 14.1 Vulkan Window

桌面预览默认使用独立 Vulkan surface/swapchain。兼容模式创建独立 OpenGL window/context；它不复用 OpenXR frame loop，也不与 Vulkan surface 共同存在。

### 14.2 Headless/Encoder

Headless 输出将 Left/Right/SBS Vulkan image 提供给编码适配器。支持 external-memory 的编码器直接导入；否则允许明确的 GPU 格式转换和 GPU copy。

文件截图、离线测试可以回读 CPU，但必须使用独立命令和日志标签，不能进入实时执行路径。

---

## 15. 配置系统

### 15.1 RuntimeConfig

核心运行时只接受规范化配置，不解析 GUI 文案或旧 `settings.yaml` 字段：

```python
@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    schema_version: int
    graphics: GraphicsConfig
    capture: CaptureConfig
    inference: InferenceConfig
    stereo: StereoConfig
    scene: SceneConfig
    openxr: OpenXrConfig
    output: OutputConfig
    telemetry: TelemetryConfig
```

配置文件采用明确 schema version。GUI 负责把用户选择转换成新 schema；一次性迁移工具负责读取旧 YAML 并输出新配置，核心运行时不包含旧字段 alias。

### 15.2 快照与热更新

`RuntimeConfigSnapshot` 不可变并带单调 `version`。Control Thread 校验差异后生成：

| 变更类别 | 示例 | 动作 |
|----------|------|------|
| Uniform Update | depth strength、convergence、effect strength | 下一帧生效 |
| Temporal Reset | depth response、temporal enabled | 下一帧清 history |
| Graph Rebuild | render scale、hole-fill quality tier | drain 有关 Frame Context 并重建图 |
| Adapter Rebuild | capture source、model、inference backend | 停止对应 adapter 后重建 |
| Session Rebuild | XR swapchain format、关键 XR 能力 | 重建 Session/Swapchain |
| Process Restart | Graphics Backend、Vulkan device、核心 feature set | 返回 restart-required；新进程重新 probe |

一帧只能使用一个完整配置版本。模块不得在帧中途读取可变全局配置。

### 15.3 配置验证

启动前验证平台、GPU、模型、格式、尺寸、场景路径和输出组合。未知字段默认报错；可选扩展字段必须位于命名 namespace，避免拼写错误被忽略。

---

## 16. 控制面与 GUI 边界

GUI 可继续使用 Flet/Python，但其职责收敛为：设备探测结果展示、配置编辑、启动停止、状态和日志展示、错误报告导出。

控制协议至少支持：

```text
hello(protocol_version)
probe()
start(config)
stop(reason)
apply_config(snapshot)
load_scene(path)
get_status()
subscribe_events()
```

所有命令返回 request ID 和结构化结果。停止命令具有最高优先级。GUI 进程退出时，Runtime 根据启动模式决定自动停止或继续 headless，行为必须显式配置。

独立进程模式下GUI不读取Runtime内部对象，也不解析自由格式日志来判断状态；单进程模式通过稳定的Python控制接口访问状态。两种模式都不得取得Vulkan、OpenXR或Filament原生对象的所有权。

---

## 17. 错误模型与恢复

### 17.1 Status

```python
class ErrorDomain(StrEnum):
    CONFIG = "config"
    PLATFORM = "platform"
    CAPTURE = "capture"
    GRAPHICS = "graphics"
    VULKAN = "vulkan"
    OPENGL = "opengl"
    INFERENCE = "inference"
    STEREO = "stereo"
    FILAMENT = "filament"
    OPENXR = "openxr"
    OUTPUT = "output"
    INTERNAL = "internal"

@dataclass(frozen=True, slots=True)
class Status:
    domain: ErrorDomain
    code: int
    severity: Severity
    message: str
    context: str
    retryable: bool
```

模块边界捕获并转换异常为`Status`或领域异常；主循环不得泄漏未处理异常。GPU资源必须显式`close()`，清理方法应幂等且不得抛出未处理异常。

### 17.2 恢复策略

| 故障 | 策略 |
|------|------|
| Capture source closed | 进入等待或停止，取决于配置 |
| 单帧推理失败 | 丢帧并计数，连续失败后重建 Adapter |
| External import 不可用 | 平台允许时选择一次 GPU copy，启动报告固定记录 |
| OpenXR Session loss | 停止 XR 提交，重建 Session 和 swapchain |
| Scene asset reload 失败 | 保留 last-good scene |
| Shader/Pipeline 失败 | 启动失败或保持旧 pipeline，不带病切换 |
| Vulkan Device lost | 收集 fault 信息并终止 Device，不原地继续提交 |
| Vulkan 启动探测失败 | 返回 OpenGL Fallback 建议和原因，由启动器受控重启 |
| OpenGL Fallback 初始化失败 | 输出缺失版本、扩展或 OpenXR binding，并终止启动 |

任何降级路径必须在初始化能力选择时确定。运行中不得从 Vulkan 静默切换到 OpenGL；OpenGL 只能通过受控重启进入。不得切换到旧图形桥接 API 或 CPU 实时渲染。

---

## 18. 日志、指标与诊断

### 18.1 结构化日志

日志事件至少包含 timestamp、severity、domain、event name、thread、frame ID、config version 和 key/value context。控制台可渲染为文本，文件保存 JSON Lines。

密钥、访问令牌、完整用户文件内容和系统私密路径不得写入报告。配置快照输出前执行字段级脱敏。

### 18.2 GPU Timing

Vulkan 阶段使用 timestamp query；CUDA/HIP 使用后端 event；OpenXR wait/submit 使用 CPU monotonic clock。以下指标必须分开：

```text
capture_fps
inference_fps
stereo_fps
scene_submit_fps
present_fps
capture_to_present_ms
gpu_preprocess_ms
gpu_inference_ms
gpu_stereo_ms
gpu_scene_ms
gpu_effects_ms
xr_wait_ms
xr_submit_ms
capture_overwrite_count
frames_in_flight
```

不得把 `xrWaitFrame` 或 present interval 计入 Stereo Compute GPU 耗时，也不得用 CPU enqueue 时长代表 GPU 完成时长。

### 18.3 Diagnostic Bundle

错误报告包含版本、commit、平台、GPU、驱动、Vulkan/OpenXR capability、配置摘要、最近日志、GPU timing、validation 摘要和 device fault 信息。默认只生成本地文件，上传必须经过用户确认。

---

## 19. 测试设计

### 19.1 单元测试

- 配置 schema、差异分类和状态机转换。
- Render Size、Parallax Budget 和视差参数。
- Frame slot、latest-frame 覆盖和 timeline 递增。
- Resource ownership、RAII move 和异常清理。
- glTF asset handle、scene switch 和 animation time。
- OpenXR event 到状态转换的纯逻辑部分。

### 19.2 Vulkan GPU 测试

- Headless Vulkan device 上执行每个 Compute Pass。
- Validation Layer 下检查 layout、access、queue ownership 和 descriptor lifetime。
- 比较 CPU reference/golden fixture。
- 覆盖 720p、1080p、1440p、4K、超宽和竖屏。
- 覆盖 dedicated compute queue 与 single universal queue。
- 运行资源重建、尺寸切换和 pipeline cache 冷热启动。

### 19.3 推理互操作测试

- external memory 一次导入、多帧复用。
- external semaphore wait/signal 次序。
- 重复初始化、失败清理和 handle 泄漏。
- 输入输出无 CPU readback 的工具级证明。
- GPU copy fallback 的次数和方向符合平台规格。

### 19.4 Filament 测试

- 静态 GLB、动画、PBR texture、透明材质和大场景。
- Scene load failure 保留 last-good scene。
- 每眼 camera pose/FOV、near/far 和 screen UV 正确。
- Vulkan render target 输出非空且颜色空间正确。

### 19.5 OpenXR 测试

- Mock/头less 可覆盖的 lifecycle 与 swapchain 顺序测试。
- Windows 至少两个真实 OpenXR Runtime 实机测试。
- Session loss、headset disconnect/reconnect 和 swapchain rebuild。
- 无新 screen frame 时 pose/controller 仍按显示帧率更新。
- Projection/Quad Layer 数量、顺序和 image release 正确。

### 19.6 OpenGL Fallback 测试

- Vulkan 不可用、用户显式兼容模式和 macOS MoltenVK 失败三类触发条件可复现。
- Windows/Linux 原生 OpenGL 和 macOS OpenGL 4.1 capability report 正确。
- 同一会话不加载 Vulkan Device、D3D11、WGL/CUDA-GL bridge 或旧 viewer。
- 基本场景、虚拟屏幕、Left/Right Eye 或 SBS 呈现正确。
- 不支持的 Compute/Glow/Reflection 功能在 UI 和日志中明确禁用。
- GPU stereo input 不可用时明确失败，不发生 CPU 实时像素回读。

### 19.7 长稳测试

正式候选版本至少运行：

| 场景 | 时长 | 通过标准 |
|------|-----:|----------|
| 1080p/90 Hz XR | 2 小时 | 无 validation error、无队列增长 |
| 4K scale XR | 1 小时 | 内存稳定、P95 延迟达标 |
| 连续 resize/source switch | 500 次 | 无泄漏、无 device lost |
| Scene hot swap | 200 次 | 无悬空资源、last-good 生效 |
| Session reconnect | 100 次 | Session 可恢复或明确失败 |

---

## 20. CI 与构建交付

### 20.1 CI Pipeline

```text
format/lint
-> Python import/compile checks
-> pytest unit tests
-> shader compile + reflection validation
-> Filament Bridge build matrix when bridge source changes
-> headless Vulkan GPU tests where runner supports
-> package
-> dependency manifest and SHA verification
```

真实 GPU/OpenXR 测试由专用硬件 runner 执行，不用普通 CI 成功替代实机验收。

### 20.1.1 Filament Bridge 三平台构建策略

Filament Bridge 的正式构建由 GitHub Actions 负责。三平台必须使用对应平台 runner 编译和链接 Filament 1.74 及其运行库，构建结果通过 Actions Artifact 或 GitHub Release 交付给 Python 运行时。

```text
native/filament/bridge source change
    -> Windows runner -> filament_bridge.dll
    -> Linux runner   -> libfilament_bridge.so
    -> macOS runner   -> libfilament_bridge.dylib
    -> ABI/link/runtime checks
    -> upload artifact or release asset
    -> download into src/xr_viewer/native/<platform>/
```

必须遵守以下规则：

- Bridge 源码、C ABI、Filament 版本或 CMake/平台构建配置变更时，CI 必须执行三平台构建矩阵；只修改 Python、GUI、配置或资源时不重新编译 Bridge。
- CI 产物必须记录平台、架构、Filament 版本、Git commit、构建类型和 SHA-256，并随产物提供依赖清单。
- 发布包使用 CI 产物，不把本地 `build/`、临时 SDK 解压目录或未验证的本地 DLL/SO/DYLIB 作为正式交付物。
- 本地开发默认下载与当前平台匹配的 CI 产物。只有在 CI 故障、调试原生崩溃或验证未提交的 Bridge 修改时，才允许本地编译。
- 本地编译结果只能用于诊断和开发验证；合并和发布前仍必须通过三平台 CI，且本地编译不得替代 Linux/macOS 平台验证。
- Python 通过 `ctypes` 加载 `src/xr_viewer/native/` 下对应平台的 Bridge。加载前检查文件存在性、架构、依赖库可解析性和 ABI 版本；检查失败必须在 capability report 中明确报告。
- Bridge 二进制与 Filament 运行库必须作为同一版本构建包管理，禁止混用不同 Filament 版本的库文件。

### 20.2 构建产物

发布包至少包含：

```text
Python source packages and entrypoints
platform Python environment lock/requirements
Filament Bridge and Filament runtime libraries
OpenGL compatibility backend when enabled
OpenXR loader where platform packaging requires
enabled Python inference packages/providers
compiled SPIR-V and shader manifest
default assets and config schema
dependency/version manifest
licenses
```

不得包含旧OpenGL/D3D11 viewer DLL、Panda3D runtime或仅供开发的build directory。发布包只允许包含新版Filament DLL Bridge这一项自有原生组件；启用Fallback时包含Python `viewer.opengl_renderer`和`xr_viewer.core_openxr_opengl`模块及其明确依赖。

推荐的运行时目录：

```text
src/xr_viewer/native/
├── windows/filament_bridge.dll
├── linux/libfilament_bridge.so
└── macos/libfilament_bridge.dylib
```

上述目录中的文件来自对应 GitHub Actions 构建产物或 Release 资产。源码修改后先提交并等待 CI 生成新产物，再更新本地运行目录；不要求每次 Python 代码修改都重新编译 Filament。

### 20.3 启动探测

`python src/tools/probe.py`必须可独立输出JSON capability report，包括GPU、Vulkan、OpenGL、MoltenVK、external memory、OpenXR Graphics Binding、swapchain format、Filament Bridge初始化和推理Provider可用性。GUI只根据该报告启用可选项。

---

## 21. 性能与资源预算

### 21.1 关键路径预算

90 Hz 目标遵循 `docs/01`：应用 GPU 关键路径不高于 10 ms，P95/P99 单独报告。工程层必须能分解 preprocess、inference、stereo、scene 和 submit。

### 21.2 内存预算

资源预算按格式和 Frame Context 数量计算并在启动时输出。4K 模式必须覆盖 capture、linear RGB、depth、disparity、mask、双眼、temporal、effects 和 swapchain 的峰值。

当预计使用量超过可用 device-local budget 的安全比例时，启动失败或要求降低 render tier；不得等到分配失败后静默降质。

### 21.3 CPU 预算

稳态帧中 CPU 只处理命令构建、状态快照和提交。禁止逐帧像素循环、逐帧大对象分配和自由格式日志拼接。Telemetry 采样使用预分配 ring。

---

## 22. 安全与健壮性

- GLB、配置和 shader cache 输入必须验证尺寸上限和路径范围。
- 不从资产或配置执行脚本、命令或动态代码。
- 本地控制通道限制为当前用户，并验证 protocol/schema version。
- 外部内存句柄只在受信进程边界内传递，并严格遵循所有权规则。
- 下载模型和 SDK 必须验证摘要；运行时不自动执行未验证二进制。
- Validation/diagnostic 信息不得泄露密钥或完整私密路径。

---

## 23. 实施顺序

### Phase 1：工程骨架与 Vulkan/OpenXR

1. 建立以`src/`为产品发布边界的Python package和平台依赖锁定文件。
2. 在`src/tools/probe.py`、`src/viewer/vulkan_renderer.py`和`src/xr_viewer/core_openxr_vulkan.py`中实现能力探测、`VulkanContext`、`GpuAllocator`和`GpuScheduler`。
3. 使用Python OpenXR代码建立Vulkan Session和每眼清屏闭环。
4. 接入Validation、timestamp、结构化日志和显式资源清理测试。

完成标准：真实头显稳定显示 Vulkan Projection Layer，运行 30 分钟无 validation error。

### Phase 2：Filament Vulkan 场景

1. 重新实现唯一的Filament DLL Bridge窄C ABI，保留已验证的GLB/Animator能力。
2. 在Python `FilamentSceneRenderer`中实现Bridge加载、handle生命周期、每眼Camera、虚拟屏幕和手柄状态更新。
3. 验证Vulkan Backend直接swapchain target；不支持时由Bridge执行一次Vulkan内copy。

完成标准：正式房间/手柄在两个OpenXR Runtime正确显示；项目中除新版Filament DLL Bridge外没有其他自有原生运行代码。

### Phase 2B：OpenGL Fallback

1. 实现Python `GraphicsBackend`协议以及`viewer.opengl_renderer`、`xr_viewer.core_openxr_opengl`隔离模块。
2. 使用Python OpenGL API实现Windows/Linux与macOS OpenGL 4.1基本渲染。
3. 使用Python OpenXR代码实现OpenGL Graphics Binding、窗口输出和受限功能报告。
4. 验证受控重启、GPU stereo input 和禁止 CPU 实时回读。

完成标准：三类 Fallback 触发条件通过，兼容会话不加载 Vulkan/D3D11/WGL 旧桥接，基本场景和虚拟屏幕可用。

### Phase 3：Stereo Compute Graph

1. 建立 shader manifest、Graph compiler 和固定资源池。
2. 实现 preprocess、depth postprocess、parallax、warp、fill、temporal、pack。
3. 完成 CPU reference、golden 和多尺寸 GPU 测试。

完成标准：`docs/01` 算法正确性和 Vulkan 正确性验收通过。

### Phase 4：厂商推理互操作

1. 原样迁入并验证现有WindowsCaptureCUDA、TensorRT和PyTorch CUDA Python实现，再接入Vulkan输出契约。
2. 原样迁入并验证现有WindowsCaptureROCm、PyTorch ROCm/MIGraphX Python实现，再接入external-memory或一次GPU copy路径。
3. DirectML与Apple Python Provider分平台验证；不得通过C++重写替代迁移。

完成标准：实时主路径无 CPU readback，句柄和同步长稳测试通过。

### Phase 5：异步光效、输出与控制面

1. 接入 Glow、颜色归约和墙面反射 Compute Graph。
2. 完成 Vulkan window/headless output。
3. 接入控制协议、GUI、配置快照和 diagnostic bundle。

完成标准：完整产品流程、热更新、故障恢复和 2 小时长稳通过。

### Phase 6：旧架构删除

保留并整理有效的Python Capture、Inference、调度和诊断代码；删除旧OpenGL/D3D11 viewer、Panda3D、WGL/CUDA-GL bridge、CPU实时fallback和历史兼容配置。旧Filament OpenGL-only Bridge由新版Vulkan/OpenGL Bridge取代。更新`src/main.py`、`src/main.bat`、依赖、CI和发布清单，保持当前启动方式并在启动阶段选择后端。

完成标准：发布包和运行依赖中不存在旧图形桥接后端；OpenGL只通过Python `GraphicsBackend`、`viewer.opengl_renderer`和`xr_viewer.core_openxr_opengl`出现；Filament Bridge是唯一自有原生组件。

---

## 24. 完成定义

Vulkan 工程迁移只有同时满足以下条件才算完成：

1. `docs/01` 与本文所有必需验收项通过。
2. OpenXR Session、场景和交换链全部使用 Vulkan。
3. 默认主路径使用 Filament Vulkan Backend；OpenGL 只在符合条件的兼容会话中启用。
4. 立体合成和光效使用 Vulkan Compute。
5. 正式实时路径每帧 CPU 图像回读为零。
6. GPU 任务、资源池和队列全部有界。
7. NVIDIA 主路径为 external-memory 零拷贝；其他平台不超过规范允许的一次 GPU copy。
8. OpenXR 无新输入时仍保持 pose、controller 和 `xrEndFrame` 节奏。
9. Validation、长稳、Session 恢复和 Device Lost 诊断通过。
10. 发布包不包含旧OpenGL viewer、D3D11或Panda3D；Python是正式运行时，禁止的仅是CPU NumPy/PIL逐帧像素往返。
11. Capture、Inference、Vulkan、OpenXR和Output均由Python源码实现，唯一自有原生组件为Filament DLL Bridge。

本文自Python Vulkan Runtime开发开始生效。后续工程实现、代码审查和发布验收均以本文和`docs/01`为唯一目标架构依据。

## 25. 全量符合性追踪

本文与 `docs/01` 的所有要求统一由 [`docs/requirements-matrix.md`](requirements-matrix.md) 追踪。矩阵按架构、捕捉、推理、Vulkan、Compute Graph、Filament、OpenXR、输出、配置、GUI、错误恢复、诊断、性能、测试、平台、CI 和安全领域登记要求，且每条要求必须关联代码映射和测试/实机验收记录。

日常开发使用 `src/tools/check_compliance.py` 检查矩阵结构；发布候选版本使用 `--strict`，只允许 `verified` 或 `accepted` 条目，并额外要求 pytest、三平台 Bridge CI 和专用 GPU/OpenXR 实机验收通过。
