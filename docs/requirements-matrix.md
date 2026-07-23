# D2S 规格符合性矩阵

本文件是 `docs/01-Realtime-2d-to-3d-specification.md` 与
`docs/02-desktop2stereo-engineering-design-specification.md` 的执行追踪表。
规格条目不能只以文字存在；每条要求必须有代码映射、测试或人工验收方式和当前状态。

## 状态定义

| 状态 | 含义 |
|------|------|
| `planned` | 已登记，尚未开始实现。 |
| `in_progress` | 已有实现或迁移工作，但尚未达到验收条件。 |
| `implemented` | 代码已经具备，仍需平台或实机验收。 |
| `verified` | 自动化测试或目标平台测试已通过。 |
| `accepted` | 自动化、平台和必要的实机验收均通过，可作为发布依据。 |

`tools/check_compliance.py` 检查编号唯一性、来源、映射和验收记录。`--strict` 模式只允许
`verified` 或 `accepted`，用于发布候选版本；日常开发允许 `planned` 和 `in_progress` 存在，
但不允许出现没有映射的需求。

## 全量需求矩阵

| ID | 领域 | 必须遵循的要求 | 规范来源 | 代码映射 | 测试/验收 | 状态 |
|----|------|----------------|----------|----------|-----------|------|
| ARCH-001 | 架构 | Python 是正式运行时，`src/` 是发布边界。 | 01§3.4; 02§2.1 | `src/app_runtime/runtime_entry.py`; `src/` | `tests/test_migration_scaffold.py`; runtime startup smoke | in_progress |
| ARCH-002 | 架构 | Vulkan 是主路径，OpenGL 只作为隔离 Fallback，运行中不得热切换。 | 01§1; 01§3.2; 02§2.4 | `src/viewer/`; `src/xr_viewer/` | `tests/test_migration_scaffold.py`; 实机验收 | in_progress |
| ARCH-003 | 架构 | 新代码不得恢复 D3D11、WGL/CUDA-GL、旧 Viewer 或 CPU 实时像素链路。 | 01§1.2; 01§15; 02§24 | `src/` | `tests/test_no_legacy_depth_imports.py`; 静态检查 | in_progress |
| ARCH-004 | 架构 | GPU 资源具有唯一所有者，模块之间只传递定义明确的句柄/契约。 | 01§3.5; 02§3.1 | `src/viewer/vulkan_resources.py`; `src/viewer/` | `tests/test_vulkan_resources.py`; 资源生命周期测试 | in_progress |
| CAPTURE-001 | 捕捉 | Capture Adapter 输出 GPU 资源、尺寸、格式和单调时间戳。 | 01§6.1; 02§8 | `src/` | `tests/test_capture_metadata.py`; `tests/test_capture_public_api.py` | implemented |
| CAPTURE-002 | 捕捉 | 保留并验证 Windows CUDA/ROCm 等有效 Python 捕捉实现，不在迁移中强制改写为 C++。 | 01§1.2; 02§8.2 | `src/` | `tests/test_capture_factory.py`; `tests/test_capture_runners.py` | in_progress |
| CAPTURE-003 | 捕捉 | 捕捉停止、源切换和异常必须有界，不阻塞渲染线程。 | 02§5; 02§8.1 | `src/` | `tests/test_capture_session.py`; 长稳测试 | in_progress |
| INFER-001 | 推理 | NVIDIA、AMD、Apple Provider 通过统一 Inference Adapter 接入。 | 01§1.1; 02§9 | `src/` | `tests/test_depth_onnx_provider.py`; `tests/test_provider_layout.py` | in_progress |
| INFER-002 | 推理 | 推理结果不得进入 CPU 像素往返，允许外部内存或一次 GPU copy。 | 01§6.4; 02§9.6 | `src/viewer/vulkan_resources.py`; `src/viewer/vulkan_interop.py`; `src/viewer/cuda_vulkan_interop.py`; `src/app_runtime/runtime_output.py`; `src/app_runtime/vulkan_runtime.py`; `src/stereo_runtime/vulkan_image_pass.py` | `tests/test_vulkan_resources.py`; `tests/test_vulkan_interop.py`; `tests/test_cuda_vulkan_interop.py`; `tests/test_runtime_output.py`; `tests/test_vulkan_runtime.py`; 真实 CUDA/ROCm 互操作测试; capability report | in_progress |
| INFER-003 | 推理 | 单帧推理失败丢帧并计数，连续失败后重建 Adapter。 | 02§9.6; 02§17.2 | `src/stereo_runtime/pipeline.py` | `tests/test_pipeline.py`; 故障注入 | in_progress |
| VK-001 | Vulkan | 默认请求 Vulkan 1.4，最低接受 Vulkan 1.2，并按 Loader/Runtime/Device 能力协商。 | 01§4.2; 01§5.1; 02§7.1 | `src/viewer/vulkan_context.py`; `src/xr_viewer/core_openxr_vulkan.py` | `tests/test_openxr_vulkan.py`; `src/tools/probe.py` | verified |
| VK-002 | Vulkan | Instance、Device、扩展和 Feature 统一由 Vulkan Context 创建和验证。 | 01§5.1; 02§7.1 | `src/viewer/vulkan_context.py` | `tests/test_openxr_vulkan.py` | implemented |
| VK-003 | Vulkan | 必须查询并显式启用 Timeline Semaphore，不得仅依据 API 版本假设启用。 | 01§5.1; 01§10.1; 02§7.6 | `src/viewer/vulkan_context.py`; `src/xr_viewer/core_openxr_vulkan.py` | `tests/test_openxr_vulkan.py` | verified |
| VK-004 | Vulkan | 使用 `pNext` Feature 链，`sType` 和链的生命周期必须正确。 | 01§5.1; 02§7.1 | `src/viewer/vulkan_context.py`; `src/xr_viewer/core_openxr_vulkan.py` | `tests/test_openxr_vulkan.py` | in_progress |
| VK-005 | Vulkan | 队列、Frame Context、Descriptor、Pipeline 和图像状态必须有界且可追踪。 | 01§5.2-5.5; 02§7.2-7.5 | `src/viewer/vulkan_context.py`; `src/viewer/vulkan_descriptors.py`; `src/viewer/vulkan_compute_pipeline.py` | `tests/test_openxr_vulkan.py`; `tests/test_migration_scaffold.py`; GPU validation; 长稳测试 | verified |
| VK-006 | Vulkan | 必须执行 Layout、Access、Queue Ownership 和提交顺序验证。 | 01§5.2; 01§10; 02§7.6 | `src/viewer/vulkan_context.py`; `src/viewer/vulkan_descriptors.py` | `tests/test_openxr_vulkan.py`; Validation Layer GPU 测试 | verified |
| VK-007 | Vulkan | Compute、Graphics 和 XR 输出不得发生未登记的 CPU 图像回读。 | 01§6; 01§8.4; 02§14 | `src/viewer/vulkan_context.py`; `src/viewer/cuda_vulkan_interop.py`; `src/app_runtime/runtime_output.py`; `src/xr_viewer/core_openxr_vulkan.py`; `src/tools/vulkan_transfer_smoke.py` | `tests/test_cuda_vulkan_interop.py`; `src/tools/vulkan_transfer_smoke.py`; 静态检查; GPU profiling | verified |
| VK-009 | Vulkan | 左右眼使用有界多帧输出图像环；CUDA external memory、image view 和 Filament Texture 按槽位持久化缓存，正常帧不得重建或销毁纹理。 | 01§8.2.1; 02§5.4 | `src/app_runtime/runtime_output.py`; `native/filament/bridge/bridge_internal.h`; `native/filament/bridge/bridge_screen.cpp` | `tests/test_cuda_vulkan_interop.py`; GPU profiling; 长稳测试 | verified |
| VK-010 | Vulkan | CUDA/Vulkan/Filament 的 producer-ready 与 consumer-release 必须通过 external semaphore/timeline 表达；CPU stream 等待只能作为显式兼容降级。 | 01§10.1; 02§6.2 | `src/viewer/cuda_vulkan_interop.py`; `src/xr_viewer/filament_vulkan_bridge.py`; native Bridge ABI | external semaphore smoke; Validation Layer; 实机长稳测试 | implemented |
| VK-011 | Vulkan | 在独立 `d2s-vulkan-1.4` 分支基于固定 Vulkan 1.4 Registry 版本远程生成 Python binding，并以同场景基准验证 `hostImageCopy` 与独立 Transfer Queue；主路径必须保留 Vulkan 1.2/1.3 回退，未经三平台 CI、Validation Layer 和实机收益验证不得合入。 | 01§4.2; 02§3.1.1; 02§7.1 | `docs/02-desktop2stereo-engineering-design-specification.md`; `docs/03-d2s_vulkan_migration_technical_report.md` | wheel reproducibility; 三平台 CI; Validation Layer; upload latency and CPU-GPU profiling | planned |
| VK-008 | Vulkan | 资源分配、释放、Resize、Device Lost 和异常清理必须可诊断。 | 02§7.2; 02§17 | `src/viewer/`; `src/app_runtime/vulkan_runtime.py` | `tests/test_vulkan_runtime.py`; 故障注入; 长稳测试 | verified |
| GRAPH-001 | 计算图 | 按规格顺序执行预处理、深度后处理、视差、变形、修补和时域稳定；每个 Pass 必须声明 workgroup 与资源读写。 | 01§6; 02§10 | `src/stereo_runtime/vulkan_graph.py`; `src/stereo_runtime/vulkan_image_pass.py`; `src/viewer/vulkan_compute_pipeline.py`; `shaders/manifest.json`; `src/tools/validate_shader_manifest.py` | `tests/test_migration_scaffold.py`; `tests/test_shader_manifest.py`; shader golden tests; GPU image smoke | in_progress |
| GRAPH-002 | 计算图 | Glow、平均色、墙面反射等异步光效使用独立有界资源。 | 01§7; 02§12 | `src/stereo_runtime/vulkan_graph.py`; `shaders/` | GPU timing; 功能验收 | in_progress |
| GRAPH-003 | 计算图 | latest-frame 覆盖旧帧，负载升高时延迟不持续累积；上游 GPU 完成点必须通过 timeline 传入 Compute submit。 | 01§10.2; 02§5.3; 02§7.6 | `src/stereo_runtime/vulkan_graph.py`; `src/viewer/vulkan_context.py` | `tests/test_migration_scaffold.py`; `src/tools/vulkan_compute_smoke.py`; 压力测试 | implemented |
| FILAMENT-001 | Filament | 场景使用 Filament Vulkan 后端，Bridge 只暴露窄 C ABI。 | 01§3.1; 01§15.2; 02§11 | `src/xr_viewer/filament_preview_bridge.py`; `src/xr_viewer/native/` | `tests/test_filament_vulkan_bridge.py`; CI | implemented |
| FILAMENT-002 | Filament | 单一 Filament Engine/Scene 共享 GLB、材质、纹理和 Shader；左右眼独立 Renderer、View、Camera、OpenXR swapchain 与外部屏幕纹理；虚拟屏幕可采样运行时 Vulkan 输出图像；手柄模型、按键动画、激光和逐手空闲显隐在 Projection Layer 内更新；GLB 的 `_value/_min/_max` 完整三元组必须在非渲染节点枚举缺失时按名称回退发现，摇杆、触控板和 thumbrest 的 touch 状态必须与 click 分离并驱动 touched 动画。手柄加载保留 WebXR Input Profiles GLB 原始 PBR 参数，并使用环境间接光与旧工程 profile 补光，不得由 Bridge 统一改写材质。暗色模型允许由手柄 profile 单独提高环境间接光倍率，PICO/Quest 保持 `1.0`；切换品牌时必须同步刷新。B 键引导端点必须从当前模型的按键动画枢轴解析，品牌切换后立即重算。 | 01§15.2; 02§8.3; 02§11.1-11.5 | `src/xr_viewer/controller_models.py`; `src/xr_viewer/controllers/`; `src/xr_viewer/core_openxr_vulkan.py`; `src/xr_viewer/core_controller_ray.py`; `native/filament/bridge/bridge_controller.cpp`; `native/filament/bridge/bridge_material.cpp`; `filament_bridge_create_eye_swapchain`; `filament_bridge_set_active_eye`; `filament_bridge_create_screen`; `filament_bridge_set_screen_image`; `filament_bridge_load_controller`; `filament_bridge_set_controller_pose`; `filament_bridge_set_controller_inputs`; `filament_bridge_set_controller_visible`; `filament_bridge_set_controller_laser`; `filament_bridge_set_ambient_light` | `tests/test_filament_vulkan_bridge.py`; `tests/test_openxr_vulkan.py`; 单 Engine 双眼 Bridge CI；GLB 原材质/环境光/动画/虚拟屏幕/各品牌手柄、B 键锚点、5 秒显隐、按键动画和激光实机验收 | in_progress |
| FILAMENT-003 | Filament | Bridge 二进制只由三平台 CI 生成并记录 ABI、Filament 版本和 SHA，产物分别存放于 `native/windows`、`native/linux`、`native/macos`。 | 01§15.1; 02§20.1.1 | `.github/workflows/`; `src/xr_viewer/native/` | Actions artifact 检查 | verified |
| FILAMENT-004 | Filament | HDR 场景使用线性 Rec709/D65 和 ColorGrading tone mapping；场景曝光进入 ColorGrading，不修改 glTF 材质颜色。显示参考的虚拟屏幕、UI 和激光进入独立无后处理 View，按 sRGB 图像导入并只解码一次；主 View 不得渲染 LDR 图元。 | 01§6.3.1; 01§8.2.2; 02§8.3; 02§11.2.1 | `native/filament/bridge/bridge_material.cpp`; `native/filament/bridge/bridge_context.cpp`; `native/filament/bridge/bridge_eye.cpp`; `native/filament/bridge/bridge_screen.cpp`; `native/filament/bridge/bridge_laser.cpp` | `tests/test_filament_vulkan_bridge.py`; `tests/test_openxr_vulkan.py`; `.github/workflows/filament-bridge.yml` | implemented |
| FILAMENT-005 | Filament | 每眼维护运行时 VkImage 到 Filament Texture 的持久化缓存；Projection Layer 两眼先提交、帧边界统一等待并统一 release。 | 01§8.2.1; 02§5.5 | `native/filament/bridge/bridge_internal.h`; `native/filament/bridge/bridge_eye.cpp`; `native/filament/bridge/bridge_screen.cpp`; `src/xr_viewer/core_openxr_vulkan.py` | Bridge CI; OpenXR 帧时序测试; 实机 FPS/Validation 测试 | verified |
| FILAMENT-006 | Filament | `filament_bridge.h` 保持稳定 Python C ABI；`filament_bridge.cpp` 只负责 ABI 转发，Engine/Scene、Eye、Scene、Controller、Laser、Screen、Material 和 Preview 策略按内部模块分离，内部 C++ 符号不得对外导出。 | 02§11.2 | `native/filament/bridge/`; `tests/test_filament_vulkan_bridge.py` | 模块布局测试; ABI export 检查; 三平台 Bridge CI | implemented |
| FILAMENT-007 | Filament | 手柄照明区分 HDR 图片环境和 3D 房间 profile；两种模式都必须叠加独立屏幕光。房间模式使用 profile ambient/head/top light；HDR 模式使用匹配环境的预过滤 reflection cubemap 与 irradiance。屏幕光在线性空间异步取样，使用屏幕法线和对角线衰减，只作用于手柄光照通道，不受 `controller_hdr_lighting` 控制。 | 02§8.3 | `src/app_runtime/runtime_output.py`; `src/xr_viewer/core_openxr_vulkan.py`; `native/filament/bridge/bridge_material.cpp`; `native/filament/bridge/bridge_screen.cpp`; `filament_bridge_set_ambient_light`; `filament_bridge_set_screen_light` | `tests/test_runtime_output.py`; `tests/test_filament_vulkan_bridge.py`; `.github/workflows/filament-bridge.yml` | in_progress |
| SCENE-001 | 场景 | 相机、near/far、每眼投影、坐标系和场景尺寸遵循 profile。 | 01§3.1; 01§8; 02§11.5 | `src/xr_viewer/` | 头显实测 | in_progress |
| SCENE-002 | 场景 | 场景加载失败保留 last-good scene，资源释放和 Resize 不得悬空。 | 02§11.3; 02§17.2 | `src/xr_viewer/` | 故障注入; 长稳测试 | planned |
| OPENXR-001 | OpenXR | OpenXR Vulkan Session 使用 Runtime 提供的 Vulkan Instance/Device 要求。 | 01§8.1; 02§13.2 | `src/xr_viewer/core_openxr_vulkan.py` | `tests/test_openxr_vulkan.py` | verified |
| OPENXR-002 | OpenXR | Frame Loop 必须遵循 poll/wait/begin/acquire/render/release/end 顺序；正式 OpenXR/Vulkan/Filament 初始化必须等待推理加载、首帧输出和 shape-dependent warmup 完成。 | 01§8.2; 02§4.3; 02§13.3 | `src/xr_viewer/core_openxr_vulkan.py`; `src/app_runtime/runtime_entry.py`; `src/stereo_runtime/pipeline.py`; `src/app_runtime/runtime_output.py` | `tests/test_openxr_vulkan.py`; `tests/test_runtime_entry.py`; `tests/test_pipeline.py`; `tests/test_runtime_output.py`; OpenXR lifecycle test; 头显实测 | in_progress |
| OPENXR-003 | OpenXR | Projection Layer、Quad Layer 和 `xrEndFrame` 由单一 Composition Builder 管理。 | 01§8.3; 02§13.5 | `src/xr_viewer/core_openxr_vulkan.py` | `tests/test_openxr_vulkan.py`; Layer 验收 | in_progress |
| OPENXR-004 | OpenXR | Swapchain acquire/wait/release 必须成对，重建前等待引用资源。 | 01§8.2; 02§13.4 | `src/xr_viewer/core_openxr_vulkan.py` | `tests/test_openxr_vulkan.py`; lifecycle test; Session loss 实测 | in_progress |
| OPENXR-005 | OpenXR | Projection Layer 输出必须使用 Runtime 支持的 Vulkan sRGB 格式；不得以 UNORM 静默回退或提供改变颜色语义的线性 8-bit 路径。无可用 sRGB 格式时必须失败并报告原因。 | 01§6.3.1; 01§8.2.2; 02§8.3; 02§13.4 | `src/xr_viewer/core_openxr_vulkan.py`; `src/app_runtime/runtime_output.py`; `native/filament/bridge/bridge_eye.cpp` | `tests/test_openxr_vulkan.py`; OpenXR 实机格式检查 | implemented |
| OUTPUT-001 | 输出 | Preview、OpenXR、Headless/Encoder 使用统一 Left/Right/SBS 输出契约；明确 `color_space=srgb`、`image_origin=top_left`，网络编码只做通道排列和标准 BT.709 元数据，不重复 gamma；各输出后端必须依据 origin 明确执行坐标适配。 | 01§8.4; 02§14 | `src/app_runtime/output_contract.py`; `src/app_runtime/runtime_output.py`; `src/viewer/cuda_vulkan_interop.py`; `src/xr_viewer/core_openxr_vulkan.py`; `src/streaming/` | `tests/test_output_contract.py`; `tests/test_runtime_output.py`; `tests/test_cuda_vulkan_interop.py`; `src/tools/vulkan_transfer_smoke.py`; 输出集成测试 | in_progress |
| CFG-001 | 配置 | 运行时使用带 schema version 的规范化配置，核心不解析旧字段 alias。 | 01§9; 02§15 | `src/` | `tests/test_settings_snapshot.py`; schema test | in_progress |
| CFG-002 | 配置 | 参数热更新按 Uniform、Temporal Reset、Graph Rebuild、Session Rebuild、Restart 分类。 | 01§9; 02§15.2 | `src/` | `tests/test_runtime.py`; 配置测试 | planned |
| GUI-001 | GUI | GUI 只负责配置、启动停止、状态和日志，不拥有 Vulkan/OpenXR/Filament 资源。 | 02§16 | `src/gui/` | GUI 启动/控制测试 | in_progress |
| ERR-001 | 错误 | 模块错误转换为结构化 Status，资源清理幂等，Device Lost 不原地继续提交。 | 01§11; 02§17 | `src/` | `tests/test_logging_setup.py`; 故障注入 | planned |
| OBS-001 | 诊断 | capability report、结构化日志、GPU timing 和 diagnostic bundle 可导出。 | 01§12; 02§18 | `src/app_runtime/probe.py`; `src/` | `src/tools/probe.py`; 诊断验收 | implemented |
| PERF-001 | 性能 | 90 Hz 应用 GPU 关键路径目标不超过 10 ms，并分开报告各阶段耗时。 | 01§13.4; 02§21.1 | `src/` | GPU benchmark; 头显实测 | verified |
| PERF-002 | 性能 | 资源池、帧上下文、队列和 Telemetry ring 有界，显存超预算时拒绝或降档。 | 01§10; 02§21.2-21.3 | `src/` | 显存压力和长稳测试 | verified |
| TEST-001 | 测试 | 每个功能必须有单元、集成、平台或人工验收记录，不能只以代码存在为完成。 | 01§13; 02§19; 02§24 | `tests/`; `docs/` | 本矩阵; CI | in_progress |
| TEST-002 | 测试 | Vulkan Validation、Shader golden、互操作、Filament、OpenXR、Fallback 和长稳测试必须分层执行。 | 02§19 | `tests/`; `.github/workflows/` | CI; 专用硬件 runner | planned |
| PLATFORM-001 | 平台 | Windows/Linux 使用原生 Vulkan，macOS 使用 MoltenVK；OpenGL Fallback 按平台能力报告。 | 01§4; 01§3.2; 02§2.4 | `src/`; `.github/workflows/` | 三平台 CI; 实机验收 | in_progress |
| PLATFORM-002 | 平台 | Vulkan 不可用、MoltenVK 失败/不可接受或用户选择兼容模式时受控进入 OpenGL。 | 01§3.2; 02§4.5 | `src/` | Fallback 集成测试 | planned |
| CI-001 | 交付 | Python 检查、pytest、Shader、Bridge 三平台构建和依赖清单纳入 CI。 | 02§20 | `.github/workflows/` | GitHub Actions | in_progress |
| CI-002 | 交付 | 发布包不包含旧图形桥接和临时 build 目录，Bridge 与 Filament 运行库版本一致。 | 01§15.1; 02§20.2 | `.github/workflows/`; `src/xr_viewer/native/` | package audit | planned |
| SEC-001 | 安全 | 资产、配置、控制通道、下载模型和诊断信息必须执行边界、权限和脱敏检查。 | 02§22 | `src/` | 安全静态检查; package test | planned |

## 变更规则

修改实现前先查影响范围；修改后必须更新对应矩阵行、测试和 `changelog.md`。新增功能如果不能映射到本表，视为需求遗漏，不能直接合并。发布候选版本必须执行：

```text
src/tools/check_compliance.py --strict
pytest -q
三平台 Bridge CI
专用 GPU/OpenXR 实机验收
```
