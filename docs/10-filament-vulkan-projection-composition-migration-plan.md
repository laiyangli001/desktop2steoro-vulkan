# Filament/Vulkan Projection Composition 迁移方案

**文档状态**：计划中
**更新时间**：2026-08-04
**适用范围**：OpenXR Vulkan Projection/Quad Layer、Filament Bridge、屏幕采样、激光、Glow 和线程调度

## 1. 目标

将 2026-07-21 以后逐步加入 Filament 的屏幕、屏幕采样、激光和 Glow 合成迁回 Vulkan，恢复清晰的渲染边界：

- Filament 只加载并渲染环境 GLB、手柄 GLB，输出每眼颜色和深度。
- Vulkan 统一合成 Projection Layer：环境/手柄结果、虚拟屏幕、激光和 Glow。
- Quad Layer 承载 FPS 面板、操作指南、固定在虚拟屏幕上的 2D 光圈、固定在虚拟键盘上的 2D 光圈和虚拟键盘。
- OpenXR Session、swapchain acquire/release、Projection/Quad layer 构建和 `xrEndFrame` 由单一 Presenter 线程拥有。
- 虚拟屏幕、效果和 UI 使用独立生产线程，通过有界 latest-frame 资源与 timeline 同步，不阻塞视频源帧率。
- Projection swapchain 优先 `array_size=2`，失败回退 per-eye；Quad array 作为独立实验，不阻塞 Projection 主路径。

## 2. 明确删减的照明范围

手柄照明只保留两类显式参数：

1. 项目设置的基础环境光。
2. 项目设置的头顶光。

删除屏幕放射光/屏幕中心聚光灯及其所有派生状态：

- 不再从屏幕图像计算平均色来驱动手柄光照。
- 不再创建 Filament 屏幕光实体。
- 不再保留 `screen_light_color`、`screen_light_intensity`、屏幕法线衰减和屏幕光更新 ABI。
- Glow 仍作为视觉效果处理，但不参与手柄照明。

## 3. 目标架构

```text
Capture / Inference
        |
        +--> Screen Worker
        |      -> latest stereo VkImage
        |
        +--> Effects/UI Worker
        |      -> Glow / FPS / Guide / Aperture / Keyboard VkImage
        |
        +--> Filament Worker
               -> environment/controller color + depth
                              |
OpenXR Presenter (唯一 layer owner)
  acquire Projection swapchain
  -> Vulkan Projection Composer
       1. composite Filament color/depth
       2. draw virtual screen
       3. draw laser
       4. draw Glow
  -> release Projection swapchain
  -> submit FPS/Guide/Aperture/Keyboard Quad Layers
  -> xrEndFrame
```

线程只负责生产或记录命令，不能并发操作同一个 Filament Engine、同一个 `VkQueue` 或 OpenXR Session。Vulkan 多线程用于命令录制和资源生产；实际 queue submit 仍遵守 Vulkan queue 外部同步。

## 4. 迁移顺序

### 阶段 0：冻结边界和回退

- 增加 `D2S_VULKAN_PROJECTION_COMPOSER=1` 实验开关。
- 保留旧 Filament 屏幕路径作为临时回退，不再向旧路径添加功能。
- 固定颜色空间、左右眼顺序、latest-frame 和资源租约契约。
- 建立 Projection/Quad Layer 数量、顺序和 image release 日志。

完成条件：旧路径行为不变，新路径可启动并能安全关闭。

### 阶段 1：Vulkan 屏幕 Projection

新增最小 Vulkan Projection Composer，先只清屏并绘制左右眼虚拟屏幕：

- 复用 `src/viewer/vulkan_context.py` 的资源状态和 timeline 提交。
- 复用 `src/utils/screen_resolution_policy.py` 的分辨率档位策略。
- 屏幕纹理直接由 Vulkan 采样，不创建 Filament 外部屏幕纹理。
- Projection 优先创建 `array_size=2`，layer 0/1 对应左右眼。
- 创建、合成或 Runtime 能力失败时回退为两个 `array_size=1` per-eye swapchain。

完成条件：Vulkan 直接绘制的左右眼画面正确，连续运行 10 分钟无 device lost。

### 阶段 2：迁移屏幕清晰化

将 `native/filament/bridge/bridge_screen.cpp` 中的采样逻辑迁为 Vulkan shader/pass：

- EASU：Vulkan Compute upscale。
- Lanczos2/RCAS：Vulkan Compute prefilter。
- MIP：Vulkan layered MIP/SPD 或等价 Vulkan 路径。
- 最终屏幕：Vulkan vertex/fragment pass，支持平面和曲面几何。

保留 `ScreenSamplingPlan` 的输入/头显分辨率矩阵，但删除其对 `filament_bridge_set_screen_*` 的依赖。所有路径保持 sRGB/linear 契约，不使用 tone mapping，不进行 CPU 像素回读。

完成条件：EASU、Lanczos2、RCAS、MIP 视觉回归通过，屏幕清晰度不低于旧路径。

### 阶段 3：Filament 改为环境/手柄颜色深度生产者

调整 Filament Bridge：

- 只加载环境 GLB、手柄 GLB、材质、动画和相机。
- 输出每眼颜色和深度资源；不创建屏幕、激光、Glow Renderable。
- 颜色/深度以 timeline 或等价完成点交给 Vulkan Composer。
- 基础环境光和头顶光保留为唯一手柄照明来源。
- 删除屏幕放射光实体、平均色到手柄照明的路径和对应 ABI。

完成条件：环境、手柄模型、按键动画和深度遮挡正确；屏幕光关闭后画面仍符合基准。

### 阶段 4：迁移激光和 Glow

激光：

- 复用 Python Presenter 的 Aim pose、滤波和逐手显隐状态。
- 将交叉锥形面、渐变和深度测试改为 Vulkan pipeline。
- 使用 Filament 输出的手柄深度遮挡激光根部。

Glow：

- 保留 `src/stereo_runtime/vulkan_glow_source_pass.py` 的 Vulkan Compute 预处理。
- 删除 Glow 外部图像导入 Filament 的步骤。
- Vulkan Projection Composer 直接采样 Glow 图像并完成合成。
- Glow 不参与手柄照明，不改变基础环境光或头顶光。

完成条件：激光遮挡、Glow 五态和 latest-frame 复用正确；Glow 线程不阻塞主屏幕。

### 阶段 5：Quad Layer 与两处 2D 光圈

Quad Layer 内容：

- FPS 面板。
- 操作指南。
- 固定在虚拟屏幕上的 2D 光圈。
- 固定在虚拟键盘上的 2D 光圈。
- 虚拟键盘。

内容生产复用 Vulkan MSDF/纹理上传路径。Quad array 实验使用一个 `array_size=2` swapchain，并分别提交：

```text
Quad Left  -> imageArrayIndex=0, eyeVisibility=LEFT
Quad Right -> imageArrayIndex=1, eyeVisibility=RIGHT
```

如果两眼内容相同，使用一个 `array_size=1` Quad、`eyeVisibility=BOTH`，不强制使用 array。Runtime 创建或提交失败时回退 per-eye Quad swapchain。

完成条件：Virtual Desktop 能接受 Quad array；即使失败，Projection array/per-eye 主路径仍然可用。

### 阶段 6：删除旧 Filament 显示代码

所有新路径通过验收后，删除或停用：

- `native/filament/bridge/bridge_screen.cpp/.h`
- `native/filament/bridge/bridge_laser.cpp/.h`
- `native/filament/bridge/bridge_glow.cpp/.h`
- `filament_bridge_set_screen_*`
- `filament_bridge_set_controller_laser`
- `filament_bridge_set_glow_*`
- 屏幕放射光和平均色驱动手柄照明的 C ABI、Python wrapper、配置项和测试

保留环境光、头顶光、环境/手柄 GLB 和必要的颜色/深度输出 ABI。

## 5. 线程和同步约束

- Presenter 线程独占 OpenXR Session、swapchain、CompositionBuilder 和 `xrEndFrame`。
- Filament worker 不与其它线程并发调用同一个 Filament Engine。
- Screen Worker 只发布已完成的最新屏幕图像；没有新帧时复用 last-good 图像。
- Effects/UI Worker 只发布 Glow、Quad 和光圈纹理；不得直接操作 Presenter-owned 资源。
- 所有跨线程图像使用有界 ring slot、producer-ready timeline 和 consumer-release timeline。
- 禁止无界队列、`vkDeviceWaitIdle`、每帧 `flushAndWait` 和 CPU 像素往返。
- 任何 worker 超时只丢弃新帧，不阻塞 XR Presenter 的帧边界。

## 6. Projection Swapchain 策略

1. 尝试创建 `array_size=2` Projection swapchain。
2. 用 layer 0/1 分别绑定左右眼 Projection View。
3. Vulkan Composer 使用 layered draw/dispatch；Filament 环境/手柄可先逐眼，稳定后再启用 multiview。
4. Runtime、尺寸、颜色/深度或 Composer 能力不满足时，销毁临时 array swapchain，回退两个 per-eye swapchain。
5. Quad array 实验与 Projection 能力解耦，失败不得影响 Projection。

## 7. 验收门槛

- 左眼/右眼颜色和屏幕纹理顺序正确。
- `array_size=2` Projection 成功，per-eye fallback 成功。
- 屏幕清晰化视觉回归通过。
- 环境 GLB、手柄 GLB、按键动画和手柄深度正确。
- 激光被手柄几何自然遮挡。
- Glow 不阻塞视频帧率，资源租约无覆盖和泄漏。
- 两处 2D 光圈和虚拟键盘在 Quad Layer 正确显示。
- 手柄只受基础环境光和头顶光影响，不再受屏幕放射光影响。
- Virtual Desktop Quad array 实验成功或有明确 per-eye 回退日志。
- 连续运行至少 10 分钟，无 `VK_ERROR_DEVICE_LOST`、validation error 或持续帧率下降。

## 8. 回退原则

- 每个阶段单独提交和实机验证。
- 新 Vulkan Composer 失败时回退旧路径，但禁止继续扩展旧 Filament 屏幕功能。
- Projection array 失败只回退 per-eye，不回退到 Quad Layer。
- Quad array 失败只回退 per-eye Quad，不影响 Projection。
- 任何同步能力不足优先丢弃新帧，禁止阻塞或回读 CPU。
