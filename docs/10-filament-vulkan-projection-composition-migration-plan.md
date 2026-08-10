# Filament/Vulkan Projection Composition 迁移方案

**文档状态**：进行中
**更新时间**：2026-08-08
**适用范围**：OpenXR Vulkan Projection/Quad Layer、Filament Bridge、屏幕采样、激光、Glow 和线程调度

## 1. 目标

将 2026-07-21 以后逐步加入 Filament 的屏幕、屏幕采样、激光和 Glow 合成迁回 Vulkan，恢复清晰的渲染边界：

- Filament 只加载并渲染环境 GLB、手柄 GLB，输出每眼颜色和深度。
- Vulkan 统一合成 Projection Layer：环境/手柄结果、虚拟屏幕、激光和 Glow。
- 单图 Tool Quad 承载 FPS 面板、操作指南、固定在虚拟屏幕上的 2D 光圈、固定在虚拟键盘上的 2D 光圈和虚拟键盘；主屏幕不使用 Quad swapchain。
- OpenXR Session、swapchain acquire/release、Projection/Quad layer 构建和 `xrEndFrame` 由单一 Presenter 线程拥有。
- 虚拟屏幕、效果和 UI 使用独立生产线程，通过有界 latest-frame 资源与 timeline 同步，不阻塞视频源帧率。

迁移工作的行为基准固定为删除前的 Filament 实现。Vulkan 只替换执行后端，禁止自行改写屏幕/Glow 的几何、跟随关系、参数公式、模式映射、颜色混合、可见性条件或降级语义。尚未完成原样迁移的效果必须保持关闭，不能用近似实现代替，也不能依赖实机反复试错来重新定义既有行为。
- Projection swapchain 优先 `array_size=2`，失败回退 per-eye。Virtual Desktop 已验证不接受双眼 Screen Quad swapchain；该实验永久停用，不能影响 Projection 主路径。
- 遮挡顺序固定：无环境模型时为“手柄 → Glow → 屏幕”；加载环境模型时为“手柄 → 屏幕 → 环境”。
  该顺序必须由 Filament producer depth 与 Vulkan Projection depth attachment 共同保证，不能用
  纯提交顺序或整屏透明度近似替代。

## 2. 明确删减的照明范围

手柄照明保留三类显式参数：

1. 项目设置的基础环境光。
2. 项目设置的头顶光。
3. Vulkan 屏幕平均色驱动的低强度动态补光。

删除依赖 Filament 屏幕 Renderable 和屏幕纹理交接的旧放射光路径：

- 不将屏幕纹理重新导入 Filament，也不恢复 Filament 屏幕几何。
- 复用 Vulkan/CUDA 异步 reduction 的线性平均色，由 Python 完成平滑、限幅和 Profile 参数换算。
- Bridge 只维护一盏朝屏幕到手柄方向照射、仅开启手柄 light channel 的动态方向光。
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

### 已停用：Screen Quad Reprojection 实验

为隔离全分辨率 Vulkan Projection Composer 的 raster/fence 成本，曾提供
`D2S_OPENXR_SCREEN_QUAD_REPROJECTION=1` 实验路径。它将左右 SBS 眼图复制到一个
双 array-layer OpenXR Quad swapchain，以 `LEFT` / `RIGHT` eye visibility 提交。
Virtual Desktop 实机验证不接受此 swapchain，因此该环境变量现在被忽略，主屏幕始终
使用 Projection swapchain。

- 保留诊断代码和测试，仅用于离线兼容性验证；不得重新接入 Presenter 主路径。
- 该结论不影响现有单图 Tool Quad。主屏幕的头姿重投影、清晰化和合成后续均在
  Projection Composer 内实现。
  -> submit FPS/Guide/Aperture/Keyboard Quad Layers
  -> xrEndFrame
```

线程只负责生产或记录命令，不能并发操作同一个 Filament Engine、同一个 `VkQueue` 或 OpenXR Session。Vulkan 多线程用于命令录制和资源生产；实际 queue submit 仍遵守 Vulkan queue 外部同步。

## 4. 迁移顺序

### 阶段 0：冻结边界和回退（已完成）

- `D2S_VULKAN_PROJECTION_COMPOSER` 默认开启，也是唯一的 SBS 屏幕路径。设置为 `0` 仅用于诊断环境/手柄渲染，不能重新启用 Filament 或 Quad 主屏幕。
- `D2S_VULKAN_PROJECTION_QUALITY_CHAIN` 默认开启；设置为 `0` 时保留 Vulkan Projection Composer，但跳过 EASU/Lanczos、RCAS 和 MIP 质量链，直接对原始 LOD0 源图进行投影。该开关用于低端 GPU 性能 A/B 验证，不降低源图或 Projection swapchain 分辨率。
- 删除旧 Filament 屏幕 Renderable、纹理导入、MIP、RCAS、屏幕光和 `filament_bridge_set_screen_*` ABI。Composer 创建、记录或提交异常时，Presenter 仅渲染 Filament 环境/手柄或清屏，并在同一 XR 帧完成 acquire/release 和 `xrEndFrame`；不得重新绑定 SBS 图像。
- 日志只在异常种类或消息变化时输出 `Vulkan projection composer fallback`；`vk_composer_fallback` 和 `vk_composer_fallback_prepare` 用于确认环境/手柄降级频率与准备开销。
- 固定颜色空间、左右眼顺序、latest-frame 和资源租约契约。
- 建立 Projection/Quad Layer 数量、顺序和 image release 日志。

完成条件：默认 Projection 路径可启动；显式关闭 Composer 或 Composer 异常时，不会重新激活旧屏幕路径，且不泄漏已 acquire 的 swapchain image。

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

当前进度：默认 Composer 路径已不再调用 `filament_bridge_set_screen_sampling` 或
`filament_bridge_set_screen_upscale`；`ScreenSamplingPlan` 由 Presenter 保留给 Vulkan 管线。
质量链必须先完成，再交给最终的平面/曲面屏幕几何投影，不能在 Projection fragment 中二次
Lanczos/EASU，也不能先把滤波结果投影进 swapchain 后再 RCAS。

```text
native_mip                 source -> LOD0 copy -> mip generation -> screen projection
downsample_lanczos_rcas    source -> Lanczos2 -> RCAS -> mip generation -> screen projection
upscale_easu               source -> EASU     -> RCAS -> mip generation -> screen projection
```

每眼使用三个有界、设备本地的 transient slot。`Lanczos2` 与 `EASU` 写入单层质量纹理，
RCAS 读取该纹理并写入带 mip 的内部纹理，随后 Vulkan linear blit 生成完整 mip chain，最终
Projection pass 只采样已完成的质量纹理。所有阶段和双眼最终绘制录入同一次 graphics queue
submit；slot 或 descriptor 尚未完成、格式不支持 linear blit 时立即回退 direct single-pass，
不等待、不扩大 slot 数。内部 `VulkanTransientImage` 使用
`COLOR_ATTACHMENT | SAMPLED | TRANSFER_SRC | TRANSFER_DST`；不得对 CUDA/ROCm 导入的
单层输入图直接生成 mip。

`native_mip` 与旧 Filament 匹配档位一致：只复制 LOD0 并生成动态 mip，不额外经过
Lanczos2 或 RCAS。最终 Projection shader 保持直接采样，避免把清晰化逻辑混入世界空间
屏幕投影。实机验收需确认 `vk_composer_*` 质量链统计、无 timeline 等待，以及屏幕清晰度不低于
已验证的 Filament 路径。

保留 `ScreenSamplingPlan` 的输入/头显分辨率矩阵，但删除其对 `filament_bridge_set_screen_*` 的依赖。所有路径保持 sRGB/linear 契约，不使用 tone mapping，不进行 CPU 像素回读。

完成条件：EASU、Lanczos2、RCAS、MIP 实机视觉和 timeline 回归通过，屏幕清晰度不低于旧路径。

### 阶段 3：Filament 改为环境/手柄颜色深度生产者

调整 Filament Bridge：

- 只加载环境 GLB、手柄 GLB、材质、动画和相机。
- 输出每眼颜色和深度资源；不创建屏幕、激光、Glow Renderable。
- 颜色/深度以 timeline 或等价完成点交给 Vulkan Composer。
- `VulkanStereoOutputFrame` 预留成对的 `left_depth`/`right_depth` 资源字段；两眼必须
  同时提供，禁止把颜色图或 CPU 深度替代真实 GPU 深度附件。
- 基础环境光和头顶光为稳定照明，屏幕平均色只提供有界的动态手柄补光。
- 删除旧 Filament 屏幕纹理放射光实体和对应纹理交接 ABI。

当前进度：已删除 `bridge_screen.cpp/.h`、屏幕 C ABI、Python ctypes wrapper 和 Presenter 的
Filament SBS 图像交接。`_filament_screen` 名称暂时保留为 Presenter-owned 屏幕几何状态，供
Vulkan Composer、曲面屏控制和交互命中使用，不代表 Filament Renderable。`bridge_glow.cpp/.h`
已删除；Glow 不再由 Filament Bridge 持有。

手柄和环境照明参数也已移出 DLL：`environments/common.json` 提供共享默认值，3D/HDR
`profile.json` 和 lighting preset 可覆盖环境 ambient 的颜色/lux、手柄 ambient 的普通/HDR
lux、Head/Top 灯的颜色、权重、相对头显偏移、falloff 和 shadow。Python 将易调的比例换算为
最终 lux/candela，通过版本化 `FilamentBridgeLightingConfig` 一次传入；DLL 只校验、保存并
应用物理参数，不再包含手柄灯位、颜色或强度比例等画面风格默认值。环境曝光、天空盒亮度
和 HDR 背景曝光仍由现有 Profile/Vulkan 路径动态加载。

动态屏幕补光现复用输出帧已有的 `screen_light_linear_rgb`：默认以 12 Hz 异步采样并在
Presenter 每帧指数平滑，按亮度上限和饱和度生成方向光颜色/lux。屏幕与手柄位置仅用于
归一化入射方向，距离不参与强度或衰减计算。光源仅加入 `foreground_scene` 的手柄 light channel，因此环境和
Glow 不受影响。开关、lux、采样率、平滑时间、最大亮度、饱和度和 shadow 全部由
`common.json`、环境 profile 或 lighting preset 提供。

完成条件：环境、手柄模型、按键动画和深度遮挡正确；屏幕光关闭后画面仍符合基准。

当前阻塞：现有 Filament external swapchain 只接收颜色 `VkImage`，尚未导出环境/手柄
深度资源，因此 `left_depth`/`right_depth` 目前为空，Composer 不启用深度敏感激光。
必须先在 native producer 完成深度附件生命周期、layout 和 ready timeline 交接，再进入
激光根部遮挡实机验收。

已确认 pinned Filament Vulkan backend 的实际接入点是
`VulkanPlatform::SwapChainBundle.depth/depthFormat`。后续实现必须填充该 backend
attachment 并交接 layout；仅增加 C ABI 句柄而不进入 `SwapChainBundle` 不算完成。

本阶段已加入 `filament_bridge_create_eye_swapchain_with_depth` 和对应 Python wrapper，
可在创建 eye swapchain 时把借用的深度 `VkImage` 注入 `SwapChainBundle`。由于当前
Presenter 尚未创建并发布这样的深度图，能力探针仍返回 false，旧路径行为不变。

Presenter 侧已加入有界 `VulkanDepthAttachment` 创建和销毁逻辑：只有 native depth
swapchain ABI 存在时才创建每眼单样本深度图并注入；旧版 bridge 或格式不支持时自动
回到颜色-only swapchain，不改变现有运行路径。该附件目前服务于 Filament producer，
尚未被 Projection Composer 作为激光深度输入使用。

native bridge 另提供 `filament_bridge_get_depth_attachment` 只读查询，用于确认每眼
深度句柄和格式确实已绑定；查询成功不等于 Composer 已完成深度测试，后者仍要求
Projection RenderPass/Framebuffer 使用同一深度资源并完成 producer-to-presenter 的
layout/timeline 交接。

Presenter 现在已提供两段式深度状态交接：Filament 完成后先登记
`DEPTH_STENCIL_ATTACHMENT_OPTIMAL` producer 状态，未来 Composer 提交再通过完成
semaphore 调用 `prepare_external_depth_for_sampling` 转为
`SHADER_READ_ONLY_OPTIMAL`。在激光 descriptor、shader 和坐标比较尚未完成前，
该转换接口不自动调用，避免改变现有激光显示行为。

当前合成顺序已接入颜色 producer：非 multiview 的每眼 Filament swapchain 在
Projection Composer 前完成环境/手柄绘制，Composer 通过 render-finished semaphore
等待后以 `LOAD` 方式叠加 SBS、Glow 和其它 Vulkan overlay。multiview 使用三缓冲的
私有 `R16G16B16A16_SFLOAT` 双层图像作为 Filament producer；Vulkan 等待 producer
完成点后逐层应用与旧前景 View 等价的 LINEAR exposure/clamp 输出，再写入 Virtual
Desktop 兼容的两只逐眼 OpenXR Projection swapchain。SBS/Glow 随后以 `LOAD` 方式
叠加，避免 Filament multiview 禁用后处理时直接写入 8-bit sRGB 目标造成高光细节丢失。

手柄优先级隔离使用 post-Composer 前景 pass：环境仍在 Composer 前由主 View
输出；屏幕和 Glow 完成后，Bridge 仅复用现有 `controller_view` 与
`controller_guide_view`，保留当前颜色并清理前景深度，再绘制手柄、激光和指南。
该路径已经通过头显验收并成为普通 OpenXR Vulkan Composer 的默认行为；如需回退诊断，
可显式设置 `D2S_FILAMENT_CONTROLLER_OVERLAY_AFTER_COMPOSER=0`，恢复 Composer 前绘制。

### 阶段 4：迁移激光和 Glow

Surround Glow 的四条完整 shell 边共享角点，但不再使用加法混合重复累计角区亮度；重叠区改为逐通道最大值混合，
既不会由后绘制边覆盖前一条边，也不会因端点透明而露出清屏背景形成黑边；
每边网格由迁移初期的 96 x 48 精简为 48 x 24，使双眼每帧顶点数由 221184
降为 55296，同时保留低频模糊光场所需的曲面平滑度，并避免头部移动时因角点
覆盖变化产生高亮闪烁。

Glow 纹理更新频率由独立 Profile 字段 `glow_sample_hz` 控制，默认 30 Hz；手柄
屏幕补光继续使用 `controller_screen_light_sample_hz=12`。两条路径不再共用节流
参数，避免 Surround 在约 30 FPS 的显示中每 2 至 3 帧阶梯式更换整块边缘颜色。
`glow_smoothing_seconds` 默认 0.10 秒；Glow compute pass 在 GPU 持久历史缓冲中
对新旧线性颜色执行指数时间融合，再写入当前外部纹理。首帧和效果模式切换会重置
历史，屏幕补光的只读采样不会修改 Glow 历史。

激光：

- 复用 Python Presenter 的 Aim pose、滤波和逐手显隐状态。
- 将交叉锥形面、渐变和深度测试改为 Vulkan pipeline。
- 使用 Filament 输出的手柄深度遮挡激光根部。

Glow：

- 保留 `src/stereo_runtime/vulkan_glow_source_pass.py` 的 Vulkan Compute 预处理。
- 删除 Glow 外部图像导入 Filament 的步骤。
- Vulkan Projection Composer 直接采样 Glow 图像并完成合成。
- Glow 不参与手柄照明，也不改变基础环境光、头顶光或独立的屏幕补光。

当前进度：Glow Compute 输出已标记为 `vulkan_projection_composer` 所有的
latest-frame 资源，Presenter 不再用 Filament Glow 外部图像 ABI 判断该资源是否可用。
主 SBS RenderPass 保持不透明绘制；此前将 Glow 图像覆盖整块 SBS 的临时 overlay 已删除。
五态显示逻辑已按删除前 `bridge_glow.cpp` 迁入 Vulkan Graphics：`Glow/Glow2` 使用随屏幕移动、旋转、缩放和曲率更新的 64 段内外边缘几何；`Veil/Frosted` 使用原 8×8 四壁体积或 64 段曲面边界；`Surround` 使用四条 48×96 测地条带连接屏幕边缘与头部椭球光壳。`Glow/Glow2/Veil/Frosted` 在不透明 SBS 后通过 `LOAD` RenderPass 绘制；`Surround` 保持旧 Filament 的背景 Scene 顺序，先清屏并绘制加法光壳，再由 `LOAD` RenderPass 绘制不透明 SBS 将屏幕范围遮住。Vulkan Compute 仍只负责 Glow 源纹理预处理和区域平均，不重复实现显示几何。原范围、采样、衰减、噪声、混合、头部位置、默认环境和透视背景可见性条件均保留；每 descriptor slot 使用独立有界状态缓冲，禁止整屏覆盖和无界资源累积。Presenter 已不再调用 Filament Glow 更新函数，旧 `filament_bridge_set_glow_*` ABI、native Glow 模块和 Python ctypes 绑定均已删除。`openxr_vulkan_composer_glow` 用于确认成功提交。

完成条件：激光遮挡、Glow 五态和 latest-frame 复用正确；Glow 线程不阻塞主屏幕。

当前激光迁移进度：按照兼容性要求，手柄模型、手柄动画、环境模型和激光继续由
Filament 统一渲染；Projection Composer 只等待 Filament 完成 semaphore，并把同一
左右眼 per-eye swapchain 结果作为环境/手柄/激光底图，再叠加 SBS 和 Vulkan Glow。
Vulkan 激光深度采样实验保持关闭，现有 Filament Bridge 激光 ABI 暂不删除。

### 阶段 5：Tool Quad 与两处 2D 光圈

Quad Layer 内容：

- FPS 面板。
- 操作指南。
- 固定在虚拟屏幕上的 2D 光圈。
- 固定在虚拟键盘上的 2D 光圈。
- 虚拟键盘。

内容生产复用 Vulkan MSDF/纹理上传路径。不得为主屏幕创建双眼 Quad swapchain。Tool Quad
如需双眼不同内容，必须在目标 runtime 上单独验证；当前 Virtual Desktop 部署只使用已验证
的单图 Tool Quad。历史 Quad array 实验不再是验收项：

```text
Quad Left  -> imageArrayIndex=0, eyeVisibility=LEFT
Quad Right -> imageArrayIndex=1, eyeVisibility=RIGHT
```

如果两眼内容相同，使用一个 `array_size=1` Quad、`eyeVisibility=BOTH`，不强制使用 array。

完成条件：Tool Quad 不影响 Projection 主路径；Virtual Desktop 主屏幕始终通过 Projection
array/per-eye swapchain 正确显示。

### 阶段 6：删除剩余 Filament 显示代码

阶段 3 已删除 Filament 虚拟屏幕，阶段 4 已删除 Filament Glow 显示代码。剩余效果迁移完成后，删除或停用：

- `native/filament/bridge/bridge_laser.cpp/.h`
- `filament_bridge_set_controller_laser`
- Glow 相关 ABI、字段和测试已删除；激光迁移完成后再删除 `bridge_laser.cpp/.h`

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
5. Screen Quad Reprojection 已停用；其环境变量不得影响 Projection。

## 7. 验收门槛

- 左眼/右眼颜色和屏幕纹理顺序正确。
- `array_size=2` Projection 成功，per-eye fallback 成功。
- 屏幕清晰化视觉回归通过。
- 环境 GLB、手柄 GLB、按键动画和手柄深度正确。
- 激光被手柄几何自然遮挡。
- Glow 不阻塞视频帧率，资源租约无覆盖和泄漏。
- 两处 2D 光圈和虚拟键盘在 Quad Layer 正确显示。
- 手柄接受基础环境光、Head/Top 灯和有界屏幕平均色补光；环境与 Glow 不得接收屏幕补光。
- `D2S_OPENXR_SCREEN_QUAD_REPROJECTION=1` 不会激活主屏幕 Quad；日志明确说明继续使用 Projection。
- 连续运行至少 10 分钟，无 `VK_ERROR_DEVICE_LOST`、validation error 或持续帧率下降。

## 8. 回退原则

- 每个阶段单独提交和实机验证。
- 新 Vulkan Composer 失败时只回退环境/手柄或清屏，禁止重新引入 Filament 或 Quad 主屏幕。
- Projection array 失败只回退 per-eye，不回退到 Quad Layer。
- Screen Quad Reprojection 已停用；其环境变量不得影响 Projection。
- 任何同步能力不足优先丢弃新帧，禁止阻塞或回读 CPU。
