# Desktop2Stereo Vulkan 项目日志

本文件只记录用户可感知的功能、行为变化、重要修复和架构里程碑，不记录逐次调试过程。新记录按日期倒序追加，并将同一目标的连续修改归纳为一条有效结果。

## 2026-08-10

### Vulkan / Filament multiview 验证状态

- **已验证 VDXR array layer 路由：** 一个 `array_size=2` 的 OpenXR Projection swapchain，在 layer 0 独立清为红色、layer 1 独立清为绿色，头显左眼显示红色、右眼显示绿色。这证明 VDXR 接受 `imageArrayIndex=0/1`，并排除了闭源 Virtual Desktop 服务端/客户端导致当前双眼同画面的可能性。
- **已验证 Filament layered 配置，但立体结果尚未通过：** Bridge 创建一个双层外部 swapchain，Engine 报告双眼 multiview stereo，Vulkan render pass 报告 `viewMask=0x3`，手柄、激光和指南程序均包含 Vulkan `ViewIndex` 内建变量。手柄模型加载仍确认正常，但双眼都显示红色的 eye-0 诊断画面，与此前手柄图像呈二维的观察一致。
- **修复并重新验证 Filament 内部深度清除材质：** Filament 1.75 生成了 multiview `clearDepth` 包，却注册了它的 instanced 包。修补后的后端现在为 multiview Engine 选择 `CLEARDEPTH_MULTIVIEW`，`compiled type 1 / engine type 2` 警告已消失。重新构建后双眼仍为红色，证明该警告确实是不兼容问题，但并非立体路由异常的唯一原因。
- **不再将 swapchain 回读作为证据：** 缩放图像 blit、精确 linear image copy 和直接 image-to-buffer copy 在头显同时显示红色时，对两个 OpenXR array layer 的回读结果均为黑色。这些捕获不能反映可见 swapchain 内容；在单独修复外部图像所有权/回读边界之前，不再用它们判断 Filament 是否写入 layer 1。
- **将故障边界下移到 Filament 以下：** 新增纯 Vulkan 诊断，绕过 Filament、GLB 模型、SBS、Glow、推理合成和材质辅助逻辑。单次全屏 multiview draw 使用 `viewMask=0x3`，并在 fragment SPIR-V 中直接读取 `gl_ViewIndex`，但实机结果仍是双眼红色。因此，在解释清楚 Vulkan 核心 multiview 执行行为之前，暂停进一步修改 Filament。
- **纯 Vulkan multiview 已通过实机验证：** 头显显示左眼红色、右眼绿色，主机可见计数器记录 `fragment_counts=(1, 1)`。这证明两个 view 均正确执行，`gl_ViewIndex`、`viewMask=0x3`、双层 attachment 写入以及 OpenXR `imageArrayIndex=0/1` 提交均正常，故障边界已收敛到 Filament 内部 shader/立体 variant 路径。
- **Filament fragment 取眼索引路径已排除：** 实机 trace 显示 `D2S Controller Eye Diagnostic` 为 `multiview=1 vertexViewIndex=1 fragmentViewIndex=0`，且 `fragmentVariant=(none)`，头显仍为双眼红色。这证明 Filament 1.75 的 stereo variant 只作用于 vertex stage，fragment 中直接调用 `getEyeIndex()` 会退化为常量 0，不能用来修复或验证正常 multiview 路径。
- **纯 Vulkan vertex-stage 路径也已通过：** vertex shader 读取 `gl_ViewIndex` 并通过 flat varying 传给 fragment 后，实机仍显示左红右绿且 `fragment_counts=(1, 1)`。因此 vertex-stage view index、跨阶段 flat varying、multiview render pass 和 OpenXR layered 提交都正常，问题进一步收敛到 Filament 自身。
- **当前下一项测量：** Filament 手柄诊断只在 vertex stage 使用 `getEyeIndex()`：eye 0 保留纯红手柄，eye 1 将手柄顶点移出视野，完全不依赖 fragment eye index 或 CUSTOM0 varying；同时一次性记录 Bridge 实际收到的左右眼相对平移和 frustum。右眼手柄消失说明 Filament vertex 眼索引有效，应继续检查相机 uniform；双眼仍显示红色手柄则证明 Filament vertex shader 运行时始终得到 eye 0。
- **后续验证顺序：** 先让纯 Vulkan 测试生成并计数不同的红/绿 view，再重复 Filament 手柄双眼诊断；仅当两者都通过后，才将独立 Filament 离屏 producer 接入正常屏幕/Glow 合成。此前会触发 device lost 的双 SwapChain deferred 路径继续保持禁用，不属于本次实验范围。

## 2026-08-09
- 新增纯 Vulkan multiview 双眼诊断，绕过 Filament、模型、SBS 和 Glow：一个 `viewMask=0x3` 的 render pass 向 OpenXR `array_size=2` swapchain 绘制全屏三角形，并直接根据 `gl_ViewIndex` 选择红色或绿色。一个主机可见的单像素计数器记录 fragment invocation 是否实际为 view index 0 和 1 执行，不依赖 OpenXR 图像回读。在普通逐层 array 路由通过、但 Filament 手柄输出仍为双眼红色后，该测试将 Vulkan multiview 执行与 Filament 隔离开来。
- 新增隔离的 VDXR Projection array 能力测试：一个 `array_size=2` 的 Vulkan Projection swapchain 将纯红色 layer 0 提交给左眼 view，将纯绿色 layer 1 提交给右眼 view，完全不加载捕捉、推理、Filament、屏幕、Glow 或手柄。专用 PowerShell 启动脚本用于验证后续 Filament multiview/离屏合成工作的准确前置条件。
- VDXR array 路由实机通过后，新增下一项隔离的 Filament multiview 门控测试：可选启动脚本通过一次立体 Filament 提交，将配置的环境和手柄渲染到一个 `array_size=2` 的 Projection swapchain，同时关闭 Vulkan SBS/Glow 合成。在该 layered Filament 输出通过目视验证前，正常逐眼 producer 仍为默认路径。
- 修复仅 Filament multiview 诊断的输出租约：未使用的 SBS 帧现在会被释放，不再成为显示帧；CUDA/Vulkan adapter 也不再为从未进入源图像采样的眼睛索引 release semaphore slot。
- 新增可选的 Filament 手柄 multiview 诊断材质：将每个手柄 GLB primitive 替换为由 `getEyeIndex()` 选择的左眼纯红、右眼纯绿，从而把 shader view-index 路由与手柄纹理、光照和环境渲染隔离。眼睛索引现在在 Filament 支持的 vertex stage 中读取，并通过自定义 interpolant 传递给 fragment stage；启动脚本同时启用后端 stereo tracing，可在一次运行中关联 View 状态、Vulkan render pass view mask、shader `ViewIndex` 和头显结果。
- 为 Filament multiview 手柄诊断新增一次性 GPU 回读。它并排保存实际 array layer 0 和 layer 1 输出，并记录每层非背景区域的平均 RGB，在不改变持续渲染路径的前提下区分 Filament 渲染故障与 OpenXR array-layer 提交故障。诊断会将每层复制到紧密排列的主机可见缓冲，并直接等待 Filament finished semaphore，避开 Windows Vulkan 驱动上不可靠的 optimal-to-linear 图像复制和含义不明确的中间同步。
- 修复 Filament 1.75 为 multiview Engine 选择内部 `clearDepth` 材质的问题：修补后的后端现在注册生成的 `CLEARDEPTH_MULTIVIEW` 包，而不是不兼容的 instanced 包，消除了 layered 手柄测试中的 stereo type 不匹配，并使深度清除几何与默认材质、skybox 遵循相同的 multiview 契约。
- 在不改变合成顺序的前提下减少 OpenXR Projection 工作量：当重新构建的 Filament Bridge 提供新的 background-frame ABI 时，Composer 前置 pass 只渲染环境和普通前景，手柄和指南仅由现有 Composer 后置 overlay 渲染，不再逐眼重复绘制。手柄和不依赖深度的指南现在共享一个全分辨率 overlay View，在保留指南优先级的同时省去另一轮逐眼后处理 pass。旧版 Bridge 二进制继续使用原有兼容路径。
- 修复补洞模式实时切换，使内部总开关随所选模式同步变化：`none` 禁用补洞，`balanced` 和 `quality` 无需重启进程即可立即恢复边缘感知路径。运行时比较补洞性能时，模式、半径、强度和实际执行的 kernel 现在保持一致。
- 在 GPU 缓冲处理和 raw queue 投递之前增加软件 pacing gate，修复 WindowsCaptureCUDA 对目标 FPS 的执行。该机制补偿高刷新率显示器上忽略 `minimum_update_interval` 的捕捉后端：选择 60 FPS 后，每秒约接收 60 个源帧，而不是处理全部 120 Hz 回调；“自动”仍跟随检测到的显示器刷新率，设置 `D2S_WGC_SOFTWARE_THROTTLE=0` 可关闭新增门控以进行对照测试。
- 新增感知渲染状态的 OpenXR 推理准入控制：配置的两路/三路 TensorRT worker 池仍保持可用，但 Presenter 正在合成或队列中已有 SBS 结果时，最多只允许一个深度任务处于 in-flight。Presenter 缺帧时会自动恢复额外 worker，在保持突发吞吐量的同时避免持续挤占 Vulkan/Filament；现有最新帧策略和显式 worker 选择保持不变。

## 2026-08-08
- 修正 OpenXR FPS 面板：SBS FPS 现在统计 Presenter 实际消费的唯一立体帧，而不是速度更快的推理 producer 帧率。OpenXR Vulkan command ring 默认改为九个 slot（可通过 `D2S_OPENXR_VULKAN_FRAME_CONTEXTS` 覆盖），使屏幕、Glow 和手柄/深度的多 pass 提交能够跨越三个 swapchain image，避免过早复用三 slot fence 而阻塞 Presenter。
- 新增可选的延迟 SBS pacing 捕获，用于帧时序诊断。经过 15 秒播放准备窗口后，它会在 Projection 合成前记录 300 个新左右眼 Vulkan 立体输出的元数据，同时只截取六张均匀分布的 SBS 截图；稀疏 GPU 降采样回读使用有界三 slot ring 和后台 PNG writer，避免逐帧回读造成严重吞吐量失真。JSON manifest 会记录源 frame ID、时间戳以及跳过或失败的图像；未使用诊断启动脚本时，正常推理和显示行为不变。
- 通过四条完整外壳边缘之间的逐分量最大值混合稳定 Surround Glow 转角：重叠区域既不会累加亮度，也不会被后绘制边缘覆盖，任何渐隐几何都不会露出已清除背景形成黑缝。每条边缘网格也由 96×48 降至 48×24，使每帧立体几何顶点数从 221184 降至 55296。Glow 现在具有由 profile 驱动的独立 30 Hz 更新率和 0.10 秒 GPU 时域历史混合，用渐进的线性颜色过渡替代可见的 2～3 帧颜色跳变，同时不影响手柄灯光采样。
- 将已验证的 Composer 后置手柄 pass 提升为 OpenXR Vulkan 正常默认路径，无需诊断启动脚本即可保持最终“环境 → 屏幕/Glow → 手柄/激光/指南”顺序；仍保留显式环境覆盖开关用于回退测试。
- 新增 profile 驱动的手柄屏幕反射光：现有异步 Vulkan/CUDA 线性屏幕颜色归约结果现在会驱动朝向手柄、经过平滑且受亮度限制的方向光。亮度只取决于屏幕亮度，与屏幕距离无关，并且只影响手柄前景光照通道，不影响房间或 Glow；所有采样、lux、饱和度、平滑和阴影设置仍位于 DLL 外部。
- 将 Filament 环境、HDR 手柄环境光以及 Head/Top 手柄灯光外观参数从 native DLL 移入共享/环境 profile。带版本的运行时 lighting ABI 现在接收最终 lux/candela、颜色、相对头显偏移、衰减和阴影标志，因此调整光照不再需要重新构建 Bridge。
- 重新平衡手柄照明：顶部灯作为 100% 主光，跟随头显的正面灯作为 70% 补光，在保持自然明暗和前景合成顺序的同时改善顶部表面与按键辨识度。
- 新增可选的 Composer 后置 Filament 前景 pass 和最小 LOD0 启动脚本：先渲染环境，再由 Vulkan 绘制屏幕/Glow，最后由现有手柄/激光/指南 View 渲染前景，以便在不重新加载模型的情况下验证前景优先级。

## 2026-08-07
- 修复 Vulkan Projection Composer 质量链与 Filament 环境及手柄输出的合成：质量 pass 现在会等待 Filament 完成，并在绘制 SBS 屏幕前通过 `LOAD` 保留颜色目标。
- 修复 Vulkan deferred compositor 的 OpenXR 3D Depth 开关和 Depth Strength 实时调整：现在每帧都会将当前运行时深度值传入 Vulkan stereo push constants，因此 `0.0` 会输出单目画面，手柄调整会在下一渲染帧生效。
- 移除存在冲突的 OpenXR Glow 连续调节快捷键；现有 3D Depth 控制保持不变。

## 2026-08-06
- Virtual Desktop 不支持 OpenXR Quad swapchain，移除 Screen Quad Reprojection 实验启动脚本；默认输出继续使用已验证的 Projection swapchain Vulkan Composer 路径，不影响左右眼 Projection Layer 提交。
- Vulkan Projection Composer 完成屏幕质量链：按输入/头显档位执行原生 LOD0→MIP、Lanczos2→RCAS→MIP 或 EASU→RCAS→MIP，再进行最终平面/曲面屏幕投影。质量链保持每个 OpenXR 帧实时执行，确保 MIP LOD、MIP 偏移和 RCAS 参数改动立即生效；提供关闭质量链的 LOD0 性能对比开关，但不降低输入或 swapchain 分辨率。
- 捕捉设备高级设置新增 Vulkan 屏幕采样实时参数：最小 LOD、最大 LOD、MIP 偏移和 RCAS 锐化，并提供中英文说明和建议值；默认 `max LOD=0.35`、`MIP bias=-0.35`、`RCAS=0.50`。
- 并行深度推理的 native TensorRT slot 改为独立 engine/runtime/context、CUDA stream、输入输出缓冲和完成 event，避免 Myelin graph 被多个 context 重复加载；创建失败或运行时压力过大自动退回安全单路。GUI 将并行推理提升为补洞模式下的常用选项，提供“单路推理 / 两路推理 / 三路推理”，默认两路，并将“显示高级立体参数”置于其右侧。
- 修复 Flet GUI 多次最大化/最小化后黑屏：窗口尺寸变化使用可取消的延迟双重刷新，避免恢复阶段丢失重绘。

## 2026-08-05
- Added a minimal OpenXR screen-Quad eye diagnostic: one `array_size=2` swapchain supplies red-left layer 0 and green-right layer 1, each with its matching eye visibility, so VDXR per-eye array-layer support can be verified before changing the runtime DLL.
- Fixed native TensorRT Myelin `enqueueV3` failures in OpenXR screen-quad reprojection: native TensorRT now uses one execution context, so the runtime stays on the safe single-depth-worker path instead of concurrently loading the same Myelin graph.
- OpenXR 并行推理实验完成实机验证：native TensorRT 现在使用两个独立 execution context、CUDA stream、输入/输出缓冲和完成 event；pipeline 在 TensorRT 按首帧尺寸延迟完成 engine 加载后自动创建两个深度 worker，按 `frame_id` 有界重排并保持补洞、temporal、OpenXR/Vulkan 提交单线程。GUI“并行推理”移至高级立体参数之后，默认开启且可手动关闭。RTX 2060 动态内容实测开启后 SBS/处理帧率较单路提升约 6~10 FPS；日志可通过 `rt_parallel_workers=2`、`rt_pending_limit=2` 和交替的 `rt_depth_slot=0/2`、`1/2` 验证实际双路执行。
- Vulkan Projection Composer 完成纯 Vulkan 图形管线迁移：平面和曲面虚拟屏幕现在以世界空间三角带直接光栅化到每眼 Projection swapchain，复用现有 zero-copy 左右眼纹理与同步契约；移除 Homography 中间图、矩形复制和 Filament 回退，OPAQUE 运行时启用实验开关后也保持纯 Vulkan 路径。
- 修复头部移动或屏幕穿过视锥边界时出现的透明矩形、闪烁、残影、丢屏和绕头旋转：每个 XR tick 清理已获取的双眼目标，由 Vulkan 固定功能完成近平面与 FOV 裁剪，屏幕离开视野后不再复用错误矩形或旧帧。
- 新增普通显示和左右眼红绿诊断启动脚本；shader 构建、清单与合规工作流现同时编译和校验 compute、vertex、fragment SPIR-V，便于独立验证 array layer、左右眼顺序及实际 `graphics_triangle_strip` 路径。
- 修复 FPS 面板 XR 速率被视图循环计数覆盖的问题：优先使用成功 `xr.end_frame` 的真实提交时间戳；仅在尚未完成两次 XR 提交时使用启动阶段回退计数，避免面板错误显示约 36 FPS。

## 2026-08-04
- 建立可选 Vulkan Projection Composer 实验边界与固定的 Projection→Quad 提交契约；增加一键启动和左右眼红绿诊断，确认 `array_size=2` 的 layer 0/1、左右眼资源顺序及源图 GPU 同步可用。早期 blit/Homography 实验实现已由 2026-08-05 的直接 Vulkan 图形管线替代。
- 修复 FPS 面板将 XR 消费帧率误当作 SBS 生产帧率的问题：XR FPS 统计实际 OpenXR 提交，SBS FPS 读取运行时生产速率，两者不再显示相同的错误值。
- 将 patched Filament Vulkan backend 从 v1.74.0 升级到 v1.75.0：同步更新三平台 release 资产校验、源码构建 ref、本地 SDK 路径与 BlueVK 固定哈希；现有外部 `VkImage`、多 SwapChain 状态和 layered array image-view 补丁已通过 v1.75.0 源码契约验证。
- 修复低分辨率 EASU 路径首帧黑屏：MIP 目标创建完成后不再错误标记为已生成，首次采样会先执行实际源图重建；同时为 EASU 无有效权重的边界像素回退到中心源样本，避免异常输入产生黑色输出。
- 修复采样策略在源图导入后才切换到 EASU 时的目标尺寸失配：切换 `upscale_scale` 或 `filter_scale` 会使旧中间纹理失效，下一帧按新策略重新创建并生成内容，避免 1K 输入出现黑屏或使用错误尺寸。
- 将屏幕采样矩阵改为按输入/头显档位选择主路径：低分辨率输入使用独立 `upscale_scale` 生成 2 倍目标纹理，GPU 执行 EASU → RCAS → MIP；同档位使用源图 → MIP，高分辨率输入继续使用 Lanczos2 → RCAS → MIP。新增 `filament_bridge_set_screen_upscale` ABI，旧 Bridge 缺少该符号时保留原兼容路径。
- 新增可选 Filament multiview 双眼诊断：设置 `D2S_FILAMENT_EYE_DIAGNOSTIC=1` 后，左眼输出红色、右眼输出绿色；默认显示路径不变。
- OpenXR multiview 视觉回归现可从同一个 `array_size=2` Projection SwapChain 分别导出 layer 0/1，稳定生成左右眼 Projection 截图与运行清单；实测定位并修复手柄和指南独立 View 丢失双眼视差的问题，multiview 现在通过同一个 foreground View 按既有渲染优先级一次输出屏幕、辉光、手柄和指南，旧双 SwapChain 路径保持原有分层渲染。

## 2026-08-03
- 修复 OpenXR Filament multiview 将虚拟屏幕、房间和手柄渲染成单眼 2D 画面的问题：相机现在按
  Filament 契约使用中心头部绝对姿态和左右眼相对姿态，保留真实 IPD 视差；Bridge ABI 升级后，
  旧二进制会自动回退逐眼渲染。同时限制 multiview 原生帧诊断仅输出前 8 帧，不再持续刷屏。
- OpenXR Projection 新增稳定的双眼一次提交路径：优先创建一个 `array_size=2` SwapChain，
  由单个 Filament multiview frame 同时写入左右 array layer，并在一次完成信号消费后提交
  两个 Projection View。仅在 Bridge ABI、GPU multiview 能力和双眼尺寸均满足时启用；
  layered 目标创建失败会只销毁临时资源并保留原双 SwapChain 逐眼路径，不再调用已证实
  不稳定的 `end_frame_deferred()` / `finish_frame_batch()`。远程 Filament SDK 构建现强制开启
  `FILAMENT_ENABLE_MULTIVIEW`，并随工作流配置变更失效旧缓存，避免 Engine 启动时解析缺失的
  内置 multiview 材质而异常退出。
- 禁用 OpenXR 双 SwapChain 的 Filament deferred batch 默认路径：实机确认该路径无论使用
  双 Renderer 还是共享 Renderer，都会在固定运行时间后于 `finish_frame_batch()` 内触发
  `VK_ERROR_DEVICE_LOST` 或原生 access violation。Projection 恢复逐眼完成后再切换
  SwapChain，并继续消费每眼 render-finished semaphore；真正的双眼一次提交改由后续
  array swapchain + multiview 实现。设备丢失后不再调用 `xrEndFrame` 覆盖首个 Vulkan 异常。
- 修复双眼 deferred batch 中仍存在的共享场景写入：右眼设置独立 Camera 时不再再次移动
  控制器共享灯光，避免 eye0 已入队后改写 Filament 场景状态并触发 Vulkan descriptor
  validation，最终随机演变为 `VK_ERROR_DEVICE_LOST`；共享灯光改为每帧仅随 eye0 更新一次，
  当前逐眼完成路径与 CUDA/Vulkan external semaphore 路径均保留该隔离修复。
- 修复 Filament 屏幕资源在双眼 deferred batch 中仍复用 Renderable/View，导致第二眼在
  第一眼 GPU 工作未完成时改写绑定并最终触发 `VK_ERROR_DEVICE_LOST`：最终屏幕与 MIP
  copy 的 material instance、Renderable entity 和 View 现全部按眼隔离，并由 layer mask
  固定到对应眼；该资源隔离继续用于当前逐眼安全路径和后续 multiview 实现。
- 修复 CUDA/Vulkan binary external semaphore 在环形输出 slot 多代复用时仍会触发
  `VK_ERROR_DEVICE_LOST`：跨 API ready/release 改为每 slot/eye 独立的 exportable
  timeline semaphore 和单调递增 generation；Filament visible 信号继续使用 Vulkan-only
  binary semaphore。CUDA wait、图像拷贝与下一次 ready signal 保持同 stream 异步顺序，
  不再用 `cudaStreamSynchronize()` 阻塞 Presenter；真实 RTX CUDA/Vulkan 300 帧循环通过。
- 修复 Output Worker 的 Glow 直接提交与 Presenter/Filament batch 并发访问同一 graphics
  `VkQueue` 导致的 device-lost；两条路径现在复用 VulkanContext 设备锁完成主机侧外部同步。
- 修复 `finish_frame_batch()` 永久卡死：复用源帧时不再重新 signal binary visible
  semaphore，Filament 直接采样已经就绪的外部图像，避免等待一个排在卡住队列后的
  signal 提交而触发 GPU/设备级死锁。
- 修复 Vulkan 优化后 XR/SBS 被拖到约 16 FPS 的屏幕 MIP 回归：虚拟屏幕源帧不变时
  Bridge 不再每帧重复执行 4K Lanczos/RCAS/MIP 生成，只有 `frame_id` 变化时才重建，
  静态画面保持原有清晰度，动态画面继续实时更新。
- 修复虚拟屏幕深度遮挡手柄：屏幕材质仍保持 Opaque 合成，但不再写入深度；
  控制器/激光 View 不受屏幕深度裁剪，保持手柄在最前，且不改变 View 架构。
- 修复 CUDA/Vulkan external semaphore 在静态源帧复用时的 binary semaphore 重复
  signal/wait：同一 frame/eye 复用只等待一次 producer-ready semaphore，但每个 XR tick
  重新 signal Filament 消费的 visible semaphore；同时恢复
  `D2S_ENABLE_CUDA_EXTERNAL_SEMAPHORE` 默认开启。
- 修复 OpenXR 未消费 Filament render-finished binary semaphore 导致的
  `VK_ERROR_DEVICE_LOST`：逐眼完成后读取各眼 SwapChain 的完成信号，使用一次 completion
  drain 消费并转换为 Vulkan timeline，再释放 zero-copy 源图。
- 修复 Filament Vulkan backend 对 OpenXR 左右眼两个 SwapChain 的单例默认 RenderTarget
  污染：`VulkanDriver::acquireNextSwapchainImage()` 现在记录当前绑定的 SwapChain/image，
  切换眼时先释放旧绑定再绑定当前眼，避免第二只眼误渲染到第一只眼的图像并触发
  `VK_ERROR_DEVICE_LOST`。
- 完成立体流水并行与同步收敛：每个 XR tick 产生的左右眼 Filament render-finished 二进制信号都会在同一次 Vulkan graphics 提交中各消费一次，再转换为 timeline 供 zero-copy 源图安全释放；静态复用帧和异常路径同样回收完成信号。满足双 TensorRT 隔离槽、Triton、无 Temporal 的路径允许两帧在途，使 CPU 入队与上一帧 GPU 执行重叠。最高质量补洞不再为双眼复制 4K mask、depth 和 shift，并跳过无破洞像素的邻域采样；Vulkan 设备丢失后只清理 CPU 租约，不再向失效设备重复提交。性能诊断每 15 秒输出一次、累计 5 次，并新增 Filament 完成信号回收耗时；FPS 面板的 SBS 数值只统计唯一生产帧。
- OpenXR Vulkan 完成双眼提交和 Compute 流水并行化优化：Projection 路径先获取完整左右眼 swapchain 图像，再分别等待两眼就绪，避免右眼 acquire 被左眼 wait 串行阻塞；控制器场景状态和 GLB 动画改为每个 XR 帧只更新一次。Filament Bridge 新增兼容的 frame-wide 双眼提交 ABI，两眼仍在 Presenter 所在线程顺序生成渲染命令，但每眼只做非阻塞提交，待两眼全部入队后统一执行一次完成等待；旧 Bridge 二进制自动保留逐眼安全等待路径。Vulkan Stereo Compute 输入改为与 FrameContext 对齐的环形槽，CUDA 与 Vulkan 通过每槽外部信号量完成“可覆盖/输入就绪”双向 GPU 同步，不再逐帧主机等待；zero-copy 输出槽复用也只保留 GPU timeline 依赖。CUDA 单帧在途期间不再主动清空最新捕捉队列，生产者仍可覆盖旧帧，GPU 完成后可立即消费当时最新画面；OpenXR CUDA Triton 路径进一步启用双 TensorRT execution context、独立输入/输出缓冲和逐槽 CUDA Graph，在同一 CUDA stream 上允许两帧命令排队，使 CPU 准备与上一帧 GPU 执行重叠，同时避免双流争抢 TensorRT、Triton 和 Filament 的 GPU 资源。Vulkan deferred、Temporal、动态 convergence 或不具备隔离槽的 provider 自动保持单帧在途。FPSBreakdown 新增双眼 acquire/wait、Filament 每眼入队/延迟提交、整对完成等待、双眼 release、Vulkan 输入等待/上传和 TensorRT 执行槽统计。

## 2026-08-02
- 手柄新增动态屏幕反射光：复用现有 Vulkan Glow Compute 输入，在同一 GPU dispatch 中将屏幕 sRGB 内容归约为线性平均光色，仅异步读取 3 个浮点数而不回读画面；手柄 PBR 光照按屏幕光 80%、基础头灯/顶部灯/间接光 20% 分配，屏幕聚光灯直接使用采样色，不再额外混入固定白色。
- 新增 Vulkan `surround` 环绕辉光：以四组互不共享极点的“屏幕边缘→半球外圈”放射条带替代经纬半球网格，每个屏幕边缘采样点沿独立球面路径均匀向四周扩散，不再向半球上下左右中心点会聚；放射条带第一圈直接锚定虚拟屏幕四边的真实世界坐标，后续顶点才从屏幕深度逐步过渡到头部射线与远端椭球的交点，使双眼视差和头部平移时发射边界仍固定贴合屏幕，不再露出反向漂移的黑色矩形。GPU 沿屏幕四边划分 8×6 分段，并按输入分辨率从最外圈向内自适应采样约 4/8/16 像素宽的窄带，避免整块区域平均造成辉光与屏幕边缘颜色不一致；相邻分段平滑插值，屏幕原图不参与模糊或改色。Surround 使用加法发光合成，不产生遮挡方块或实体球体背景；取样颜色在 sRGB 感知域计算区域均值后仅解码一次进入 Filament 线性工作流，效果仍仅在 Default 环境显示。

## 2026-08-01
- 将补洞模式收敛为三个正式选项：`none`（关闭 / 不补洞）、`balanced`（均衡 / 标准）和 `quality`（增强 / 高质量）；删除没有独立 kernel 价值的 `soft_low_ghost` 与 `sharp_test` 选项、标签和适配映射，不保留旧模式专用兼容分支。
- 更新补洞 GUI 中文文案：原“内容感知 / 最高质量”改为“增强 / 高质量”；Tooltip 仅说明三档有效行为，并明确立体模式默认映射为电影→均衡、游戏→关闭、图片→增强/高质量。
- 同步 GUI 与运行时预设：电影使用 `balanced, radius=1, strength=0.6`；游戏使用 `none, radius=0, strength=0.0`；图片使用 `quality, radius=3, strength=1.0`。
- 为 Vulkan Compute 建立统一补洞三态 ABI：`BALANCED=0`、`QUALITY=1`、`NONE=2`。`d2s_stereo_fused`、`d2s_stereo_layered`、`d2s_stereo_layered_tiled` 和 OpenXR zero-copy 的 `d2s_stereo_layered_output` 均执行同一模式契约。
- Vulkan 最高质量补洞迁移完整 radius-3 方向内容感知公式：深度/位移方向可靠性判断、三点方向平均、方向候选与 box average 的 0.75/0.25 混合、UI 亮度边缘保护和深度边缘保护；关闭模式在 shader 主路径直接跳过遮挡、羽化和补洞邻域计算，并输出零 mask。
- 更新 Vulkan 运行时 debug 字段，统一报告 `vulkan_hole_fill_mode` 与实际 `hole_fill_backend`，避免配置显示为最高质量但执行均衡公式。
- 使用 Vulkan SDK 1.4.350.0 重新编译四个 SPIR-V；RTX 3090 实际 Vulkan 调度验证三态均可执行，关闭模式 `mask_max=0`，最高质量与均衡输出存在有效差异。
- 回归验证：专项测试 121 项通过，全量测试 `713 passed`；`git diff --check` 通过。

## 2026-07-31
- 修复静止源帧复用后 FPS 面板 SBS 计数归零的问题：SBS FPS 现在按实际 XR 显示 tick 统计，而不是按生产者 `frame_id` 去重；同时恢复前景 View 不执行二次后处理，避免前景合成阶段覆盖房间场景。
- 修复场景曝光更新和 Bridge 销毁时前景 View 残留旧 ColorGrading 句柄的问题，避免 Filament 访问已释放句柄导致原生崩溃。
- 修复前景 View 中手柄材质发白、反光碎片化并被误认为透明的问题：手柄、屏幕和激光所在的前景 View 现在绑定与房间 View 相同的 ColorGrading，并独立执行一次输出变换；两个 View 先后写入同一交换链并不构成同一像素的二次编码。
- 修复 `preview_room_layout.py` 中 `3d_bedroom` 材质大面积发黑的问题：桌面预览现在读取环境 profile 的 `env_ambient_color` 创建独立 Filament 间接光，并在未配置 `preview_exposure` 时回退使用 `env_exposure`，保留方向补光，避免无环境光导致背光材质全黑或明暗对比异常。
- 将 Filament 光照拆分为房间 Scene 与前景 Scene：房间 GLB 使用全局间接光，手柄、屏幕、激光和 UI 使用独立前景 View；`ambient_light_multiplier` 不再放大房间全局光，`controller_hdr_lighting` 现在实际控制前景 controller 间接光开关。前景 HDR IBL 资源仍保留 `hdr_ibl_pending_profile_fallback` 约束，待三平台 KTX IBL 接入后使用真实 HDR 预过滤环境。
- 前景 View 关闭后处理，仅在房间主 View 执行一次最终颜色变换，避免双 View 合成时房间颜色被二次 tone-map 或编码。
- 将 Filament 默认场景曝光和天空盒亮度从 `settings.yaml` 迁移到 `xr_viewer/environments/common.json`；环境 `profile.json` 仍可按环境覆盖，旧 YAML 字段仅保留兼容回退。
- 明确区分电影立体合成质量与补洞质量：`quality_4k` 继续表示 Cinema 的立体合成后端，选择“最高质量”补洞后统一记录为 `hole_fill_mode=quality`；启动/热切换日志和 15 秒 FPSBreakdown 现在同时输出补洞模式、半径和强度。
- 修复 Vulkan Filament 虚拟屏幕 MIP 采样缺少旧工程 `LOD_BIAS=-0.35` 的问题；最终屏幕采样现在使用与旧 OpenGL runtime eye 纹理一致的负 LOD 偏移，避免在相同屏幕 footprint 下过早选择较软的 MIP 级别。
- 保持 `filter_scale=1` 路径不执行 Lanczos2 和 RCAS，仅进行原图 LOD0 拷贝与动态 MIP 链生成；不修改颜色空间、输入分辨率或显示几何。
- 实机验证：MIP 路径文字边缘清晰度已接近旧工程 legacy 路径。
- 本地回归测试：`38 passed`；Filament Bridge Windows/Linux/macOS 三平台 GitHub Actions 构建成功，二进制已同步。

## 2026-07-30
- 修复 Lanczos2/RCAS 屏幕材质使用旧式 `materialParams_<name>` 访问导致 Filament shader 编译失败的问题，统一改为当前 Filament MaterialBuilder 要求的 `materialParams.<name>` 结构体访问。
- 删除正常运行路径中的 `07_filament_screen_*.png` 固定相机 readback/PNG 导出及对应 C ABI；屏幕显示只保留 GPU 采样、MIP 计数和同步路径，历史 artifact 仍可由离线脚本比较。
- 完整迁移 legacy 屏幕清晰化两级 GPU pass：第一 pass 为 4x4 Lanczos2 重建，第二 pass 为完整 FSR RCAS（luma 自适应、RGB limiter 和有界 sharpness），RCAS 输出再生成动态 MIP 链；不引入 CPU 像素往返或跨帧混合。
- 动态屏幕 MIP 优化与两级质量 pass 在每只眼的 Filament `begin_frame` 内执行，使用线性空间 sRGB 下采样、三线性过滤和 16x 各向异性过滤；规格书和需求矩阵同步更新。
- AMD FidelityFX SPD/Vulkan compute downsampler 暂不直接替换稳定路径，后续必须以相同 `07_filament_screen` 源一致性、计数和 heatmap 指标做 A/B 验证后再决定。
- 补齐虚拟屏幕动态 MIP 采样的可验证闭环：native Bridge 记录每眼外部源图像绑定次数和 MIP 生成次数，并通过 `filament_bridge_get_screen_sampling_stats` C ABI 暴露给 Python。
- `07_filament_screen_*.png` 捕获 manifest 现在写入 `screen_sampling_update=dynamic_per_frame_mip`、MIP 动态更新标记和每眼采样统计，便于确认每帧 `generateMipmaps()` 是否实际执行。
- 修复屏幕采样视觉回归脚本的 manifest 识别：优先读取 `screen_sampling_runtime_manifest.json`，保留旧 `visual_regression_runtime_manifest.json` 回退，legacy/mip 对比不再误判 07 捕获缺少配置。
- 已新增对应单元测试和 native 静态 ABI 断言；原生 Bridge 计数 ABI 需要 GitHub Actions 三平台远程构建后才能在实机日志/manifest 中出现真实数值。

## 2026-07-29
- 新增 GUI 头显型号驱动的 2K/4K/8K 屏幕采样档位，并按实际输入屏幕 `capture_size` 将 1K/2K/4K 输入映射到 2K/4K/8K 推荐头显。
- 非 16:9 输入仅按最长边近似归档，不裁剪、不拉伸实际源图像；匹配档位保持原始纹素 footprint，低档头显接收高档输入时才启用有界面积预滤。
- OpenXR runtime recommended extent 仅作为交换链上限，不再覆盖 GUI 头显选择；新增 Filament Bridge `set_screen_sampling` ABI。
- 验证：屏幕采样策略、头显预设、runtime 配置、OpenXR Vulkan 和 Filament Bridge 测试共 166 项通过；原生 Bridge 仍需 GitHub Actions 三平台远程构建。
- 已将上述输入/头显矩阵、非 16:9 归档、尺寸职责边界、预滤公式和 Bridge ABI 验收条件补入两份正式规格书及需求矩阵。
- 全量 Python 回归测试通过：680 passed；提交内容包含本轮规格书、采样策略、Bridge ABI、MSDF Quad 和既有未提交改动。
- 修复 `3d_bedroom/environment.glb` 的 glTF scene root：移除重复挂到 scene root 的非根节点，避免 Filament desktop preview 报 `Unable to parse glTF file`。
- 为旧房间 profile 增加 `view_pose_space=scene` 兼容：`3d_bedroom` 的座位坐标按 GLB 原始场景坐标解释，不再被 `model_position` 逆变换推到房间外；预览和 OpenXR profile 加载路径保持一致。
- 预览工具保存座位时同步遵守 `view_pose_space`，避免移动座位后把旧房间 profile 再写回成错误坐标。
- `npx --yes @gltf-transform/cli validate src/xr_viewer/environments/3d_bedroom/environment.glb`: `No errors found`。
- `src/python3/python.exe -m py_compile src/xr_viewer/preview_room_layout.py src/xr_viewer/core_openxr_vulkan.py` 通过。
- `3d_bedroom` 预览相机计算结果保持为 GLB scene 坐标 `[0.0018, 0.7381, 0.0202]`，不再变成 `[0.0018, 1.7381, 3.0202]`。

## 2026-07-27
- 修复 Requirements Compliance 在 Linux runner 上的 Windows 键盘状态机测试：
  `ctypes.windll` 仅在 Windows 存在，测试现在显式允许注入 FakeUser32，仍完整验证
  修饰键、普通按键和释放状态，不改变生产代码。
- The screen size/distance and preset OSD now decode the bundled MSDF atlas
  in a Vulkan compute shader, writing the glyph coverage and background into a
  Quad-sized storage image before copying it into one OpenXR Quad Layer. This
  removes the unreadable Projection/Quad split while keeping the OSD text on
  the real GPU MSDF path.
- Added a local MSDF JSON-coordinate preview tool so OSD layout can be checked
  before OpenXR hardware testing.
- Corrected MSDF V-coordinate adaptation for Filament's bottom-left texture
  convention and updated coverage sampling for the 2048x2048 atlas.
- 修复 MSDF Filament 材质编译失败：`sample` 在目标 GLSL 兼容编译器中是保留字，已改为 `msdf_sample`；三平台 Bridge 远程构建和二进制回写成功。
- 修复 MSDF 空页提交：零长度 NumPy 缓冲改用 `tobytes(order="C")`，并在首次提交失败后关闭 MSDF 路径，避免每帧重复报错刷屏并恢复旧 Quad Layer。
- Started the GPU text migration contract: imported the requested `3500.txt` UI charset (3,958 unique characters), defined paged MSDF atlas generation and shared linear atlas sampling requirements, and kept an explicit legacy Quad Layer fallback while the native Bridge ABI is being added.
- Generated the first three-page MSDF atlas from the complete UI charset; verified 3,959 glyph records and 2042x2032 atlas pages.
- Moved the MSDF runtime assets into `src/xr_viewer/fonts/` so packaged OpenXR runtime resources do not depend on repository-root asset paths.
- Added the first native MSDF text-overlay ABI: atlas pages are uploaded as
  linear GPU textures and packed glyph geometry is updated on the Presenter
  thread. Existing Quad Layer bitmap rendering remains the compatibility
  fallback until rebuilt Bridge artifacts are deployed.
- The Presenter uploads the shared MSDF atlas once to a resident Vulkan
  storage image and submits only changed glyph metrics to the compute pass.
  The native Filament MSDF Projection ABI is not used for this Quad-only text
  path; keyboard and laser cursor textures remain on their existing Quad path.
- OSD Quad canvases now size themselves from MSDF text advance and atlas line
  height, then preserve that aspect ratio while scaling with the virtual screen.
- FPS and both screen/controller operation-guide panels now submit their text
  as GPU MSDF runs into the same Quad-layer intermediate; keyboard and laser
  cursor textures remain unchanged.
- Menu and B overlay state machines are now mutually exclusive. The screen-side
  guide keeps the full current screen height, and its MSDF text uses the same
  canvas proportion so the guide content fills the background instead of
  becoming a tiny centered block.
- Restored live FPS-panel resolution values: XR now uses the first eye
  swapchain size and Screen uses the current per-eye output render size instead
  of the temporary `0x0` placeholders. Resolution changes invalidate the
  cached MSDF panel text.
- Restored the complete vertical Menu operation guide: the one-column MSDF
  layout now renders every legacy guide row instead of applying the controller
  panel's two-column split and dropping the second half.
- FPS now reads the runtime `depth_strength` metadata instead of displaying a
  hard-coded `0.00`. Depth adjustment, reset, and 2D/3D controller shortcuts
  also trigger a legacy-style Quad OSD for 2.5 seconds; the stereo toggle uses
  the legacy `3D mode on/off` message while depth changes show `Depth Strength`.
  Controller changes are reflected immediately from the accepted runtime target;
  older in-flight output frames cannot overwrite the new OSD value.
- Right-grip/right-stick screen distance and size controls now use only the
  hold-time acceleration curve; the generic guide dispatcher no longer adds a
  second fixed-speed delta in the same XR frame.
- Right-grip screen distance and size controls start at 0.10 m/s, accelerate
  linearly with hold time, and reach 10.0 m/s after five seconds. Releasing or
  reversing the stick resets the ramp for precise adjustment.

### 已实现

- 按旧工程恢复屏幕上方尺寸/距离 OSD：使用 512x78 深灰圆角面板、灰色标签、青色数值和 24px 字体，整组文字居中，并沿用屏幕宽度比例与顶部间距。
- 修正右 Grip + 右摇杆屏幕调节：恢复旧工程的 X 轴指数加速缩放，移除 22m 硬上限；Y 轴距离调节统一使用 0.35-3.0 m/s 的指数速度曲线。

- 修复 OpenXR 运动门控导致 SBS 更新过慢：对可复用的捕获/GPU 输入缓冲区建立独立运动采样快照，且恢复旧工程默认关闭 motion gate；输入未变化时复用上一帧仍可通过 `D2S_RUNTIME_MOTION_GATE=1` 显式启用。
- 将 OpenXR 未获得焦点时的手柄输入日志改为 `Controller input deferred`，避免把正常的 `SessionNotFocused` 状态误标为失败。
- 清理 CUDA external semaphore 正常状态日志：无实际错误时不再输出 `error=none`，仅在同步初始化确实失败时追加错误信息。
- 将 `FPSBreakdown` 从每秒输出改为稳定运行 15 秒后输出一次，避免持续刷屏且不影响内部统计。
- 将 Vulkan validation layer 重复的 descriptor binding 明细合并为一条摘要，避免启动日志输出几十行相同类型诊断。
- 将 FilamentBridge 每帧左右眼 `acquired/begin/end` 正常明细合并为一条摘要，避免持续输出渲染循环日志。
- 修复 OpenXR 屏幕分辨率诊断刷屏：只在源图像分辨率、交换链目标尺寸或 render size 发生变化时输出，头显位姿导致的投影 footprint 变化不再触发日志。
- 修正 OpenXR projection 输出的左右眼资源顺序；普通 SBS Vulkan 路径保持原有合成顺序，不复用 OpenXR 专用交换逻辑。
- 新增 `d2s_stereo_layered_tiled.comp/.spv` 作为 `quality_4k` 的并行 tiled reference shader：保持原有五个 storage buffer、76 字节 push constant 和立体合成公式，仅将深度邻域缓存到 workgroup shared memory，供 warp、遮挡、羽化和补洞采样复用。
- `VulkanStereoComputeBackend` 新增可选 `layered_shader_path`，默认仍使用 `d2s_stereo_layered.spv`；tiled shader 当前只作为对照和性能基准路径，不改变生产默认选择。
- 将 tiled reference shader 登记到 `shaders/manifest.json`，并通过 `glslc` 编译和 `spirv-val` 校验。

### 验证结果

- `tests/test_pipeline.py`: `8 passed`；Vulkan/CUDA 相关回归测试：`32 passed, 2 warnings`；`py_compile` 和 `git diff --check` 通过。
- 3840×2160 同参数对照：原 layered shader 与 tiled reference 的左右眼及遮挡 mask 最大绝对差异均为 `0`。
- Vulkan 相关回归测试：`21 passed`；此前扩展的 OpenXR、runtime、synthesis 和 Vulkan 集成测试：`109 passed`。

### 未决事项

- tiled reference 尚未替换生产 shader；下一步使用稀疏破洞、密集边缘和真实深度帧分别比较 GPU dispatch 时间、画质和端到端 FPS，再决定是否进入生产路径。

## 2026-07-26

### 已实现

- 按旧工程恢复手柄屏幕操作：右 Grip+右摇杆 Y 沿头部到屏幕的径向距离移动，并使用旧工程指数加速曲线（0.35–3.0 m/s、死区 0.15）；左 Grip 按下时记录手柄旋转锚点，腕部旋转超过 45° 后将屏幕 Roll 吸附旋转 90°。
- 修正 CUDA 外部 semaphore 输出的 Filament 诊断元数据：明确报告 `vulkan_readback=none` 和 `vulkan_output_path=presenter_owned_storage_image`，避免直连路径已激活却显示为 `missing`。
- 修正右 Grip+右摇杆前后方向：按当前 Vulkan 输入层的 Y 轴约定恢复旧工程符号，摇杆向前时屏幕远离头部，向后时靠近头部。
- 修正 Vulkan Compute 零拷贝未生成 request 的条件：移除已完成分层视差迁移后仍残留的 `layered_parallax_not_supported` 守门条件；即使普通 Vulkan backend 已经初始化，OpenXR 也优先生成 Presenter-owned `vulkan_compute_request`。
- 修复 Vulkan host fallback 对 OpenXR `HxWx4` 眼图的尺寸识别，将 `(2160, 3840, 4)` 正确解析为 `3840x2160`，避免输出转换异常退出。
- 修复 OpenXR cinema 全合成路径丢失 `vulkan_compute_request`：运行时不再先走 `process_rgb_frame` 再包装 OpenXR 结果，统一通过 `process_openxr_frame` 保留 Presenter-owned Vulkan zero-copy request，避免每帧约 1 秒的 host-visible 回读。
- 为 zero-copy 实机验证固定运行条件：`src/settings.yaml` 使用 `Stereo Compute Backend: vulkan`、关闭 `Temporal`/`Auto Scene Reset`，OpenXR 预变形默认开启；Presenter 状态日志现在直接打印 `vulkan_readback`、`vulkan_output_path`、`vulkan_output_sync` 和 `active`，便于确认是否真正进入 Presenter-owned Vulkan image 路径。
- 收紧 Filament 外部屏幕图像路径的运行时诊断：CUDA producer 现在发布外部 semaphore 请求是否被环境变量或 Bridge ABI 阻止、初始化异常及最终 active 状态；Presenter 记录 direct screen path 的具体回退原因，不再把 zero-copy 未启动静默表现成普通 GPU copy。
- 修复 Filament 直接采样源图像的租约生命周期：显示中的 Vulkan ring slot 会保持到新帧替换或关闭，环形缓冲即将复用该 slot 时由 Presenter 先完成 finished semaphore 后再释放，避免 Filament 仍在采样时被 CUDA 覆盖而产生模糊。
- 修正 Vulkan zero-copy 投影屏幕发白：Compute 输出是 `VK_FORMAT_R8G8B8A8_UNORM` storage image、Filament 外部纹理是线性 `RGBA8`，shader 现在先将显示参考 sRGB 输入解码到线性光再执行 warp/补洞和写入；OpenXR sRGB 目标仍只做最终一次编码。
- 优化 4K layered shader 的安全路径：移除未使用的重复 `occlusion_at` 计算，遮挡搜索命中 edge 后提前结束，并在 `mask==0` 时跳过 `box_average`/方向补洞；有洞像素的 warp、补洞窗口和 blend 公式不变。当前中心+四方向羽化采样属于有边缘画质风险的近似，仍需实机重点检查斜向破洞。
- 实测 RTX 4K 后撤回 Triton 的简单 mask-predicated 补洞改法：无洞、10% 破洞、全破洞分别约为原 kernel 的 `0.92x`、`0.38x`、`0.83x`，额外分支控制抵消了屏蔽加载收益；Triton 保持原 kernel，后续若优化需采用稀疏索引/专用 active-pixel 两阶段方案并重新基准。
- 恢复旧工程的头显推荐屏幕几何：OpenXR Link 读取 GUI 的 `XR Headset Model`，按对应最佳观看距离和 60° 水平视场自动计算 16:9 屏幕宽高；Pico 4 / 4 Ultra 为 `23.09m × 12.99m @ 20m`，不再使用固定的 `16m × 9m @ 16m`。
- 对齐旧工程 OpenXR 工具交互：菜单键循环为“FPS → FPS+屏幕左侧竖向操作指南 → 全隐藏”，屏幕指南改用 `build_team_help_rgba`，不再误用手柄双列指南。
- 恢复旧工程键盘头向和激光命中点拖屏逻辑：键盘每帧以头部为目标重新朝向，单手 Grip 保持激光命中的屏幕局部锚点，手柄平移或旋转都能拖动屏幕；补回激光终点光圈显示。
- 输入路径异常现在只做一次可见日志，不再静默吞掉键盘、拖屏和快捷键更新失败。
- 修正右手 Grip 旋转屏幕的回归：旧工程 `openxr_right_grip_screen_rotation` 默认关闭，Vulkan 端现在明确禁止右手腕部旋转屏幕，仅保留旧工程允许的左 Grip 旋转和左 Grip+右摇杆旋转。
- 恢复旧工程右 Grip 单手拖屏的球面轨道：屏幕围绕头部保持固定距离移动，并在轨道移动后重新朝向头部。
- 补齐 B 长按三态循环：隐藏 → 手柄 FPS → 手柄 FPS+手柄操作指南 → 隐藏；不再把 B 长按错误实现为二态切换。
- 建立 `docs/05-openxr-behavior-migration-matrix.md`，按旧工程函数、Vulkan 函数、渲染层和验证方式登记 OpenXR 行为迁移状态；后续迁移项必须同时补自动对照测试和头显验证项。
- 按旧工程恢复 reference-space change pending 处理：运行时重定位后重建共享基础空间，重新应用 profile pose，并清空旧头部缓存，避免屏幕、手柄和投影视图使用不同坐标系。
- 按旧工程补齐曲面屏圆柱射线求交和 UV 到曲面世界坐标转换；左 Grip 同时支持旧工程的左右摇杆旋转，键盘轨道不再错误要求 stick click。
- 修正平面屏激光命中的旧工程 UV 方向：下边缘为 `v=0`、上边缘为 `v=1`，命中点拖动和 Quad 光圈与实际屏幕纹理保持同一上下方向。
- 恢复旧工程激光屏幕边缘吸附：平滑射线越过有限屏幕、原始姿态射线也未命中但仍处于边缘 6° 释放锥内时，使用无限屏幕平面 UV 夹到最近边缘，并将激光和交互命中保持在该边缘。
- 新增第一版 Vulkan Compute 立体合成融合 pass：单次 dispatch 完成视差计算、左右眼水平 warp、遮挡边缘膨胀、方向感知补洞和边缘保护；深度模型推理路径保持由各厂商后端负责。
- 新增 `vulkan_stereo_benchmark.py` 和 Vulkan smoke 校验；Windows RTX 3090 的 3840×2160 初测为约 `31.34 ms / 31.9 FPS`，同机现有 CUDA/Triton `fast_plus` 端到端初测约 `26.29 ms / 38.0 FPS`。该结果是首版融合计算对比，尚不代表最终零拷贝端到端性能。
- 明确立体合成后端选择：`auto/vendor` 先做真实 Triton kernel 探测；NVIDIA 走 CUDA Triton，AMD 走 ROCm/Windows Triton，只有探测失败或厂商不是 NVIDIA/AMD 时才走 Vulkan Compute。显式 `vulkan` 可在 NVIDIA、AMD、Intel 全部使用 Vulkan Compute，且不改变深度模型推理后端。
- 统一 Triton 运行时门控：视差、warp、遮挡、补洞、时域和输出阶段不再直接用 `is_cuda` 判断厂商，统一读取 GPU vendor；NVIDIA 与 AMD 使用同一套 Triton kernel 源码，Intel 等其它厂商不会误进入 Triton。
- CUDA 12.8 profile 改为稳定配套：PyTorch `2.11.0`、torchvision `0.26.0`、Linux Triton 3.6 系列和 Windows `triton-windows==3.6.0.post26`；AMD ROCm7 profile 保留独立 nightly 配套版本。Torch/Triton 升级需要重新验证 TensorRT、深度推理和 Triton kernel。
- 实际升级嵌入式 Python 到 `torch==2.11.0+cu128`、`torchvision==0.26.0+cu128`、`triton-windows==3.6.0.post26`；RTX 3090 / CUDA 12.8 / TensorRT 10.14.1 导入和 Triton kernel 探测均通过。依赖与 Vulkan layered 接入完成后的全量测试为 `610 passed, 6 warnings`。
- 将 Vulkan fused stereo pass 接入 `StereoRuntime` 的 `fast_plus` 生产路径：`Stereo Compute Backend=auto` 在真实 Triton 探测失败或厂商不支持时选择 Vulkan，显式 `vulkan` 可在 NVIDIA、AMD、Intel 上运行同一套 Vulkan 视差/warp/遮挡/补洞 shader；NVIDIA/AMD 仍保持 Triton 优先。
- 新增 `VulkanHostOutputAdapter`，Vulkan fallback 的左右眼结果可通过 Presenter 自有 Vulkan host image 输出，不再因没有 CUDA/HIP interop 而静默丢帧；该兼容路径使用同步 GPU copy/Quad Layer，不伪装成 Filament zero-copy semaphore 路径。
- 新增独立 `d2s_stereo_layered` Vulkan Compute pass，并接入 `quality_4k/hq_4k`；按现有分层合成逻辑执行深度分层权重、逐层水平 warp、遮挡膨胀、屏幕边缘抑制和 balanced/directional 补洞，未将高质量模式错误降级为 `fast_plus`。
- 为 Vulkan Compute 和 host fallback 增加分段耗时诊断：分别记录 host upload、Vulkan submit/wait、host readback、输出 wait 和输出 upload；当前路径仍明确标记为 host-visible fallback，未伪装成 zero-copy。
- 完成 Vulkan Compute 的真正输出零拷贝链路：新增 `d2s_stereo_layered_output.comp/.spv` 和 `VulkanStereoImagePass`，在 Presenter-owned Vulkan context 中直接把左右眼写入持久化外部 `VkImage`，不再读取 Vulkan 输出 buffer 到 CPU，也不再由 CPU 重新上传左右眼像素。
- 新增 `VulkanZeroCopyOutputAdapter` 和 `VulkanComputeRequest`：推理线程只发布 RGB/Depth/参数请求，Presenter 线程完成 Compute submit、`GENERAL -> SHADER_READ_ONLY_OPTIMAL` barrier、per-eye visible semaphore 以及 Filament finished/release 状态机；不支持直接外部采样时仍显式走 GPU copy 回退。
- 完成 NVIDIA CUDA 到 Vulkan Compute 输入的 GPU 互操作第一阶段：RGB/Depth 使用可导出的 Vulkan storage buffer、CUDA external memory mapped buffer 和 external ready semaphore，Presenter 不再执行每帧 Tensor `.cpu()` 与 host-visible storage buffer 上传；不支持 CUDA external buffer 时明确回退并在日志中报告 `vulkan_input_path=host_visible_buffer`。
- Filament 屏幕状态日志新增 `vulkan_input_path` 与 `vulkan_input_upload_ms`，用于区分真正的 CUDA external buffer 输入和兼容性 host 上传路径。
- 优化 `d2s_stereo_layered_output` 的 4K 热点：无空洞像素不再执行补洞采样，遮挡羽化避免嵌套 7×7 重复膨胀；RTX 3090 实测 CUDA external input 约 `0.1–0.2ms`，均匀/单边缘深度 steady-state 约 `10ms`，随机高边缘压力场景约 `61ms`，后续仍需用实机深度帧验证画质与稳定 FPS。
- 增加 Vulkan timeline 主机等待接口，保护输入 buffer 重用和 fallback source image release，不使用逐眼 `vkDeviceWaitIdle` 作为正常同步；同步契约和 CUDA external buffer 输入、host-visible 显式回退的范围已写入三份规范与需求矩阵。

### 验证结果

- `src/python3/python.exe -m pytest -q tests/test_openxr_vulkan.py tests/test_cuda_vulkan_interop.py tests/test_runtime_output.py`：105 passed，2 warnings。
- `src/python3/python.exe -m pytest -q tests/test_openxr_behavior_parity.py tests/test_openxr_vulkan.py`：107 passed，2 warnings。
- `src/python3/python.exe -m pytest -q`：618 passed，6 warnings。
- 当前 Windows Bridge 能力探针：外部屏幕图像、Vulkan external image、ready semaphore、finished semaphore、async submit 均为可用；CUDA runtime external semaphore API 也可见。
- 真实 Vulkan runtime 小帧验证：`StereoRuntime(stereo_compute_backend="vulkan")` 完成 Vulkan context 创建、fused dispatch、同步和左右眼读回；Vulkan host output adapter 在真实 Vulkan context 上创建并上传 8×4 左右眼图像成功。
- 真实 Vulkan layered runtime 小帧验证：`d2s_stereo_layered.spv` 在 NVIDIA RTX 3090 上完成 16×32、3 层 dispatch，左右眼和遮挡 mask 尺寸正确且均为有限值。
- 真实 Vulkan direct-image smoke：`d2s_stereo_layered_output.spv` 在外部 device-local `VkImage` 上完成左右眼 dispatch，状态转换到 `SHADER_READ_ONLY_OPTIMAL`，输出元数据为 `vulkan_readback=none`、`vulkan_output_sync=vulkan_compute_external_semaphore`；Presenter `VulkanZeroCopyOutputAdapter` ring smoke 通过。
- Vulkan 接入回归测试全部通过；覆盖自动后端切换、OpenXR prewarp 接入、Vulkan runtime integration 和 host output adapter。

## 2026-07-24

### 已实现

- 默认开启 Filament 外部源 `VkImage` zero-copy 实验路径；Presenter 仍执行完整能力门控，条件不满足时自动回退 Vulkan GPU copy/Quad Layer。设置 `D2S_ENABLE_FILAMENT_SCREEN_IMAGE=0` 可恢复旧路径进行回归对比。
- 接入 Filament v1.74.0 Vulkan backend 源码远程构建：新增 `native/filament/patches/apply_d2s_vulkan_external_image.py`，为 `VulkanPlatform` 增加正式的借用式外部 `VkImage` 元数据和工厂接口；GitHub Actions 在 Windows、Linux、macOS 远程构建 patched Filament 与 Bridge，并通过 `D2S_FILAMENT_VULKAN_EXTERNAL_IMAGE` 编译开关报告能力。本机不编译 C++，旧 Bridge/stock SDK 仍由能力探针自动回退 GPU copy。
- 修复 Filament 源码远程构建的跨平台差异：Linux runner 显式使用 Clang，macOS 外部图像元数据接口改为无 nullability 警告的引用参数，Windows Bridge 从源码安装前缀递归发现实际 `.lib`；上一轮 CI `30158908856` 因这些构建配置问题失败，未生成可用三平台 Bridge。
- 修复 Linux Filament 源码构建缺少 BlueVK XCB 头文件：CI 安装 `libxcb1-dev`；CI `30161119621` 的 Windows、macOS 已成功，Linux 仅因该依赖失败。
- 优化 Filament 远程构建耗时：GitHub Actions 按 runner、架构、Filament 版本和源码补丁 hash 缓存源码、CMake 构建目录及安装前缀；缓存命中时跳过 Filament 全量编译，只构建 Bridge。Linux 同时补齐 `libx11-dev`。
- 补齐 Linux BlueGL 构建依赖 `libgl1-mesa-dev`；CI `30163398945` 的 Windows、macOS 成功，Linux 最后缺少 `GL/gl.h`。
- NVIDIA CUDA producer-ready/consumer-release external semaphore 现在默认开启；设置 `D2S_ENABLE_CUDA_EXTERNAL_SEMAPHORE=0` 可单独关闭 CUDA 外部同步进行回归对比。CUDA runtime 或 Vulkan 能力不足时仍自动回退 GPU copy。
- 定位并阻止 Filament Vulkan 外部纹理 native 崩溃：Filament v1.74 公共 `Texture::Builder::import()` 仅支持 OpenGL/Metal 纹理标识，不接受裸 Vulkan `VkImage`。新增 `filament_bridge_vulkan_external_image_abi_available` 能力门控，当前 stock SDK 报告不支持时不再进入危险直采样调用；zero-copy 请求仍保留，运行时安全回退 Vulkan GPU copy，后续扩展 Filament Vulkan backend 后再打开真实路径。
- 修复 OpenXR FPS 面板和操作指南导致的帧率骤降：工具 Quad layer 现在缓存 PIL 栅格化纹理，并复用已上传且已释放的 Vulkan swapchain image；内容未变化时每帧只重建轻量 layer pose，不再重复字体绘制、host staging map/copy 和 acquire/wait/release。FPS 面板接入 Presenter 的真实 XR 提交帧率、运行时输出帧率和输出延迟，并按旧工程每秒采样一次；操作指南保持静态 GPU 纹理复用。虚拟屏幕的每帧立体输出不受影响。
- 修复 Projection Layer 内的显示顺序：虚拟屏幕从 Renderable priority `7` 调整为背景 priority `0`，保持不写深度；手柄 PBR 和深度测试激光在其后渲染，屏幕不再覆盖手柄和激光。该修改需要三平台 Filament Bridge 远程构建后实机验证。
- 修复 OpenXR `Default` 无房间环境中手柄和激光发白：Default profile 显式使用 `preview_exposure: 0.0`，不再继承运行时 `2.0 EV` 的线性曝光默认值。手柄 PBR 材质和激光仍共用既有 View 色彩管线，房间环境与 Default 的曝光行为现在一致。
- 明确 Vulkan 外部屏幕图像直采样的长期规范：每张源 `VkImage` 只创建一次 Filament 外部纹理，并保存格式、尺寸、layout、队列归属、producer-ready 和 consumer-release 状态；CUDA、ROCm/HIP 等 GPU producer 写入后都必须经过 barrier 和 queue ownership transfer 到 `VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL`，Filament 采样完成后才能复用槽位。屏幕源同步与 OpenXR 输出交换链同步分离；能力不完整时回退一次 Vulkan GPU copy/Quad Layer，不退回 CPU 像素传输。直接采样实验路径现已默认开启，但仍待实机长稳验证后转为稳定默认路径。
- 接通 Filament 外部源 `VkImage` 的 zero-copy 同步闭环：CUDA/ROCm producer-ready external semaphore 由 Presenter 在 source barrier submit 中等待，`GENERAL -> SHADER_READ_ONLY_OPTIMAL` 完成后发出每槽位的 device-local visible semaphore，Filament 通过现有 `set_screen_ready_semaphore` 等待后直接采样持久化外部纹理；帧完成后执行 `SHADER_READ_ONLY_OPTIMAL -> GENERAL` release barrier，并以每槽位 exportable consumer-release semaphore 通知 producer，producer 在复用 ring slot 前通过 CUDA/HIP stream wait 消费该 semaphore。同步能力不完整时仍自动回退 Vulkan GPU copy/Quad Layer，不退回 CPU 像素传输。该路径仍需 Validation Layer、三平台 Bridge CI 和实机长稳验证后再改变默认开关。
- 扩展稳定 Filament C ABI：native Vulkan platform 保存 Filament `present()` 提供的 per-eye `finished_drawing` semaphore，并通过 `filament_bridge_get_finished_drawing_semaphore` 返回借用句柄；Python release barrier 现在直接等待该 GPU 完成点后发出 producer consumer-release，不再依赖 `wait_for_idle()` 推断采样结束。该 ABI 修改需要三平台 Bridge 远程构建产物才能实机使用。
- 固化“命令队列 + Presenter 线程执行”不可回退约束：后台线程只提交原始结果或资源描述，所有 Vulkan/Filament 图形资源创建、外部图像导入、barrier、队列归属转换、纹理绑定、释放和 Projection/Quad Layer 提交均由 Presenter 在 OpenXR 帧边界执行。后续外部 `VkImage` 直采样重构不得恢复后台线程直接调用 Filament C ABI 或操作 Vulkan 资源。
- 按旧工程行为基准恢复控制器输入语义：Vive/WMR 触控板上/中/下方向模拟、模拟按键与真实 click 的互斥、OpenXR changed 边沿回退、touch/click 区分以及按键/摇杆动画目标平滑重新接入 Vulkan Presenter。该修复只调整 Python 输入策略，Filament/Vulkan 资源仍由 Presenter 命令队列线程独占。
- 修复工具 FPS Quad Layer 的隐性每帧重绘：延迟值改为与 XR/SBS FPS 一起按约 1 秒快照更新，内容 key 不再被每帧延迟变化击穿；操作指南继续复用静态纹理，XR 帧只更新 Quad layer 姿态和提交结构，避免重复字体栅格化与 staging 上传拖慢 Projection Layer。
- 收紧屏幕外部 `VkImage` 直采样门槛：显式开启实验路径时必须提供 per-eye producer-ready、source prepare、visible semaphore 和 consumer-release 回调；源图像初态为 `GENERAL`，由 Presenter 在 GPU submit 中完成采样前转换，不再要求生产者伪造 shader-read 初态，也不再仅凭 producer semaphore 导入未完成同步的图像。
- 输出契约现在发布左右眼源 `VkImage` 的真实 layout 和 queue family；CUDA 写入后的 `GENERAL` 状态会被明确暴露，未经过 source barrier 的图像不会被误判为可供 Filament 采样。
- Vulkan context 增加显式的 source barrier/release barrier API，并同步维护 `ImageStateTracker`；提交器同时支持 binary semaphore wait/signal，并与现有 timeline wait/signal 合并到同一次 `vkQueueSubmit2`/`vkQueueSubmit`。

### 验证结果

- 增加 Default profile 曝光回归测试；待实机确认手柄高光和激光饱和度。

## 2026-07-23

### 已实现

- 修复 OpenXR 环境选择仍硬编码旧 `Artemis` 目录的问题：运行时现在按 `settings.yaml` 的 `Environment Model` 解析环境目录，并读取所选 `profile.json` 的 `glb` 字段；所选目录、profile 或 GLB 无效时记录具体原因并回退 `Default`。`Default` 的 `glb: null` 被视为合法无房间环境，并使用旧版默认的 `2.4m x 1.35m`、距离 `2.0m` 虚拟屏幕，不再静默进入无房间且无屏幕的黑场。
- 修复 OpenXR 单 View 颜色跨帧叠加：Filament `Renderer::ClearOptions` 默认 `clear=false`，此前只在绿色透视背景快捷键切换时才初始化，普通外部 swapchain 渲染会保留上一帧颜色。Bridge 现在为每只眼在创建时固定设置黑色 clear color、`clear=true` 和 `discard=true`。
- 修复单 View 重构后 OpenXR 画面跨帧残影：删除原“主 View 与激光 View 同帧共享深度”遗留的 `setChannelDepthClearEnabled(0, false)`，单 View 现在每帧清理 channel 0 深度；此前设置会让外部 OpenXR swapchain 的深度跨帧保留，导致场景、屏幕和激光连续叠加。
- 重构 Filament 控制器激光路径：删除独立 LDR 激光 View、重复 controller asset 和深度遮挡副本；GLB、手柄 PBR、屏幕/UI 与激光现在在一个主 View 中共享同一 Scene 和深度缓冲。手柄外壳写入深度后，激光以深度测试自然被遮挡。
- 主场景 ColorGrading 改为 `ToneMapping::LINEAR`，不再应用 ACES；保留后处理以在最终 sRGB 输出目标进行唯一一次编码，避免将线性工作空间和最终 transfer function 混为一处。

- 修复 Vulkan 手柄模型近乎纯黑：对照 WebXR Input Profiles 官方 Viewer，Bridge 不再覆盖控制器 GLB 的原始 `roughnessFactor` 和 `specularColorFactor`；新增向后兼容环境光 C ABI，将旧工程环境 profile 的 `env_ambient_color` 作为 Filament SH irradiance 接入，同时保留跟随头部主光和顶部补光。
- 调整 Vulkan OpenXR 启动顺序：同步完成模型与推理后端加载，首帧推理、立体合成和 shape-dependent warmup 发布就绪信号后，才创建 OpenXR Vulkan/Filament presenter。
- 输出消费者在 presenter 初始化完成前不再取走 `runtime_q` 中的首帧，避免图形启动期间丢弃已经预热完成的第一个可提交结果。
- 明确 Vulkan/OpenXR 图形预热契约：启动期预创建 Device、队列、swapchain、Filament 材质/资源和持久化输出槽；首个 graphics pipeline 提交仍在合法 OpenXR frame loop 内完成。
- 按旧工程恢复独立屏幕光叠加：运行时在线性空间异步提取双眼虚拟屏幕平均色，保持旧版 `82%` 屏幕色与 `18%` 中性色混合，并由屏幕中心、法线和对角线衰减驱动只照亮控制器通道的 Filament 聚光源。
- 将屏幕光与 `controller_hdr_lighting` 完全解耦：3D 房间模式继续使用 profile ambient/head/top light，HDR 模式在真实预过滤 IBL 接入前明确使用 profile 回退，两个模式都始终保留屏幕光。
- 修复更换手柄模型后 B 键引导端点错位：统一从当前右手柄 GLB 的 `b_button_pressed_value` 动画枢轴解析局部锚点，覆盖 HP、Index、PICO、Quest、Vive 和 YVR；品牌切换后立即清除缓存、重新计算并记录锚点，再应用当前 profile 校正和 Grip 世界变换。
- 增加手柄品牌级环境光补偿：HP、Valve Index、Vive 和 YVR 暗色模型通过各自 `profile.json` 使用 `20.0x` 环境间接光，PICO/Quest 保持 `1.0x`；启动加载和运行时切换模型都会立即刷新倍率，屏幕光、直接补光及 GLB 原始材质不变。
- 补齐控制器 GLB 动画三元组的等价 native 实现：`_value/_min/_max` 节点不再依赖 Filament `getEntities()` 是否枚举非渲染节点，Bridge 回退表覆盖六品牌全部完整三元组，并继续使用平移/缩放插值与四元数 SLERP。
- 恢复此前只创建但未消费的摇杆、触控板和 Quest thumbrest 触摸状态；触摸通过现有 `button_mask` 第 6 位穿过冻结 C ABI，驱动 touched 节点及触摸轴动画，不新增或改签名。
- 修正手柄激光遮挡方案、透明外壳及连续帧几何残影回归：撤销在两个 View 提交之间临时替换同一 Renderable 材质的异步不安全实现；控制器 GLB 改用 Filament instanced asset，主实例永久保留原始 PBR 材质并只进入 HDR 主 View，独立遮挡实例共享纹理、材质资源和顶点缓冲，但永久绑定 `colorWrite=false`、`depthWrite=true` 的深度材质并只进入激光 View。两实例同步姿态、按键动画和显隐，不再逐帧修改材质绑定。
- 修复原生 GLB 加载导致的 Filament 线程归属崩溃：撤销将 Filament Engine/AssetLoader 放入独立 `FilamentNativeLoader` 的方案；Filament Engine、GLB 资源、控制器、屏幕和眼睛渲染统一由 Presenter 线程拥有，避免 `This thread has not been adopted` 和跨线程 Vulkan 生命周期错误。原生 GLB 阶段日志保留用于定位 `createAsset`、`loadResources` 和 `flushAndWait`。
- 启动 Presenter 线程命令队列重构：输出消费者不再跨线程直接修改 Presenter 的待显示帧，而是投递 `submit_output` 命令，由 Presenter 在帧边界消费；队列覆盖和 Presenter 关闭时都会释放未消费的输出槽位，Filament C ABI 调用继续保持 Presenter 线程归属。
- 为 `FilamentVulkanBridge` 增加 owner 线程绑定：创建、渲染、资源操作和销毁都会在 Python ABI 层校验 Presenter 线程，未来任何跨线程调用会立即得到明确错误，而不是触发 native `This thread has not been adopted`。
- 修复运行时关闭竞态：主线程不再以 2 秒超时抢先调用 Presenter `close()`，先等待 `run_until` 在 owner 线程完成 Filament/Vulkan 释放，再执行无副作用兜底关闭。
- 收紧 Presenter 命令队列边界：OpenXR 输出消费者现在只投递原始推理结果，CUDA 到 Vulkan 图像导入、external semaphore、屏幕光采样、输出槽位租约和 Filament 提交统一在 Presenter 线程执行；非 Vulkan sink 保留兼容转换路径。
- 修复首帧后画面冻结：Presenter 每个 XR tick 只转换命令队列中最新的一条原始结果，避免一次 tick 连续耗尽 Vulkan 输出 ring 后在槽位租约上等待自身完成下一帧释放。
- 修复桌面预览拖影：Preview Bridge 现在显式清理每帧颜色和 channel 0 深度；此前只有 OpenXR eye renderer 设置了 ClearOptions，桌面交换链会保留上一帧的模型和网格。
- 调整桌面预览移动速度：默认为 `1.0 m/s`，按住 `Shift` 加速到 `5.0 m/s`，按住 `Ctrl` 降速到 `0.5 m/s`；VIEW 向下移动改用 `Alt`，避免与 Shift 加速冲突。

### 验证结果

- 启动顺序、pipeline 就绪事件、输出首帧门控与 OpenXR 定向回归 `81 passed, 2 warnings`。
- 手柄环境光、独立屏幕光、异步屏幕颜色采样和输出首帧定向回归 `13 passed, 2 warnings`。
- 完整测试套件 `526 passed, 6 warnings`，requirements-matrix 合规检查 53 项通过；待三平台 Bridge CI 和 OpenXR 实机亮度验收。
- 六品牌 B 键动画枢轴、引导几何、品牌环境光倍率及切换后立即刷新定向回归 `20 passed, 2 warnings`；完整测试套件 `539 passed, 6 warnings`。
- 控制器动画三元组、touch 输入链和稳定 C ABI 定向回归累计 `100 passed, 2 warnings`；完整测试套件 `551 passed, 6 warnings`，requirements-matrix 合规检查 53 项通过。
- Presenter 原始输出命令队列及线程归属回归 `90 passed, 2 warnings`；Python `py_compile`、`git diff --check`通过。
- 上一版“手柄/激光同一彩色 View”以及后续“同一 Renderable 临时换深度材质”方案均经实机判定无效：前者不能正确遮挡且会改变外壳合成，后者与 Filament 异步命令消费竞争并造成场景/手柄连续帧残影。现已统一替换为资源共享、Renderable 独立、材质绑定持久不变的双实例遮挡结构；本地完整回归 `550 passed, 1 deselected, 6 warnings`，排除项仅为工作区 Artemis 目录改名导致的旧路径测试；三平台 Bridge CI `29991685595` 全部通过，自动二进制提交 `af11576` 已拉取，Windows DLL SHA256 为 `7D27C6E2298192C72A0B1718CD6BDC979757E45D3AB0865A9ADD38089815CF3A`，待头显遮挡验收。

### 未决事项

- HDR 图片环境尚缺与源 HDR 匹配的预过滤 reflection cubemap 与 irradiance KTX 接入；当前日志为 `hdr_ibl_pending_profile_fallback`，不会将 profile 回退误报为完整 IBL。
- 新增的环境光和屏幕光 C ABI 需要提交后由 GitHub Actions 完成 Windows、Linux、macOS 三平台 Bridge 远程构建，再下载产物进行实机亮度 A/B。

## 2026-07-22

### 已实现

- 重构原生 Filament Bridge：`filament_bridge.cpp` 仅保留稳定 C ABI 转发，Python ctypes 接口和 `filament_bridge.h` 不变。
- 将共享 Engine/Scene、双眼目标、GLB 场景、控制器动画、3D 激光、外部 VkImage 屏幕、材质色彩和桌面预览拆分为独立 `.cpp/.h` 模块，内部共享类型集中到 `bridge_internal.h`。
- CMake 显式编译各 Bridge 模块，并默认隐藏内部 C++ 符号，防止模块实现意外扩展 Python ABI。
- 三平台 Bridge 二进制改为分别存放在 `src/xr_viewer/native/windows`、`linux`、`macos`，运行时解析、能力探测、CMake 输出和 GitHub Actions 产物回写使用同一目录契约。
- 对齐旧工程控制器生命周期：补齐双手上一帧姿态、最后移动时间和移动阈值，修复 Aim 更新因状态未初始化而被静默清空。
- 增加逐手控制器显隐 ABI：Grip 跟踪有效且 5 秒内有移动时显示，静止超时或跟踪丢失时隐藏模型和激光，恢复移动后立即重新显示。
- 补齐此前缺失的 native 激光实现：在共享 Filament Projection Layer Scene 中创建双手独立 3D 光束实体，跟随 Aim 负 Z 位姿并与控制器同步显隐。
- 对齐旧工程激光标定参数：采用 Grip 上移 20mm、前移 110mm、Aim 绕局部 X 轴偏转 12 度、0.4m 长度和 6mm 根部宽度。
- 迁移旧工程激光稳定逻辑：位置使用 One Euro Filter，方向使用四元数 SLERP 和 0.3 度死区，逐帧更新后供 Vulkan Projection Layer 光束使用。
- 修复控制器 ABI 判定：手柄模型加载只依赖旧工程已验证的加载、姿态和输入三个接口，不再被可选激光接口缺失阻断。
- 修复实机手柄模型消失：当远程 DLL 尚未导出激光接口时保留控制器 GLB、姿态和按键动画；激光调用改为按 ABI 能力门控。
- 按旧工程恢复手柄激光外观：使用两张交叉锥形面、6mm 根部宽度、2mm 尖端宽度和沿光束流动的蓝/青/绿/黄/橙/红动态渐变，不再使用单张蓝色透明平面。
- 修复 native 按键动画插值：按钮、Trigger、Grip 和摇杆输入采用旧工程 24Hz 响应平滑，旋转由矩阵逐元素插值改为四元数 SLERP，同时保留平移与缩放插值。
- Bridge 加载手柄 GLB 时输出逐手动画节点数量、节点名和语义，避免 `_value/_min/_max` 匹配失败后静默运行。
- 修复彩色激光导致 OpenXR 初始化退出：将材质参数改为 `laser_time`，并按 Filament 约定使用普通参数访问器 `materialParams.laser_time`；sampler 才使用 `materialParams_<name>`。
- 规范捕捉到渲染的颜色空间路径：OpenXR Projection Layer 严格选择 sRGB swapchain，显示参照的虚拟屏幕和激光使用独立无后处理 LDR View，禁止对 SDR 颜色重复曝光、色调映射或传输函数转换。
- 修复手柄激光颜色偏淡及流动方向错误：激光改为不透明材质，颜色从手柄根部向远端流动，并绕过场景 ACES 色调映射。
- 按旧工程恢复手柄照明方式：手柄仅接收跟随眼睛的主光和顶部补光，环境灯与 HDR 反射不参与；灯光位置、颜色和顶部补光比例与旧工程保持一致，并将无单位强度转换为 Filament 坎德拉。
- 修复手柄初始纹理透明及隐藏恢复后变暗：移除会绕过 glTF 不透明材质合成链路的独立手柄 View，初始加载和 5 秒隐藏/恢复始终使用同一主场景层和专用灯光通道。
- 修复手柄按键动画完全无响应的根因：`_value`、`_min`、`_max` 节点统一从各自控制器 GLB 查询，不再错误地从环境 GLB 查询；左右手各 9 组动画节点按旧工程契约补齐并去重。
- 修正右手菜单动画节点名 `RMenu_pressed_value` 为 GLB 中真实存在的 `RMenu_value`；动画节点为空时控制器加载明确失败并报告错误，不再静默显示无动画模型。
- GitHub Actions 完成 Windows x86_64、Linux x86_64、macOS arm64 Filament Bridge 编译及二进制回写，本地同步至提交 `2aa8bbf`。
- 新增右手柄 B 键近距导引：手柄距头显 0.4 米内自动显示，仅保留 B 键透明说明框，并从 PICO GLB 的真实 B 键节点计算引导端点。
- 将 B 键导引由 OpenXR Quad Layer 迁移到 Filament Projection Layer 的无后处理 LDR 层；透明纹理、白色边框和文字不经过场景曝光或色调映射，面板逐帧朝向头部并跟随按键旋转。
- 修复 Projection Layer 导引显示成白色大方块：按 Filament `transparent` 材质契约在纹理采样后执行预乘 Alpha，透明区域不再把保留的白色 RGB 直接混入画面。
- 将旧工程控制器短按/长按判定抽取为渲染后端无关的快捷键状态机，Vulkan 现复用相同语义处理 A/B/X/Y、菜单键、摇杆点击及握持组合键。
- 以操作指南为完整快捷键契约，补齐 A+B 手柄品牌切换/模型校准、Grip+摇杆屏幕旋转缩放与深度调整、桌面方向键/滚轮、虚拟键盘移动旋转缩放，以及单手/双手 Grip 激光拖动；键盘专用组合键与深度快捷键互斥，不再发生漏导入或按键抢占。
- 补齐 Vulkan 快捷键后端：A 键切换 48 段圆柱弧/平面屏幕，Y 键复位或轮换旧工程屏幕预设，X 键切换键盘、环境亮度或绿色透视背景，摇杆组合键控制 2D/3D、深度复位及系统复制/剪切/粘贴/回车。

### 验证结果

- Python 编译检查通过。
- OpenXR/Filament Bridge 定向测试 `53 passed`，完整测试套件 `486 passed`。
- 最新 OpenXR/Filament 控制器定向回归 `54 passed, 2 warnings`，左右手 GLB 各 9 组 `_value/_min/_max` 节点逐名校验通过。
- 用户实机验收通过：手柄纹理初始显示不透明，静止隐藏后恢复亮度一致；Trigger、Grip、摇杆及可用实体按键动画均可正常响应。
- B 键 Projection Layer 导引定向回归 `62 passed, 2 warnings`，Python 编译检查与 Git diff 空白错误检查通过。

### 未来目标

- 建立独立 `d2s-vulkan-1.4` 实验分支，基于固定版本 Vulkan 1.4 `vk.xml`/Headers 远程生成项目自用 Python binding wheel；生产分支继续保留 Vulkan 1.2/1.3 能力回退。
- 在 Vulkan 1.4 binding 可用后，对 `hostImageCopy` 与独立 Transfer Queue 的工具纹理上传路径进行同场景基准测试；只有 Validation Layer、三平台 CI 和实机性能数据证明有净收益时才合入主路径。

## 2026-07-21

### 已实现

- 修复实机控制器按键动画未更新：按旧工程的中性 `value -> min/max` 方式插值 `_pressed_value` 节点，补齐 PICO `LPico/RPico` 语义。
- 将手柄激光从易丢失的 Quad Layer 改为 Filament Projection Layer 3D 几何体，使用 Aim 负 Z 射线和每帧世界变换提交。
- 修复实机控制器按键动画未更新：识别 PICO `LPico/RPico` 节点别名，并让 Bridge 的每帧动画刷新同时更新控制器 `_pressed_value` 节点。
- 提高 Vulkan 激光在头显中的可见性：沿旧工程 Aim 负 Z 射线保持 Quad Layer 提交，但扩大纹理采样核心和光束宽度，避免细光束在实际角分辨率下消失。
- 修复非 Windows 测试导入 `_KEYEVENTF_KEYUP` 失败：Windows 输入常量在 no-op 平台分支保持同名导出，GitHub Actions Linux 合规测试可正常收集 OpenXR 测试。
- 补齐 GitHub Actions OpenXR 测试依赖 `Pillow`，确保工具 Quad Layer 纹理模块在 Linux 合规环境可导入。
- 修复实机 `FilamentBridge.end_frame()` access violation：日志确认崩溃来自缺少源图像同步时的运行时 VkImage 直导入屏幕路径；无同步契约时使用旧工程已验证的 Projection Layer 场景加 Quad Layer GPU copy，zero-copy 仅在能力门控通过时启用。
- 将 Filament 屏幕直采改为能力门控：默认保留 zero-copy 意图，但只有输出帧同时提供左右眼 `gpu_external_semaphore` ready semaphore、Bridge 屏幕图像 ABI 和 semaphore ABI 时才启用；未满足同步契约时自动回退 Quad Layer GPU copy，避免未同步 raw `VkImage` 进入 Filament。
- 新增后端无关的 `GpuProducerAdapter` 契约；CUDA 适配器改为具体实现，输出同步模式统一为 `gpu_external_semaphore` / `gpu_synchronized`，为 ROCm/HIP 等后端接入保留同一 Vulkan producer 边界。
- Presenter 改为通过后端注册表创建 GPU producer，不再直接依赖 CUDA 类；已注册 CUDA 与 ROCm/HIP 适配器，后端自动识别，ROCm/HIP external semaphore 默认按 runtime 能力启用，API 不可用时明确回退 GPU copy，不会伪装成其它厂商或走 CPU 像素回传。
- ROCm/HIP runtime 缺失或 external-memory ABI 不可用时，适配器创建失败现在被 Presenter 捕获并限次记录，不再让 OpenXR Presenter 线程异常退出；后续帧仍可重试适配器创建。
- capability report 新增 GPU producer 自动选择结果、CUDA/HIP runtime 状态和覆盖标记，AMD 实机可通过 `--probe` 直接确认是否选择 ROCm，无需猜测环境变量。
- 修复 FPS/键盘等工具 Quad Layer 上传崩溃：`VulkanHostImage.upload()` 改用 PyVulkan 映射内存提供的可写 buffer，按 `rowPitch` 写入像素，不再把 cffi 映射对象错误转换为整数指针；菜单键打开 FPS 面板不再触发 `TypeError` 退出 XR 线程。
- 对齐旧工程控制器动画语义：补齐 PICO `photo/home/app` 按键到菜单动画映射，并将摇杆按下状态传入 native Bridge；控制器 `_pressed_value` 节点现在可响应按钮、摇杆和扳机输入。
- 修复 Vulkan 激光不可见：旧实现使用 Aim 射线负 Z 方向绘制长条光束；Vulkan Quad Layer 现在提交沿 Aim 射线排列的蓝色长条纹理，不再把光束放在控制器后方的微小圆点位置。

- 继续迁移旧工程完整 OpenXR 控制状态机：菜单、A/B/X/Y 和左右摇杆按键均支持短按/长按计时；短按/长按快捷键通过 Windows 输入注入，X 键切换虚拟键盘，菜单/A/B/Y 控制工具面板与屏幕复位。
- 迁移旧工程键盘输入保持状态：触发器进入、悬停、按住、切换按键和释放均由 `CoreInputHelpersMixin` 管理，支持 Shift/Ctrl/Alt/Win、Caps Lock、双击修饰键和方向键注入；Grip 按下时抑制误触键盘。
- 迁移旧工程鼠标长按/拖动状态：触发器先点击，超过 350ms 后进入拖动，释放时保证发送对应鼠标抬起；左右手分别映射右键/左键，键盘命中时不再穿透为桌面鼠标。
- 补齐 Vulkan 工具交互：左 Grip 锁定并移动键盘或虚拟屏幕，右 Grip 按横向位移调整屏幕宽度并保持纵横比；键盘 Quad Layer 使用当前位姿、Shift 状态和实时键盘尺寸生成。

- 修复 OpenXR 场景发白：保留旧工程验证过的 `R8G8B8A8_SRGB`/`B8G8R8A8_SRGB` Projection Layer 目标，将 Filament ColorGrading 输出改为线性 Rec709，由 sRGB 目标执行唯一一次 OETF；虚拟屏幕 Quad Layer 继续独立使用 UNORM 链。
- 修复实机 Quad Layer 屏幕变成长条：profile 未显式提供高度时按宽度自动计算 16:9 高度；修复 profile 校准后控制器仍使用旧 OpenXR reference space，手柄位姿现在与场景使用同一世界空间。
- 修复 Artemis 星空纹理转头闪烁：为 `Skybox__6464723579082975951` 的 8192x4096 纹理启用三线性 mipmap 采样，保留原始星空图像内容和空间位置。
- 修复 Quad Layer 虚拟屏幕上下颠倒：运行时输出和 Quad swapchain 统一采用 `top_left` 行序，拷贝路径不再强制额外 Y 翻转，Projection Layer 复制路径不受影响。
- 修复 Quad Layer 虚拟屏幕左右镜像：移除拷贝路径中的 X 翻转，避免把源图像方向问题误当成屏幕姿态问题。
- 完善统一输出契约：`VulkanStereoOutputFrame` 现在显式声明 `color_space=srgb` 和 `image_origin=top_left`；Quad Layer 不再对 `top_left` 源图像重复做方向转换。
- 修复 OpenXR Quad Layer 色彩路径：优先选择 sRGB Quad swapchain，与旧工程验证过的输出策略一致；OpenXR 配置现在使用用户选择的控制器型号。
- 修复 Filament 控制器模型全黑：控制器 GLB 加载后加入共享 fill-light channel，并保留各控制器 `profile.json` 的偏移/旋转校正。
- 对齐旧工程环境视角校准：profile reference space 应用时水平化初始头显姿态，再重新定位视图，避免实机视角偏离预览位置。
- 修复 Quad Layer sRGB 回归：UNORM runtime eye 到 sRGB Quad swapchain 现在使用 Vulkan blit 完成兼容格式转换，不再因格式不一致导致 OpenXR 线程退出。
- 统一 Quad Layer 图像方向：`image_origin=top_left` 在 Vulkan 拷贝路径不执行额外 X/Y 翻转，屏幕姿态与图像行序独立处理，避免历史硬编码镜像污染原始画面。

- OpenXR 运行时 Vulkan 中间图像保持 UNORM 存储；Filament 屏幕纹理按 sRGB 语义采样，Projection Layer 使用 UNORM 目标，避免已编码输出重复执行传输函数。
- 虚拟屏幕接入运行时左右眼 Vulkan 输出：导出图像增加 `SAMPLED` 用途，Filament Bridge 新增窄 C ABI，将借用的 Vulkan 图像导入屏幕材质；不引入 CPU 回读。
- 补充 Pico 4、Pico 4U 和 Pico Neo3 的 OpenXR interaction profile 绑定别名，控制器模型继续使用 Grip 位姿并回退到 Aim 位姿。
- 对照旧 `4k-stereo-synthesis-lab` 的已验证 Projection/Quad Layer 路径修正色彩契约：运行时输出帧显式标记 `color_space=srgb`，Filament 屏幕纹理使用 `SRGB8_A8` 采样；Projection Layer 使用 sRGB 目标且 Filament 输出线性 Rec709，Quad Layer 独立使用 UNORM；本地预览、MJPEG 和 RTMP 保持 display-referred sRGB，不重复 gamma。
- 桌面 Filament Preview native window swapchain 同样启用 `CONFIG_SRGB_COLORSPACE`，避免 Preview 与 OpenXR 使用不同的目标转换。
- 修复 Hugging Face 模型下载链：`snapshot_download()` 现在在实际选中的 `HF_ENDPOINT` 上执行；残缺的“只有权重”缓存不会再被误判为完整模型，降级 HTTP 下载会补齐 `config.json`，并保留在线 endpoint fallback。
- 按旧工程模型边界区分配置来源：DA3、InfiniDepth、VideoDepthAnything 使用 `src/stereo_runtime/model_impl` 内置结构配置，只要求远程权重；通用 Transformers 模型继续要求远程 `config.json`。
- 对齐旧工程 OpenXR 待机恢复逻辑：头显未连接或处于待机时不再退出 Vulkan 线程，而是使用可中断退避等待；`STOPPING/LOSS_PENDING` 后释放并重建 OpenXR/Vulkan 资源。
- 接入待机推理门控：头显等待前 60 秒保留 source 推理宽限期；持续不可用超过 60 秒后清空队列、停止捕捉和推理，头显恢复后重新打开推理并清理旧帧。
- 修复实机待机回调调用不存在的 `StereoRuntime.set_inference_active`：StereoRuntime 现在提供统一推理门控，并在暂停状态拒绝新的 RGB/OpenXR 推理帧。
- 修复 `WindowsCaptureCUDA` 与 TensorRT CUDA Graph 的 stream 冲突：检测到 CUDA 捕获时强制关闭已遗留的 depth CUDA Graph，并重建 provider 后使用普通 TensorRT enqueue。
- 明确记录 OpenXR 头显等待状态：首次检测不到头显或头显待机时输出一次等待提示；恢复时重置提示状态，避免等待逻辑静默。
- 重构 Filament Vulkan Bridge：左右眼现在共享一个 Filament Engine、Scene、GLB、控制器、屏幕材质和 Shader；每只眼睛仅保留独立 View、Camera、外部 OpenXR swapchain 和 acquired image。
- 头显未连接时明确记录 `xr.get_system` 尚未获得 HMD form factor，Vulkan/Filament 初始化会延迟到头显唤醒，不再让日志看起来像 Engine 创建失败。
- 修复头显从 60 秒 hard idle 恢复后的 Vulkan 外部图像生命周期竞态：等待态清空并拒绝旧输出帧，只有 `session_running` 且头显恢复渲染后才接收新帧；Filament 销毁导入屏幕纹理前等待 GPU 完成，避免 `Handle ... is being used after it has been freed` 导致原生进程中止。
- 修复 OpenXR 首帧退出：运行时输出尚未到达时，Filament 屏幕 Renderable 不再提前加入 Scene；收到有效 Vulkan 屏幕图像后才绑定 sampler 并显示，避免未设置 `screenTexture` 触发无效句柄访问。
- 修复双眼外部 Swapchain 的 Filament 帧状态隔离：共享一个 Engine、Scene 和资源，但左右眼各自使用独立 Renderer、View、Camera 和 Swapchain，避免单 Renderer 在两个 OpenXR Swapchain 间切换造成首帧 access violation。
- 为 OpenXR native Bridge 增加有界诊断：记录前八个立体帧的 eye、acquired image index、VkImage、Renderer 和 Swapchain 句柄，便于区分 OpenXR 图像句柄失效与 Filament 内部资源失效。
- 按旧工程 `OpenXRFrameGate` 补齐首帧门控：`should_render` 仅表示运行时允许渲染；在 `_pending_output` 尚未收到有效立体帧时只提交空 OpenXR 帧，不访问 Filament 或外部 swapchain，避免待机恢复阶段使用失效句柄。
- 修复首帧 Filament access violation 根因：不再把普通 Vulkan `VkImage` 直接传给 Filament 未定义 Vulkan 行为的 `Texture::Builder::import()`；虚拟屏幕后续按旧工程使用独立 OpenXR Quad Layer 接入。
- 接入 OpenXR Quad Layer 屏幕路径：首帧输出后按实际推理尺寸延迟创建左右眼 UNORM swapchain，使用 Vulkan GPU copy 写入并提交独立 Quad Layer；Quad 资源格式不再错误复用投影 sRGB 格式。
- 修复 Quad Layer 接入后的画面闪烁：首帧建立后，在没有新推理帧的 OpenXR tick 中复用上一帧 Projection/Quad Layer，不再提交空 layer；只有首帧前才进入等待状态。
- 对齐旧工程世界姿态处理：profile 座位姿态只在首个有效头部姿态时写入 OpenXR LOCAL reference space，并重新定位一次 views；后续 Filament 相机与 Projection Layer 使用同一套世界坐标 views，避免场景跟随头显初始姿态或转头抖动。
- 进一步对齐旧工程 reference space 选择：OpenXR Vulkan 路径优先使用 `STAGE` 地面世界坐标，运行时不提供时才回退 `LOCAL`；profile 校准复用实际选择的 reference space 类型，避免 LOCAL 原点绑定头显启动方向。
- 修复头显转动时场景回弹抖动：首帧后每个 OpenXR tick 都按当前头显 pose 重新渲染 Filament 世界，仅复用没有新推理帧的 Quad Layer 输入，避免用上一张旧姿态投影图替代当前相机姿态。
- 修复 GUI 子进程日志拼接误报：stdout/stderr 合并后若 profile 成功消息与 `[FPSBreakdown]` 粘连，先按日志标记拆分再分类，避免 `fx_entry_failed=` 等统计字段把成功消息标成 ERROR。
- 完成 CUDA/Vulkan/Filament external semaphore ABI 的三平台远程编译：GitHub Actions 运行 `29818061943` 的 Windows、Linux、macOS Bridge 构建及二进制回写全部成功；本地已同步 `filament_bridge.dll`、`libfilament_bridge.so` 和 `libfilament_bridge.dylib`。
- Vulkan 优化状态明确为分阶段完成：输出图像环、持久化纹理缓存、external semaphore 异步同步、双眼统一提交和单 Engine 资源共享已完成；完整 Compute Graph、Validation Layer、跨厂商互操作、性能基准和实机长稳验收仍未完成，不能标记为整体完成。
- 修复实机 `Windows fatal exception: access violation`：根因是 CUDA `cudaSignalExternalSemaphoresAsync` 的 ctypes 调用参数数量和 `cudaExternalSemaphoreSignalParams` 内存布局错误；现已按 CUDA Runtime 头文件使用 `extSemArray + paramsArray + count + stream` 的 ABI，并加入结构体偏移/尺寸回归测试。
- 针对 external semaphore 接入在 Filament `beginFrame` 阶段暴露的 native 生命周期风险，改为 `D2S_ENABLE_CUDA_EXTERNAL_SEMAPHORE=1` 显式启用，默认使用已验证的 CUDA stream 同步降级；Vulkan 输出图像环和持久化纹理缓存继续启用，避免实机默认路径再次发生 native access violation。
- 补齐 Vulkan 输出槽位消费端释放/复用保护：新增 producer lease 和跨线程条件等待；pending 帧被新帧替换时释放，当前 Filament 屏幕帧在 Projection/Quad 提交完成前保持占用，头显待机、提交失败和关闭路径统一释放，ring wrap 不再覆盖仍被消费的 VkImage。
- 修复实机首次 Projection 渲染 access violation：默认关闭运行时 CUDA `VkImage` 直接导入 Filament 屏幕材质的路径，屏幕改由已验证的 OpenXR Quad Layer Vulkan GPU copy 提交；保留 `D2S_ENABLE_FILAMENT_SCREEN_IMAGE=1` 作为后续 Validation Layer 验证用显式实验开关。
- 修复共享 Filament Engine 双眼切换 access violation：每只眼 `endFrame` 后先执行 `flushAndWait`，再切换到另一只外部 Vulkan Swapchain，避免上一只眼仍在后端处理时调用下一眼 `beginFrame`；该安全串行基线需要三平台 Bridge 重新远程编译。

### 验证结果

- 项目 Python 环境 `src/python3/python.exe` 完成语法检查。
- OpenXR、输出契约和运行时输出定向测试：`31 passed, 2 warnings`。
- 待机门控、CUDA 捕获隔离和 OpenXR 定向测试：`57 passed, 2 warnings`。
- 单 Engine 双眼 Bridge ABI 与 presenter 定向测试：`33 passed, 2 warnings`。
- 旧工程首帧门控契约测试：`1 passed`。
- OpenXR Quad Layer 定向测试：`37 passed, 2 warnings`。
- 用户实机验收通过：Vulkan Validation Layer 全路径验证通过；NVIDIA OpenXR 实机长稳、帧率和显存压力测试通过。
- `git diff --check` 通过。

### 未决事项

- external semaphore 仍默认关闭，待独立启用实验路径的跨 API 验证；完整 Vulkan Compute Graph、AMD ROCm/Apple 互操作和 Preview/OpenXR 色彩 AB 仍待完成。

### 下一项内容

- 完成 Vulkan Compute Graph 全路径接入，并继续验证 external semaphore 实验路径和跨厂商互操作。

## 2026-07-20

- Added a bounded runtime output consumer that converts only registered Vulkan eye resources into the unified output contract and reports Torch/CPU results as waiting for a vendor interop importer; no implicit CPU image readback is allowed.
- Extended the compliance workflow to run Vulkan resource, interop, output, runtime-output, pipeline, CUDA interop, and OpenXR lifecycle tests.
- Added exportable Vulkan image slots with Win32 HANDLE/FD export through the raw Vulkan loader entry point; resource ownership remains explicit and bounded.
- Added the Python-only NVIDIA CUDA Runtime importer: one-time external-memory import per slot and asynchronous CUDA-to-Vulkan RGBA copy, followed by stream synchronization before Vulkan copy.
- OpenXR Vulkan device creation now merges the Runtime-required device extensions with the platform external-memory extensions before xrCreateVulkanDeviceKHR.
- Runtime output now lazily creates two CUDA/Vulkan eye slots and submits the resulting Vulkan resources through the existing OpenXR projection path.
### 未决事项

- NVIDIA CUDA external-memory + 单次 GPU copy 已实现并通过 RTX 实机验证；ROCm/HIP、Apple Metal/IOSurface 和 CUDA/Vulkan external semaphore 仍待补齐。
- OpenXR 交换链到双眼推理图像的真实头显提交尚未实测；当前机器头显不可用，不能把清屏或单元测试视为头显验收。
- 完整预处理、深度后处理、视差、变形、修补和时域稳定 Compute Pass 尚未全部接入 Vulkan Graph。

### 下一项内容

- 使用已实现的 NVIDIA CUDA external-memory 单次 GPU copy 路径进行 OpenXR Projection Layer 头显实测；随后补 CUDA/Vulkan external semaphore 和 AMD ROCm/HIP 适配器。

### 已实现

- 新增 `shaders/manifest.json`，为每个 Compute Shader 固化入口、workgroup、descriptor binding、push constant 大小、精度和 SPIR-V 文件映射。
- 新增 `src/tools/validate_shader_manifest.py` 与 `tests/test_shader_manifest.py`，校验 Shader 源码声明、manifest 和已提交 SPIR-V 文件一致；GitHub Actions Shader Job 现在会执行该校验。
- 新增 `src/viewer/vulkan_interop.py`，建立 Capture/Inference 到 Vulkan 的非 CPU 回读资源边界：能力报告、外部图像导入请求、有限 in-flight 生命周期和 OpenXR/厂商适配器注册入口已统一；CUDA/ROCm/DMABUF 的平台句柄导入仍必须由各自适配器实现，当前不会伪造零拷贝状态。
- `VulkanImageCopyPass` 和 `VulkanRuntimeSession` 现在接受外部导入的 `VulkanImageResource`；新增 `submit_external_image_pair()`，厂商适配器可将资源直接送入 Compute Graph，并透传上游 timeline 完成值。
- OpenXR Projection Layer 组装集中到 `OpenXrCompositionBuilder`；swapchain image 在 acquire 成功后无论 wait 或渲染是否失败都会 release，避免 wait 异常留下悬挂 acquired image。
- 新增 `VulkanStereoOutputFrame` 和 `LatestFrameOutputRouter`，统一 Preview、OpenXR、Headless/Encoder 的左右眼、SBS、格式和 GPU ready timeline 输出契约，并限制每个输出路由只保留最新帧。
- 新增 `src/tools/vulkan_transfer_smoke.py`，验证两个 Vulkan storage image 在无 CPU 回读条件下通过 `vkCmdCopyImage` 和 layout barrier 完成 GPU copy，目标图像进入 `COLOR_ATTACHMENT_OPTIMAL`。
- Vulkan Context 关闭时现在先清理外部 image registry；即使 pending 状态导致正常注销失败，也会丢弃非拥有型句柄引用，不把已销毁 Device 的资源留在 Context 对象中。
- 迁入并接通 Python runtime context/callbacks，新增 `run_processing_runtime()`；GUI 调用的 `--runtime` 现在会启动 CaptureSessionLoop 和 RuntimePipelineLoop，不再返回“runtime is not assembled yet”。
- OpenXR 模式现在由 `run_processing_runtime()` 启动并管理 `OpenXrVulkanPresenter.run_until()` 线程；Presenter 的关闭顺序纳入运行时 shutdown，不再依赖独立 smoke 入口才能建立 Vulkan Session。
- RuntimePipelineLoop 现在对单帧推理异常执行丢帧并计数，连续达到 `D2S_RUNTIME_REBUILD_AFTER_ERRORS`（默认 3）后重建 Depth Provider、清除时域状态并记录重建失败。

- Compute Graph 的 `VulkanStereoSubmission` 新增可选 `ready_timeline`，上游 GPU 任务完成值现在会通过 `VulkanContext.submit_on("compute", wait_for_timeline=...)` 进入 Compute Queue；没有依赖值的旧调用保持兼容。
- Vulkan Context 新增 `last_submitted_timeline_value`，提交时校验队列角色并检查 FrameContext fence 超时，避免未知队列或无限等待被静默吞掉。
- ImageStateTracker 新增资源注销和 pending ownership transfer 保护；`VulkanStorageImage.close()` 释放 GPU 图像时同步移除状态，避免重用句柄后残留旧 layout/queue owner。
- 需求矩阵补充 VK-005 测试映射，并将 GRAPH-003 的上游 timeline 依赖记录为已实现的执行契约。
- `VulkanComputeGraph` 新增多 Pass 执行入口；`VulkanPassDeclaration` 固定 Pass 名称、workgroup 和资源读写集合，重复 Pass 名称或非法资源声明会在构图时失败。
- 多 Pass 之间仅在前一 Pass 写入、后一 Pass 读取或写入相同资源时插入 Compute Shader memory barrier，避免无条件全局 barrier。
- 新增 `shaders/d2s_copy_image.comp` 和对应 `vulkan_compute_smoke.py` 双 storage-image Descriptor 路径，作为 RGB/Depth 图像 Pass 的第一条真实输入输出链。
- 通过 `winget` 安装 Khronos Vulkan SDK `1.4.350.0`，使用官方 `Bin/glslc.exe` 生成 `shaders/d2s_copy_image.spv`，并重新编译项目 Compute Shader。
- 在 `.github/workflows/compliance.yml` 新增独立 Shader CI：安装 `glslc`/`spirv-tools`，编译全部 `.comp` 并执行 `spirv-val`，输出写入临时目录，不改写仓库中的二进制。

- Vulkan Context 新增 Graphics/Compute/Transfer 队列族选择和队列句柄暴露；优先选择专用 Compute/Transfer 队列，不具备时回退到 Graphics 队列。OpenXR adopt 路径明确复用 Runtime 已创建的 Graphics 队列。
- capability probe 现在报告 `graphics_queue_family`、`compute_queue_family` 和 `transfer_queue_family`，并新增队列族选择回退单元测试。
- Vulkan Context 新增默认容量为 3 的 `FrameContext` 环，命令池、命令缓冲和 fence 按槽位成组管理；提交不再每帧立即等待，而是在复用忙碌槽位时等待对应 fence。
- 用 `ImageStateTracker` 替换裸布局字典，记录 image 的 layout、access mask、pipeline stage 和 queue family；清屏路径现在从已登记状态构造转移 barrier，并在提交后登记目标状态。
- Vulkan 提交路径新增 Timeline Semaphore；当 Python Vulkan binding 暴露完整 Submit2 API 时使用 `vkQueueSubmit2`，否则使用带 `VkTimelineSemaphoreSubmitInfo` 的受控兼容提交。
- ImageStateTracker 新增 Queue Ownership 校验和 `VulkanContext.queue()/queue_family()` 角色查询；图形队列不会静默操作仍归 Compute/Transfer 队列所有的 image。
- ImageStateTracker 新增 pending ownership transfer 状态机，显式区分 transfer begin/release 与 complete/acquire；转移完成前资源不可被任一队列继续使用。
- 每个 FrameContext 现在为 Graphics/Compute/Transfer 分别持有有界 CommandPool、CommandBuffer 和 Fence；新增 `submit_on(role, record)`，`submit()` 保持 Graphics 兼容入口。
- `submit_on()` 新增 `wait_for_timeline` 参数；Submit2 使用 `VkSemaphoreSubmitInfo`，兼容提交使用 `VkTimelineSemaphoreSubmitInfo`，统一表达跨队列 wait/signal 顺序。
- 新增 `VulkanComputeGraph` 最小调度层：支持 `enqueue/flush/submit`、latest-frame 覆盖，并将 Compute Pass 录制回调提交到 `submit_on("compute")`。
- 新增 `shaders/d2s_noop.comp` 作为首个无资源 Compute Pass 源码，以及 `scripts/compile_shaders.ps1`；当前机器没有 `glslc`，未生成或伪造 `.spv` 二进制。
- 新增 Python `VulkanComputePipeline`：校验 SPIR-V、创建 ShaderModule/PipelineLayout/ComputePipeline，并提供 `vkCmdBindPipeline + vkCmdDispatch` 录制入口；没有 SPIR-V 文件时会明确报错。
- `VulkanComputeGraph` 新增 `from_pipeline()` 标准入口，并新增 `src/tools/vulkan_compute_smoke.py`，将 Graph、Pipeline、Dispatch 和 Timeline 验证串成可重复 smoke。
- 新增有界 `VulkanDescriptorArena`：按 `DescriptorBudget` 创建 DescriptorPool、限制 DescriptorSet 数量，并提供幂等释放。
- Compute Pipeline 支持 `DescriptorBinding` 列表，创建对应 DescriptorSetLayout 并挂入 PipelineLayout；默认无 binding 的 noop pipeline 保持兼容。
- 新增 `VulkanStorageBuffer` 和 DescriptorSet storage-buffer 更新路径；`d2s_storage_increment.comp` 实机验证 GPU 将 uint32 从 41 写为 42。
- 新增 `VulkanStorageImage` 和 storage-image DescriptorSet 更新路径；`d2s_storage_image.comp` 实机验证 image 创建、并发队列共享、UNDEFINED→GENERAL 布局转换和 `imageStore` Dispatch。
- Storage Image 的布局转换现在通过公开 Context API 登记到 `ImageStateTracker`，记录 `GENERAL + SHADER_WRITE + COMPUTE` 状态，后续 barrier 可复用统一状态。
- 安装 Vulkan SDK 1.4.350.0 到 `D:\VulkanSDK\1.4.350.0`，使用 `Bin\glslc.exe` 编译生成 `shaders/d2s_noop.spv`。

### 验证结果

- `src/main.py --runtime --runtime-seconds 2` 在 `D2S_RUNTIME_DIAG_STAGE=raw` 下成功启动并关闭 Capture/Runtime 线程；Vulkan transfer smoke 和 CUDA-to-Vulkan image copy smoke 通过；定向互操作测试 `25 passed, 2 warnings`；全量回归 `446 passed, 6 warnings`。
- `check_compliance.py` 通过 45 条需求，Shader manifest 校验通过，GitHub Actions workflow YAML 本地解析通过。
- NVIDIA Vulkan 实机 `vulkan_compute_smoke.py` 通过：`vulkan_compute_smoke: PASS timeline=1 state=ready`、`storage_image_dispatch: PASS`。
- `src/python3/python.exe -m py_compile` 覆盖本轮修改的 Vulkan Graph、Context、Descriptor 和测试文件通过。
- 全量测试 `417 passed, 4 warnings`；同时移除 `src/xr_viewer/gltf/materials.py` 的 UTF-8 BOM，使既有 legacy-depth 静态检查恢复可执行。警告均为 `mss.mss` 弃用提示。
- Vulkan 定向测试与迁移脚手架测试共 `30 passed`，覆盖上游 ready timeline 透传和图像状态注销。
- 迁移脚手架和 OpenXR Vulkan 定向测试共 `31 passed`，覆盖多 Pass barrier 计划和资源依赖声明。
- `vulkan_compute_smoke.py` 通过 `py_compile`，并在 NVIDIA Vulkan 环境中通过双 storage-image GPU smoke：`vulkan_compute_smoke: PASS timeline=1 state=ready`、`storage_image_dispatch: PASS`。
- 本地 4 个 Compute Shader 均通过 `glslc` 和 `spirv-val`；全量测试 `418 passed, 4 warnings`。
- GitHub Actions run `29743777308` 的 `Requirements matrix` 和 `Compile Vulkan shaders` 两个 Job 均通过，Shader 编译已纳入可复现 CI 验证。
- 新增 `VulkanImageCopyPass`，将双 storage-image dispatch 从 smoke 内联代码提升为可复用运行时 Pass；Pass 固定 `8x8` workgroup、有限 Descriptor 资源，并在提交前验证图像为 `GENERAL` 布局且归属 Compute Queue。
- `tests/test_migration_scaffold.py` 新增图像 Pass 的 workgroup、Descriptor 绑定和布局前置条件测试；定向测试 `14 passed`。
- 新增 `tests/test_vulkan_runtime.py` 验证运行时会话的尺寸校验、提交转发和资源关闭所有权；Vulkan 运行时定向测试共 `17 passed`。
- `VulkanImageCopyPass` 纳入 `stereo_runtime` 公共懒加载导出，后续运行时装配不需要依赖内部模块路径。
- 新增 `app_runtime.VulkanRuntimeSession`，统一持有 Vulkan Context 与图像 Pass；支持外部 Context 注入、内部 Context 生命周期和 ready timeline 透传，暂不接管 Capture/Inference。
- `vulkan_compute_smoke.py` 改为通过 `VulkanRuntimeSession.submit_image_pair()` 执行双 storage-image GPU Dispatch，完成从 app_runtime 到 Compute Graph 的实机链路验证。
- `VulkanRuntimeSession.close()` 现在先执行 `wait_idle()`，再销毁 Compute Pass 和自有 Context，避免 GPU 仍在使用 Pipeline 时发生资源释放竞态；测试锁定关闭顺序。
- `VulkanImageCopyPass` 提交前新增 Context 身份校验，拒绝来自其他 Vulkan Device/Instance 的 storage image；迁移脚手架新增跨 Context 回归测试。
- `compliance.yml` 新增 Vulkan runtime scaffold CI，自动执行迁移脚手架和 `VulkanRuntimeSession` 定向测试。
- 全量回归测试 `423 passed, 4 warnings`；全部 Compute Shader 重新编译并通过 `spirv-val` 校验。
- `VulkanRuntimeSession.resize()` 新增有界 Resize 流程：新尺寸 Pass 创建成功且 GPU idle 后才替换旧 Pass；Resize 失败时保留原运行资源。
- Resize 和生命周期定向测试共 `18 passed`。
- 新增 `VulkanDeviceLostError` 和 Session 健康状态；识别 Device Lost 后记录原始错误并拒绝后续提交，要求上层重建 Session。
- Device Lost、Resize 和运行时生命周期定向测试共 `19 passed`。
- 需求矩阵将 `VK-008` 更新为 `in_progress`，映射 `VulkanRuntimeSession` 和运行时生命周期测试；仍待专用硬件长稳与真实 Device Lost 注入验收。
- `GRAPH-003` 新增 1000 帧 latest-frame 压力测试，确认连续入队后只提交最后一帧，不累积旧帧延迟；Graph/Runtime 定向测试共 `20 passed`。
- GPU smoke 将 storage image 布局转换产生的最大 timeline 作为 `ready_timeline` 传入运行时图像 Pass，完成上游 GPU 完成点到 Compute submit 的实机验证。
- 需求矩阵将 `GRAPH-003` 更新为 `implemented`；latest-frame 覆盖、timeline 透传和实机 Compute 等待链路均已有代码与验证记录，仍需长期压力验收后才能升级为 `verified`。
- `VulkanImageCopyPass` 新增 source/output 图像别名保护，禁止同一 `VkImage` 同时作为只读输入和写入输出；Graph/Runtime 定向测试共 `21 passed`。
- 新增 Resize 失败回滚测试，确认新 Pipeline 创建失败时保留旧 Pass、旧尺寸和旧运行资源；Graph/Runtime 定向测试共 `22 passed`。
- 修复 OpenXR Vulkan Device 创建路径错误使用 `_require_timeline_semaphore_features()` 返回值的问题；现在正确解包 `pNext` Feature 链，并把 `synchronization2_enabled` 传入 adopted Context。
- OpenXR、Graph 和 Runtime 定向测试共 `41 passed`。
- OpenXR Feature 链修复后的全量回归测试 `428 passed, 4 warnings`，`VK-004` 继续保留专用设备创建集成验收状态。
- 新增 `pNext` Feature 链单元测试，验证 Synchronization2 链头、Timeline Semaphore `pNext` 节点和启用标志；OpenXR 定向测试 `20 passed`。
- 修复 `VulkanContext.adopt()` 未接收 `synchronization2_enabled` 参数的问题，避免 OpenXR 真实启动时因 Feature 状态透传触发 `TypeError`；adopt Context 现在记录该能力。
- 新增 `VulkanImageResource` 和 `VulkanExternalImageRegistry`，定义非拥有式外部图像句柄、尺寸、格式、状态和队列归属契约；Vulkan 只登记状态，不销毁 Capture/Inference 资源。
- 将 `ARCH-004` 和 `INFER-002` 更新为 `in_progress`；外部资源契约定向测试与 OpenXR/Graph/Runtime 测试共 `44 passed`，真实 CUDA/ROCm/DMABUF 导入仍待平台适配器。
- 外部资源契约接入后的全量回归测试 `431 passed, 4 warnings`。

- `py -m py_compile src/viewer/vulkan_context.py src/app_runtime/probe.py` 通过。
- 使用项目环境 `src/python3/python.exe -m pytest -q tests/test_openxr_vulkan.py`，18 项通过。
- Graph、SPIR-V loader、Descriptor budget、DescriptorSetLayout 与 Vulkan 定向组合测试：`28 passed`。
- `src/tools/vulkan_compute_smoke.py` 实机通过：Storage Buffer 从 41 更新为 42，`vulkan_compute_smoke: PASS timeline=1 state=ready`。
- Storage Image 实机通过：`storage_image_dispatch: PASS`。
- 实机 Compute 验收通过：Vulkan 1.4.329、`synchronization2_enabled=True`，真实创建 ComputePipeline 并执行 `vkCmdDispatch(1,1,1)`；Timeline value=1，Validation Layer 无 synchronization2 错误。
- `src/tools/probe.py` 实机探针通过：NVIDIA GeForce RTX 2060、Vulkan 1.4.329、Graphics=0、Compute=2、Transfer=1、Timeline Semaphore=true。
- 全量测试结果：404 项通过，1 项因既有 `src/xr_viewer/gltf/materials.py` 的 UTF-8 BOM 导致 AST 解析失败；该文件未由本次改动修改。
- `VK-002` 更新为 `implemented`；`VK-005` 更新为 `in_progress`，FrameContext 已建立，Descriptor、Pipeline 和完整 ImageStateTracker 仍未完成。
- `VK-006` 更新为 `in_progress`，当前已覆盖 layout/access 的清屏转移、状态记录和提交序列号；Queue Ownership 转移及 Validation Layer GPU 验证仍待完成。
- `GRAPH-001` 与 `GRAPH-003` 更新为 `in_progress`；当前仅完成 Graph 调度和 latest-frame 契约，真实 shader/pipeline 及完整处理链仍待接入。
- `GRAPH-002` 更新为 `in_progress`；shader 资源目录和编译入口已建立，待 Vulkan SDK 环境生成 SPIR-V 并完成 GPU dispatch 验收。
- 当前验证覆盖 Python 状态机和 Context 创建；Compute/Transfer 实际 shader 提交、跨队列 semaphore 等待和 Validation Layer GPU 验收仍待完成。
- 当前已验证提交结构和 Python API；真实 Compute Graph pass 尚未接入 `submit_on()`，因此仍需 GPU 实机验证跨队列同步。

### 未决事项

- 当前提交仍使用 `vkQueueSubmit` 和 fence；timeline semaphore、`vkQueueSubmit2`、Descriptor/Pipeline 生命周期和 ImageStateTracker 仍未完成。

### 下一项内容

- 为图像布局、访问掩码和 Queue Ownership 建立可追踪状态，并开始 timeline/submit2 调度迁移。

### 已实现

- 按 `docs/02-desktop2stereo-engineering-design-specification.md` 和 `docs/03-d2s_vulkan_migration_technical_report.md` 复核 Vulkan 主路径：保留 Python OpenXR/Vulkan 生命周期和唯一 Filament Vulkan Bridge 原生边界，继续禁止 D3D11、WGL/CUDA-GL 和 CPU 实时像素回读路径。
- 移除旧 Filament StarGlim 预览特效及其 sidecar、C ABI、Bridge 实现和三平台二进制残留；Artemis 星空改由 `environment.glb` 内嵌天空盒纹理负责。
- 修复 Filament 天空盒遮挡场景的问题：天空盒 renderable 使用背景优先级 `0`，避免遮挡土星环和其他 GLB 几何体。
- 恢复 Filament 桌面预览虚拟屏幕：新增屏幕四边形、窄 C ABI、Python ctypes 更新接口、尺寸/位置/旋转同步和半透明蓝色网格材质。
- 修复 Artemis 预览坐标空间错误：`view_pose` 继续从 profile 世界坐标转换到 GLB 场景坐标；`screen.position` 按当前 profile 约定直接作为 GLB 场景坐标，避免重复减去 `model_position`。
- 屏幕材质恢复旧版蓝色 `16x9` 网格效果，并关闭屏幕自身深度测试，避免被环境深度缓冲隐藏。
- Filament Bridge 通过 GitHub Actions 远程完成 Windows、Linux、macOS 三平台编译，最新二进制已下载回 `src/xr_viewer/native/`。

### 验证结果

- `py -m py_compile src/xr_viewer/preview_room_layout.py src/xr_viewer/filament_preview_bridge.py` 通过。
- `git diff --check` 通过。
- Filament Bridge CI runs `29723126977`、`29724387172`、`29725167239` 和 `29726120882` 的三平台构建通过。
- 当前预览桌面窗口仍按 `preview_room_layout.py` 的 `1280x720` 初始化；该尺寸只代表桌面预览，不代表 OpenXR 头显交换链分辨率。

### 未决事项

- `src/app_runtime/bootstrap.py --runtime` 仍未完成正式运行时装配，当前打印 `runtime is not assembled yet`。
- `src/stereo_runtime/vulkan_graph.py` 仍是提交契约骨架，尚未建立 Compute Pass、固定资源池、shader manifest 和 GPU 同步闭环。
- `VulkanContext` 已能选择并暴露 Graphics/Compute/Transfer 队列族，但仍以一次性 Command Buffer/Fence 提交为主，尚未达到规范要求的 FrameContext 池和完整 ImageStateTracker。
- Vulkan/OpenXR 清屏 smoke 已验证；Filament 场景 Bridge 的头显视觉验收、虚拟屏幕纹理采样和 Compute Graph 仍需继续打通。

### 下一项内容

下一项按 Phase 1/Phase 3 交界推进：先把 Vulkan Context 的 Graphics/Compute/Transfer 队列和有界 FrameContext/Synchronization 契约补齐，再接入最小可执行 Compute Graph，并同步更新需求矩阵和测试。

## 2026-07-19

### 已实现

- 将颜色调节选项暂时全部放入现有“高级立体参数”区域，不新增主界面分组；新增曝光、对比度、饱和度、Gamma、色温和色调六项控制。
- 将`src/main.py`默认入口接入Flet GUI，保留`--probe`能力探针入口；启动新项目不再停留在迁移脚手架提示。
- 补齐GUI颜色控件的运行时快照回写，热更新后曝光、对比度、饱和度、Gamma、色温和色调会同步显示当前生效值。
- 增加新Schema到旧GUI平面配置的启动兼容层：GUI启动时从`graphics/capture/inference/stereo/openxr/output`读取迁移配置，并补齐GUI和运行时所需默认字段，不直接覆盖原始嵌套配置。
- 修复兼容层`Model List`类型错误：按旧项目格式提供每个模型的`resolutions`对象，解决Flet启动时`'str' object has no attribute 'get'`。
- 颜色调节统一放在深度推理完成之后、立体合成和输出分发之前，因此本地预览、网络推流和 OpenXR 使用同一套颜色结果，且不改变 AI 深度输入。
- 新增颜色参数的配置保存、加载、GUI 热更新和运行时快照字段；调整颜色参数不触发模型、Filament 或 OpenXR 管线重建。
- 色温和色调采用相对值：范围均为`-100..100`，默认`0`；色温负值偏冷、正值偏暖，色调负值偏绿、正值偏洋红。

- 将`src/xr_viewer/preview_room_layout.py`的场景加载和逐帧渲染全面切换到Filament Desktop Preview Bridge。
- 新增`FilamentDesktopPreview` ctypes封装，通过Filament AssetLoader/ResourceLoader加载profile对应GLB，使用Filament Scene、Camera、View和Renderer提交桌面窗口帧。
- 桌面窗口使用GLFW原生句柄创建Filament SwapChain，支持Windows、Linux和macOS平台句柄，并同步窗口尺寸变化到Filament viewport。
- 删除预览入口中遗留的ModernGL shader、手写GLB解析、OpenGL资源上传和旧渲染辅助代码。
- Filament Bridge新增桌面预览生命周期、GLB加载、相机、投影、viewport和render C ABI；三平台产物自动回写`src/xr_viewer/native/`。
- 修复桌面预览profile座位偏高：profile中的座位保持世界坐标，加载Filament GLB前使用模型变换逆矩阵转换到场景坐标，保存时再转换回世界坐标；Artemis `y=901.0986`正确转换为GLB场景`y=58.0132`。
- 排查Artemis预览显存风险：旧项目原分辨率稳定约2.65 GB，新Filament原分辨率60秒稳定约2.64 GB，末端一次性上传完成后约2.93 GB，未发现逐帧增长或显存泄漏；Filament Bridge现在在GLB上传后执行`flushAndWait()`并释放GLB源数据，预览循环限制为60 FPS。
- 保留原分辨率为默认行为，新增可选的`--max-texture-size 4096`内存保护模式；该模式只重建内存中的预览GLB，不修改原始资源文件。
- 固化桌面预览Bridge调参ABI：新增曝光和方向补光的C ABI，Python通过profile或`--exposure`、`--fill-light-intensity`调整颜色，不再为亮度和灯光参数修改反复编译Filament Bridge。
- 修复Filament预览发黑：GLB纹理继续交给Filament按glTF sRGB/线性规则处理，View增加曝光色彩分级，并提供线性颜色方向补光；三平台Bridge构建通过。
- 将桌面预览默认曝光调整为`2.0 EV`；命令行显式`--exposure`和profile中的`preview_exposure`仍可覆盖默认值。
- 将天空盒与座位主体亮度解耦：Filament补光仅使用独立光照通道照亮非天空盒实体，天空盒材质通过独立的`skybox_brightness`乘数调节。
- 新增`--skybox-brightness`和profile字段`preview_skybox_brightness`；预览窗口使用`,`/`.`独立降低或提高天空盒亮度，`[`/`]`继续只调节座位主体曝光。
- 完成单View独立亮度方案：删除桌面预览全局ColorGrading，按GLB加载时保存的原始`baseColorFactor`分别缩放座位主体和天空盒材质。
- OpenXR Filament Bridge新增同一套`scene_exposure`与`skybox_brightness` ABI；每眼仍只提交一个View，不增加双View渲染开销。

### 验证结果

- 颜色相关 Python 文件 `py_compile` 通过，`git diff --check` 通过。
- `tests/test_settings_snapshot.py` 和 `tests/test_hot_reload.py` 共 31 项通过。
- `src/main.py --probe`通过；`gui.gui`模块成功导入，原先因`Stream Quality`缺失导致的启动异常已消除。
- Flet桌面客户端包已补齐；可直接运行`src\python3\python.exe src\main.py`启动GUI。
- 已使用`src/gui/flet_packages/flet-windows.zip`成功解压并启动GUI，`gui_ready.flag`已生成，Flet窗口初始化完成。

- Python `py_compile`和`git diff --check`通过。
- GitHub Actions run `29654473319`和`29654653736`的Windows、Linux、macOS构建全部通过。
- Windows DLL已确认导出`filament_preview_create`、`filament_preview_load_glb`、`filament_preview_set_viewport`和`filament_preview_render`。
- Artemis桌面预览进程可正常启动并持续运行，GLB资源加载无Python异常；日志仅有源图片的libpng iCCP警告。
- 代码提交：`7c38fbd`、`fee0eee`；原生二进制提交：`b06bad0`、`d905408`。

### 未决事项

- 需要用户确认Filament桌面窗口中的房间画面、profile座位高度和场景完整性。
- 尚未进行桌面预览与头显Projection Layer的最终视觉一致性对比。
- 单View材质亮度方案等待三平台Bridge构建及桌面/头显画面实测确认。

### 下一项内容

下一项：完成三平台Bridge构建，先验证桌面预览独立亮度，再进行头显双眼场景实测。

## 2026-07-18

### 已实现

- 建立独立项目`desktop2steoro-vulkan`，保持原项目的Python源码组织方式，不在运行时依赖原仓库。
- 迁移可复用的Capture、AI推理、Stereo、GUI、OpenXR平台无关模块、Samples、测试和工具；原项目文件保持不变。
- 迁移`native/filament`及Windows、Linux、macOS多平台GitHub Actions构建流程，统一产物目录为`src/xr_viewer/native/`。
- 确立Vulkan为主图形路径、OpenGL为隔离Fallback，不迁入旧Panda3D、D3D11 OpenXR、WGL/CUDA-GL Bridge和旧OpenGL上传链路。
- 实现Python Vulkan基础层，包括Instance、物理设备选择、Device、Graphics Queue、Command Pool、Command Buffer、Fence、图像布局转换、清屏提交和资源释放。
- 实现基于`XR_KHR_vulkan_enable2`的Python OpenXR Vulkan Phase 1，包括运行时选定物理设备、Session、双眼交换链、Projection Layer、事件处理和纯色帧提交。
- 新增`src/tools/openxr_vulkan_smoke.py`，用于头显环境下独立验证双眼Vulkan交换链。
- 更新`src/requirements.txt`，明确`pyopenxr==1.1.5301`和`vulkan==1.3.275.1`为Vulkan/OpenXR主路径依赖，PyOpenGL归入Fallback依赖。
- 修正pyopenxr Composition Layer提交方式，使用`ctypes.pointer(layer)`满足`FrameEndInfo.layers`的Base Header指针约定。
- 更新能力探针、README和迁移清单，使Phase 1状态与头显实测结果一致。

### 验证结果

- Filament Vulkan Bridge的Windows、Linux和macOS GitHub Actions构建已通过（run `29650016647`）。
- 新增手动发布工作流，可从成功的三平台 CI 运行中下载 DLL、so、dylib，打包为 GitHub Release 资产并生成 SHA-256 校验文件。
- 更新 Filament Bridge CI：三平台构建完成后自动将 DLL、so、dylib 下载到`src/xr_viewer/native/`并提交到`main`，不再只保留为临时 Actions artifact。
- 本机Vulkan探针识别到NVIDIA GeForce RTX 3090、Vulkan 1.4.341和Graphics Queue Family 0。
- Virtual Desktop OpenXR Runtime可加载，并声明支持`XR_KHR_vulkan_enable2`。
- Vulkan/OpenXR新增及迁移状态测试共13项通过。
- 最终全量测试394项全部通过；期间既有Hugging Face Provider测试曾因外部站点SSL EOF短暂失败，网络恢复后复测通过。
- `py_compile`和`git diff --check`通过。
- 未连接头显时，Smoke入口按设计返回`FormFactorUnavailableError`并完成资源清理。
- 连接头显后成功创建双眼3648x3648 Vulkan交换链，并完成300/300帧提交。
- 用户确认头显内稳定显示深蓝色双眼画面，无OpenXR调用顺序、Vulkan同步或资源释放错误。
- 开始实现Filament Vulkan Render Target Bridge：新增VulkanSharedContext接入、OpenXR VkImage外部SwapChain、Python ctypes封装和跨平台构建配置。
- Bridge明确借用Python/OpenXR所有Vulkan对象，不创建或销毁OpenXR资源；结束帧前使用Filament `flushAndWait`完成GPU同步。
- 将Bridge以显式配置方式接入`OpenXrVulkanPresenter`：左右眼分别绑定外部OpenXR VkImage，帧内传递acquire index，关闭顺序先Bridge后OpenXR交换链；未配置Bridge时保持原有Vulkan清屏路径。
- 在Filament Bridge内建立`Scene`、`Camera`和`View`，加载GLB后将实体加入场景，并在每帧调用`Renderer::render`；三平台CI重新编译通过，最新二进制已自动回写`src/xr_viewer/native/`。
- 增加每眼OpenXR Camera同步：Python根据View pose计算look-at参数，根据View FOV计算垂直视场角和aspect，并通过C ABI更新Filament Camera；三平台新Bridge构建和19项聚焦测试通过。
- 扩展`openxr_vulkan_smoke.py`支持显式指定`--filament-bridge`和`--filament-glb`，默认仍保持纯Vulkan清屏模式；README补充Bedroom环境GLB的Filament头显测试命令。
- 修复Filament 1.74外部SwapChain的`FixedCapacityVector`容量初始化，并保存平台层ExternalSwapChain句柄；Windows RTX 3090头显实测无GLB 60/60帧、QUEST控制器GLB 120/120帧、Artemis `environment3.glb` 120/120帧通过。
- 修复Presenter在Filament渲染完成后仍调用Python `clear_color_image`的问题；该清屏操作会覆盖Filament场景，导致帧提交成功但头显只显示深蓝色。Bridge启用时现在跳过Python清屏，Artemis场景再次完成120/120帧提交。
- 重新运行Artemis Filament头显测试：RTX 3090、双眼`3648x3648`交换链、300/300帧提交成功，进程正常退出。
- 新增Filament profile视角加载：读取`view_pose_index`选中的`view_poses`，将初始头部位姿映射到profile座位，同时保留运行时头部移动、双眼间距和Projection Layer位姿一致性。
- `openxr_vulkan_smoke.py`新增`--filament-profile`/`--profile`和`--seconds`参数，支持按profile视角进行长时间头显观察。
- Artemis `Model Center` profile长测通过：RTX 3090、双眼`3648x3648`交换链、120秒、8548帧提交成功，进程正常退出。
- 修复profile视角黑屏：`environment3.glb`已包含部分模型变换，而profile座位仍使用旧世界坐标；加载时按`model_position`将座位转换为GLB坐标。修正后20秒头显实测提交`1434`帧正常。
- 按原项目实际实现修正profile座位：含`x/y/z`的`view_pose`直接作为座位位置，`rotation_deg`或`angle`直接作为相机朝向；`screen`仅用于屏幕布局，不参与初始profile相机定位。
- 回退错误的屏幕相对座位变换后，Artemis原始`environment.glb`进行10秒头显实测，提交`717`帧正常。
- Filament Bridge新增非对称相机frustum ABI，Python按每眼OpenXR的left/right/up/down切角设置投影；profile的`xr_projection_near/far`也会传递到Filament，Artemis使用`0.1/20000.0`避免大场景裁剪。
- 实现桌面房间布局预览：`preview_room_layout.py`可加载profile对应GLB，显示环境模型和虚拟屏幕，并支持SCREEN/VIEW编辑、鼠标视角、座位移动、屏幕预设、裁剪范围和profile保存；补齐独立项目缺少的ModernGL glTF解析包及OpenGL状态辅助模块。
- 确认Artemis profile对应的原始`environment.glb`已可被当前Bridge加载；使用该匹配资源进行30秒头显实测，RTX 3090双眼`3648x3648`交换链提交`2117`帧正常。`environment3.glb`不再作为Artemis profile的默认测试资源。
- 将颜色曝光、对比度、饱和度、Gamma、色温和色调控件从高级立体参数移动到捕捉设备的高级设置中，并由“高级设备选项”统一控制显示。
- 将颜色控件显示名称调整为“亮度、对比度、饱和度、Gamma”；颜色行继续使用与上方参数一致的标签宽度、下拉框宽度和列间距。
- 按照运行模式选项框的尺寸，将颜色选项框统一调整为 `130`，并保持左侧选项列对齐。
- 修正颜色标签未参与全局标签列宽计算的问题，使亮度等颜色选项框与运行模式选项框使用同一左侧列基准。
- 将亮度从曝光补偿改为亮度倍率：`1.0` 为中性值，运行时直接乘以倍率；配置、热更新和运行时字段统一改为 `Color Brightness` / `color_brightness`。
- 将亮度倍率上限从 `4.0` 调整为 `2.0`，选项范围为 `0.2 - 2.0`。
- Artemis 房间预览接入 Filament 桌面预览动画 ABI，每帧播放 GLB 内嵌的 16 条卫星轨道动画和 3 条飞船轨道动画；按 `R` 重新加载 profile 时动画时间同步重置。
- Artemis 预览接入 `star_glim.json`：加载 stars/mask PNG，创建 Filament Vulkan 加法叠加材质，并按 sidecar 的密度、速度、软阈值和强度参数驱动星点闪烁。
- 重写 StarGlim 窄接口：仅保留动态材质创建、stars/mask 纹理、`intensity/speed/seed` 参数和时间更新四类 C ABI；删除旧的 `shine_speed/cell_*` 参数链。
- 预览每帧只计算一次 `animation_time`，同时传给 GLB 卫星动画和 StarGlim shader，确保两者使用同一时间轴。
- 将 StarGlim 动态材质创建放入 GLB 加载后的 Filament 场景初始化阶段；Python 语法检查和 JSON 校验通过。

### 未决事项

- CodeGraph数据库被当前MCP进程占用，本轮无法重建索引；代码和测试不受影响。
- 既有Hugging Face Provider测试依赖外部站点可达性，需要后续消除测试对网络状态的依赖。
- Filament Bridge的真实场景渲染尚未验证；当前Python封装只覆盖Bridge ABI和生命周期，不接管OpenXR acquire/release。
- Artemis和QUEST GLB已完成头显帧提交实测，等待用户确认头显内实际模型画面；FOV同步使用对称等效投影，OpenXR非对称左右/上下切偏移仍需使用自定义投影矩阵精确处理。
- Bedroom `environment.glb` 在Filament `load_glb`阶段解析失败，文件头和GLB声明长度一致，需后续用glTF Validator定位其扩展或资源兼容性问题。

### 下一项内容

下一项：提交源码并由 GitHub Actions 三平台重编译 Filament Bridge，下载新二进制后再测试 Artemis 星空与卫星动画同步效果。
## 2026-07-21

- Unified Preview and OpenXR Filament color processing in the shared native Bridge: both Views now explicitly use ACES legacy tone mapping, Rec709/sRGB/D65 output color space, and enabled post-processing.
- Kept scene exposure, skybox brightness, and directional fill-light values profile-driven and shared by Preview/OpenXR; the new CI Bridge binary is required before headset comparison.

- Added configurable OpenXR swapchain color mode: `sRGB`, `UNORM`, or `Auto`; the selected Vulkan format is logged for headset A/B validation.
- Added focused coverage for sRGB versus UNORM selection and invalid mode rejection. The default remains `sRGB`.

- Reused the legacy controller semantics in the Vulkan path: all complete brand folders under `src/xr_viewer/controllers/` are discovered, while the selected brand remains controlled by `D2S_CONTROLLER_MODEL`.
- Added narrow Filament Bridge controller ABI for left/right GLB loading, grip-root pose updates, trigger/grip/stick values, and button bitmasks.
- Implemented Filament-side `_value/_min/_max` node animation using the existing controller naming convention; no replacement renderer or new controller asset format was introduced.
- Connected the copied OpenXR action bindings and grip pose locator to the Vulkan presenter so controller input and model animation use the same frame loop.
- Python checks and focused OpenXR/Bridge tests pass: `26 passed`; the new Bridge ABI still requires the GitHub Actions three-platform rebuild before headset validation.

### 验证结果

- `src/python3/python.exe -m py_compile` passed for the new controller modules and OpenXR presenter.
- `src/python3/python.exe -m pytest -q tests/test_openxr_vulkan.py tests/test_filament_vulkan_bridge.py`: `26 passed`.
- `git diff --check` passed.

### 未决事项

- The controller ABI source is complete, but the checked-in native binaries do not contain these exports until the next GitHub Actions build.
- Headset acceptance still needs to confirm model placement and real PICO input/button animation.

### 下一项内容

- Commit and push the controller ABI/source changes, then download the three-platform CI Bridge artifacts and run the OpenXR headset test.

- OpenXR runtime now resolves the packaged platform Filament Bridge, Artemis GLB, and profile automatically; manual `D2S_FILAMENT_*` environment variables are no longer required for the Windows headset test.
- Set the default test configuration to `OpenXR Link` with scene exposure `2.0` and skybox brightness `1.0`.
- Fixed the OpenXR Vulkan device setup to stop calling the enable1-only `xrGetVulkanDeviceExtensionsKHR` while using `XR_KHR_vulkan_enable2`.
- OpenXR Artemis lighting now reads the same exposure, skybox, and directional fill-light profile values as the desktop preview; the updated Bridge binary must be rebuilt by CI.
- GitHub Actions run `29766759073` successfully rebuilt and committed Windows x86_64, Linux x86_64, and macOS arm64 Filament Bridge binaries; all three binaries were synchronized locally.
- The next validation is headset A/B comparison of Preview and OpenXR brightness, tone mapping, and sRGB/UNORM output.
- Fixed OpenXR startup failure caused by calling the obsolete `_initialize_controller_actions`; Presenter now calls the existing `_init_controller_actions` Mixin method.
- Fixed profile loading variable reuse: GLB camera position and virtual screen position now use separate variables, preventing `.tolist()` startup failure and preserving the profile camera pose.
- Fixed OpenXR Filament output setup: sRGB swapchains now pass `CONFIG_SRGB_COLORSPACE`; each frame now advances GLB animations on one shared timeline.
- Controller pose updates now fall back from grip pose to aim pose, and startup logs report controller brand, screen dimensions, and loaded Bridge state.
## 2026-07-21

- 修复 OpenXR Quad Layer 颜色路径：运行时 `uint8` 输出是已经编码的显示用 sRGB 字节，CUDA 导出图像和 Quad Layer 统一使用 `R8G8B8A8_SRGB`，同格式路径使用 `vkCmdCopyImage` 原样复制，避免 `UNORM -> SRGB` Blit 再次编码导致画面发白。
- Filament 虚拟屏幕继续以 `SRGB8_A8` 采样，并只在采样边界解码一次；CUDA 互操作只处理 RGBA 通道布局，不执行颜色转换。
- 修复 OpenXR Quad Layer 方向适配：保持输出契约 `image_origin=top_left`，仅在 Quad Layer 提交边界执行 Y 适配，不再进行 X 翻转。
- 修复环境 `profile.json` 相机高度：恢复旧工程的 `model_position/model_rotation_deg/model_scale` 逆变换，将世界坐标 `view_poses` 转为 GLB 局部坐标后再校准 OpenXR reference space。
- 修复控制器 profile 姿态：`model_rotation_deg` 按旧工程约定绕控制器模型局部 X 轴应用。
- 修复 Quad Layer 屏幕姿态：profile 的 `[yaw, pitch, roll]` 现在按旧工程的 Y/X/Z 旋转顺序转换为 OpenXR 四元数，不再把 yaw 错误当成 X 轴旋转。
- 规范化预览运行时和保存的姿态角：view 和 screen 的旋转始终保持在 `[-180°, 180°)`，避免连续旋转后出现 `902°` 等等价但难以阅读的角度。
- 修复 Projection Layer 虚拟屏幕无立体输入：每只眼睛的 Filament screen material 现在绑定对应的运行时 Vulkan eye image，避免屏幕纹理未接入或左右眼复用同一张图像。
- 对齐旧工程 Projection Layer 屏幕路径：屏幕仍作为场景几何体参与每眼投影渲染，纹理按 Vulkan image handle 缓存复用，不改为单张 2D 合成层。
- 按旧工程的异步提交边界优化 Projection Layer：左右眼 `end_frame` 只提交 Filament 工作，整帧两眼完成后统一等待一次，避免每眼一次 `flushAndWait` 串行阻塞；旧 Bridge 二进制仍保留兼容路径，需 CI 重编译后生效。
- 增加 CUDA/Vulkan/Filament external semaphore 路径：每个输出槽位创建可导出的 Vulkan binary semaphore，CUDA copy 完成后异步 signal，Filament Bridge 在目标 swapchain acquire 时等待对应 semaphore；平台或运行库不支持时自动退回 CUDA stream 同步。
## Unreleased

- Repacked the MSDF atlas into fixed 64x64 cells in the charset order, with
  deterministic left-to-right and top-to-bottom pages.
- Improved native MSDF coverage calculation for small VR OSD glyphs and forced
  linear atlas filtering with edge clamping.
### Unreleased

- Fixed controller surfaces being clipped by the room depth buffer after the
  room/controller lighting split. The foreground pass now clears only depth,
  preserving the room color while keeping opaque controller geometry intact.

### Filament v1.75.0 bridge compatibility

- Narrowed the downloaded BlueVK depth-clamp declaration guard for older Vulkan headers.
