# Desktop2Stereo Vulkan 项目日志

本文件记录项目重大更新和每日工作收尾。新记录按日期倒序追加；每个工作日结束时更新“已实现”“验证结果”“未决事项”和“下一项内容”。

## 2026-07-20

### 已实现

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

- `src/python3/python.exe -m py_compile` 覆盖本轮修改的 Vulkan Graph、Context、Descriptor 和测试文件通过。
- 全量测试 `417 passed, 4 warnings`；同时移除 `src/xr_viewer/gltf/materials.py` 的 UTF-8 BOM，使既有 legacy-depth 静态检查恢复可执行。警告均为 `mss.mss` 弃用提示。
- Vulkan 定向测试与迁移脚手架测试共 `30 passed`，覆盖上游 ready timeline 透传和图像状态注销。
- 迁移脚手架和 OpenXR Vulkan 定向测试共 `31 passed`，覆盖多 Pass barrier 计划和资源依赖声明。
- `vulkan_compute_smoke.py` 通过 `py_compile`，并在 NVIDIA Vulkan 环境中通过双 storage-image GPU smoke：`vulkan_compute_smoke: PASS timeline=1 state=ready`、`storage_image_dispatch: PASS`。
- 本地 4 个 Compute Shader 均通过 `glslc` 和 `spirv-val`；全量测试 `418 passed, 4 warnings`。

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
