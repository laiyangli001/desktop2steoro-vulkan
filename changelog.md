# Desktop2Stereo Vulkan 项目日志

本文件记录项目重大更新和每日工作收尾。新记录按日期倒序追加；每个工作日结束时更新“已实现”“验证结果”“未决事项”和“下一项内容”。

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
