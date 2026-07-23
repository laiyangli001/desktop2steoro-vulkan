# Desktop2Stereo Vulkan 项目日志

本文件记录项目重大更新和每日工作收尾。新记录按日期倒序追加；每个工作日结束时更新“已实现”“验证结果”“未决事项”和“下一项内容”。

## 2026-07-23

### 已实现

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

### 验证结果

- 启动顺序、pipeline 就绪事件、输出首帧门控与 OpenXR 定向回归 `81 passed, 2 warnings`。
- 手柄环境光、独立屏幕光、异步屏幕颜色采样和输出首帧定向回归 `13 passed, 2 warnings`。
- 完整测试套件 `526 passed, 6 warnings`，requirements-matrix 合规检查 53 项通过；待三平台 Bridge CI 和 OpenXR 实机亮度验收。
- 六品牌 B 键动画枢轴、引导几何、品牌环境光倍率及切换后立即刷新定向回归 `20 passed, 2 warnings`；完整测试套件 `539 passed, 6 warnings`。
- 控制器动画三元组、touch 输入链和稳定 C ABI 定向回归累计 `100 passed, 2 warnings`；完整测试套件 `551 passed, 6 warnings`，requirements-matrix 合规检查 53 项通过。
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
- 将 Filament 屏幕直采改为能力门控：默认保留 zero-copy 意图，但只有输出帧同时提供左右眼 `cuda_external_semaphore` ready semaphore、Bridge 屏幕图像 ABI 和 semaphore ABI 时才启用；未满足同步契约时自动回退 Quad Layer GPU copy，避免未同步 raw `VkImage` 进入 Filament。
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
