# OpenXR 旧工程行为迁移矩阵

本文是 `4k-stereo-synthesis-lab` 到 Vulkan 工程的行为迁移基线。它描述的是
“用户可观察行为”，不是把旧 OpenGL/D3D11 的渲染代码原样复制到 Vulkan。

## 迁移规则

1. 旧工程的 Python 状态机、输入语义、姿态计算和快捷键优先复用；只有
   Projection/Quad 资源提交和 Filament 绘制由 Vulkan 适配层实现。
2. 同一个行为必须能沿着“旧函数 → Vulkan 函数 → 状态/渲染层 → 验证”追踪，
   不允许只在 Vulkan 函数中重新发明一套等价逻辑。
3. 颜色、源图像分辨率和外部图像内容不做没有规格依据的映射。zero-copy 不满足
   同步契约时只能记录原因并使用 GPU copy 回退。
4. 本工程的产品约束是：虚拟屏幕、手柄模型和激光束属于 Projection；键盘、
   FPS、操作指南、光圈/激光命中圈属于 Quad。所有激光命中圈统一走 Quad，
   不因旧工程曾经把屏幕命中圈画在 Projection FBO 而恢复旧实现。

## 函数级迁移清单

状态含义：

- `✅`：代码和自动测试已对齐；
- `🧪`：代码路径已对齐，仍需头显验证；
- `⚠️`：已定位缺口，尚未完成迁移；
- `❌`：当前没有对应实现。

| 状态 | 旧工程函数/文件 | Vulkan 函数/文件 | 行为、状态和渲染层 | 自动对照 | 头显验证 |
|---|---|---|---|---|---|
| 🧪 | `CoreOpenXRInputMixin._poll_xr_events` | `OpenXrVulkanPresenter.poll_events` | Session 状态、实例丢失和 reference-space change；统一重建基础空间后重新应用 profile pose | `test_openxr_behavior_parity.py` | 挂起/唤醒、重定位后屏幕和手柄不跳变 |
| 🧪 | `implementation._render_eye`、`projection_layer_presenter` | `run_frame`、`_render_projection_layer` | 每帧按头部姿态提交双眼 Projection；屏幕和 3D 模型在 Projection | 现有 projection 测试 | 实机确认屏幕、手柄、激光遮挡关系 |
| ✅ | `CoreWindowInputMixin._cycle_a_panel` | `CoreControllerShortcutsMixin._update...`、`_set_shortcut_panel` | Menu 短按：隐藏 → FPS → FPS+屏幕竖向指南 → 隐藏 | `test_menu_panel_cycle_keeps_fps_when_vertical_screen_guide_is_shown` | 确认面板位于屏幕左侧且文字为中文 |
| ✅ | `CoreWindowInputMixin._cycle_b_panel` | `_set_hand_shortcut_panel`、`_render_tool_quad_layers` | B 长按：隐藏 → 手柄 FPS → 手柄 FPS+操作指南 → 隐藏 | `test_vulkan_b_long_press_cycles_hand_fps_and_operation_guide` | 确认 B 键长按时序和右手柄朝向头部 |
| 🧪 | `CoreOpenXRInputMixin._poll_controller_input`、`environment._cycle_light_from_x` | `_update_x_shortcuts`、`_cycle_filament_glow_mode`、`_dispatch_controller_shortcut` | X 释放小于 1 秒切换键盘、1～4 秒按 `Surround → Glow → Veil → Off` 循环、超过 4 秒切换旁路背景；不再错误循环房间灯光预设 | `test_x_long_press_action_cycles_v25_glow_modes_not_room_lighting` | Default、房间环境、旁路透视分别验证 |
| 🧪 | `implementation` 中 stick click 分支 | `_update_stick_shortcut`、runtime callback | Grip+左摇杆点击切换 2D/3D，Grip+右摇杆点击重置深度；无 Grip 时复制/剪切/粘贴/Enter | `test_vulkan_shortcut_delegates_runtime_owned_actions` | 每个组合键按下/释放和长按实测 |
| ✅ | `CoreControllerActionsMixin`、`CoreControllerPoseMixin` | 同名 Vulkan mixin、`_sync_controller_inputs` | 统一读取两手 A/B/X/Y/Menu、Grip、摇杆和 Aim/Grip pose | 现有 controller input 测试 | 不同控制器 profile 实测动作映射 |
| 🧪 | `implementation._poll_controller_input` 左 Grip 分支 | `_handle_vulkan_pointer_input`、`_handle_controller_guide_input` | 左 Grip 拖动屏幕；左 Grip+左/右摇杆旋转；屏幕命中点保持锚定 | 屏幕拖动/guide 对照测试 | 左右摇杆分别验证旋转、拖动不跳点 |
| 🧪 | `implementation._poll_controller_input` 右 Grip 分支 | `_handle_vulkan_pointer_input`、`_dispatch_controller_shortcut` | 右 Grip 拖动屏幕围绕头部球面旋转并保持距离；右摇杆改大小/距离；右腕旋转不改屏幕 | `test_right_grip_drag_orbits_screen_around_head_and_faces_head` | 旋转、拉近、拉远、松手后保持姿态 |
| ⚠️ | `implementation` 双 Grip 与 seat-adjust 分支 | `_handle_vulkan_pointer_input` | 双 Grip 旧逻辑包含头部球面拖动和长按进入座位调整；当前只做中心线性移动，未完整迁移 | 待补充几何测试 | 双 Grip 长按进入/保存座位调整 |
| ⚠️ | `CoreScreenControlMixin`、`CoreScreenStateMixin` | 当前 Presenter MRO 未接入这两个 mixin | 旧工程的 reset gaze、seat adjust、profile 保存和环境锁定函数目前只是仓库中的孤立代码，不能算已复用；需要先建立字段/空间模型映射再接入 | MRO/字段契约测试待补 | reset gaze、座位保存和环境锁定 |
| 🧪 | `implementation._laser_screen_hit_uv`、`_screen_uv_to_world` | `_screen_ray_hit`、`_screen_uv_to_world`、`_screen_hit_world_for_hand` | 已迁移平面 v 方向、曲面圆柱求交、UV→曲面世界坐标和命中圈曲面姿态 | `test_curved_screen_ray_hit_matches_legacy_cylinder_uv`、`test_flat_screen_ray_hit_keeps_legacy_bottom_to_top_v` | 曲面屏边缘命中、拖动和光圈位置 |
| ✅ | `core_keyboard` 的 anchor/pose/hit | `_keyboard_pose_mat4`、`_keyboard_plane_hit`、`_handle_keyboard_input` | 键盘相对屏幕定位、独立朝向头部、激光命中和按键输入 | `test_keyboard_pose_faces_head_independently...` | 键盘旋转头部时实时朝向头部 |
| 🧪 | `core_keyboard` 的 grip-to-move | `_handle_vulkan_pointer_input` | 键盘 Grip 拖动和屏幕/键盘 target latch | 现有 keyboard position 测试 | 键盘拖动、输入不误触桌面 |
| ✅ | `CoreLaserRenderMixin._laser_beam_setup` | `_update_filament_controllers`、`bridge_laser.cpp` | 激光束方向、隐藏计时、模型/激光共同生命周期；流动方向以实机视觉结果为准 | Filament bridge ordering/animation tests | 两手激光方向、5 秒隐藏、颜色深度 |
| ✅ | 旧 screen/keyboard cursor overlay | `_cursor_overlay_specs`、`_upload_tool_quad` | 屏幕和键盘命中圈统一生成透明 Quad；不进入 Projection 命中圈路径 | `test_laser_cursor_ring_is_emitted_at_screen_hit` | 命中圈大小、透明边缘和跟随旋转 |
| ✅ | `ControllerModelsMixin`、Filament controller scene | `_update_filament_controllers`、`bridge_controller.cpp` | 手柄模型、按钮动画、控制器优先级高于屏幕实体 | `test_filament_vulkan_bridge.py` | 手柄/激光显示在屏幕前，按钮动画可见 |
| 🧪 | `core_overlay_panels`、`overlay_quad_presenter` | `_render_tool_quad_layers` | FPS、屏幕操作指南、手柄操作指南、键盘、光圈均是 Quad；FPS 位于屏幕左下/手柄附近 | overlay texture/pose tests | 中文操作指南、面板位置和层顺序 |
| ⚠️ | `core_screen_quality._prepare_screen_quality_texture`、`screen_layer_presenter` | Filament external screen image path | 旧工程的 screen quality filter/RCAS 配置尚未在 Vulkan Filament 路径建立等价实现；禁止未经确认直接改颜色或锐化 | 待建立 source-resolution/sampler test | zero-copy 与 GPU-copy 清晰度对比 |
| 🧪 | 旧 `effects` Glow 路径 | `_projection_glow_state`、`VulkanGlowSourceComputeBackend`、`d2s_glow_source.comp`、Vulkan Projection Composer | 保留 Surround、Glow、Veil 和 Off 四态、矩形 SDF、现有 MIP 足迹及 premultiplied alpha；CUDA 源经 external buffer+semaphore 交给同 family 的独立 Vulkan Compute queue，生成 320x180 线性 RGBA8 外部图像；只绑定已完成槽位，未完成时复用上一帧，不阻塞主屏 zero-copy | Shader compile、`test_vulkan_glow_source.py`、GPU source/release-order tests | 三种可见效果、曲面屏、右 Grip+左摇杆透明度、更新率、主屏帧率和两小时长稳 |
| 🧪 | `implementation._poll_source_frame`、旧屏幕 presenter | `_render_projection_layer`、`_can_use_filament_screen_image` | 外部图像同步完整时 zero-copy；能力不足时 GPU copy，并记录唯一原因 | 现有 interop/screen path tests | 日志不刷屏、4K 清晰度和帧连续性 |
| 🧪 | `core_screen_control._reset_seating_vertical` | profile reference-space + `_apply_profile_reference_space` | 头部初始位置、设备 preset、屏幕默认距离统一从 profile/headset preset 计算 | profile tests | Default 场景头部位于屏幕中心 |

## 自动对照测试约定

测试不尝试在 CI 伪造 OpenXR runtime，而是分三层：

1. **状态机对照**：用同一组输入快照驱动旧工程可提取的纯函数语义和 Vulkan
   dispatcher，比较动作名、参数和状态序列。
2. **几何/层对照**：用固定 head/pose/屏幕参数比较屏幕距离、朝向、命中 UV、
   Quad/Projection 归属及实体 priority。
3. **实机验证**：需要 OpenXR runtime、Filament DLL 和控制器姿态的项目，测试
   记录为独立清单，不把“代码执行成功”当作视觉通过。

每迁移一组旧函数，必须同时更新：

- 本表中对应行的状态和验证条件；
- `tests/test_openxr_behavior_parity.py` 或现有对应测试；
- `changelog.md` 的用户可见行为变更；
- 若涉及 native ABI，追加远程 GitHub Actions 三平台产物验证。

## 当前必须实机验证项

- Projection 内虚拟屏幕、手柄模型、激光束的遮挡顺序；
- Default 场景头部中心和 headset preset 屏幕距离/尺寸；
- Menu/B 面板三态顺序、中文字体、面板位置；
- 键盘实时朝向头部、键盘命中圈和按键输入；
- 右 Grip 球面拖动、左 Grip 旋转、双 Grip/座位调整；
- zero-copy 外部图像清晰度与 GPU copy 回退清晰度；
- 光圈 Quad 的透明度、尺寸和激光命中点跟随；
- 手柄按钮动画、激光颜色和 5 秒隐藏生命周期。
