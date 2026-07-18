import gettext
from types import MappingProxyType


DEFAULT_LOCALE = "EN"

MESSAGE_CATALOGS = {
    "EN": {
        "Monitor": "Input Monitor",
        "Window": "Input Window",
        "Refresh": "Refresh",
        "Show FPS": "Show FPS",
        "Debug Mode": "Debug Mode",
        "Convergence:": "Convergence:",
        "Dynamic Convergence:": "Dynamic Convergence:",
        "Display Mode:": "Display Mode:",
        "Depth Model:": "Depth Model:",
        "Depth Strength:": "Depth Strength:",
        "Depth Quick:": "Depth Quick:",
        "Soft": "Soft",
        "Standard": "Standard",
        "Enhanced": "Enhanced",
        "Stereo Mode:": "Stereo Mode:",
        "Traditional / Fastest": "Traditional / Fastest",
        "Cinema": "Cinema / Balance",
        "Game / Low Latency": "Game / Low Latency",
        "Image  / High Quality": "Image  / High Quality",
        "Debug / Export": "Debug / Export",
        "Synthetic View:": "Synthetic View:",
        "Parallax Budget:": "Parallax Budget:",
        "Depth Separation:": "Depth Separation:",
        "separation_default": "Default",
        "separation_standard": "Standard",
        "separation_strong": "Strong",
        "separation_weak": "Weak",
        "comfort": "Comfort",
        "standard": "Standard",
        "strong": "Strong",
        "extreme": "Extreme",
        "fast": "Lowest",
        "fast_plus": "Medium",
        "quality_4k": "High",
        "hq_4k": "Highest",
        "Temporal Strength:": "Temporal Strength:",
        "Temporal": "Temporal",
        "Scene Threshold:": "Scene Threshold:",
        "Auto Scene Reset": "Auto Scene Reset",
        "Edge Dilation:": "Edge Dilation:",
        "Mask Feather:": "Mask Feather:",
        "Edge Threshold:": "Edge Threshold:",
        "Hole Fill Mode:": "Hole Fill:",
        "Balanced": "Balanced",
        "Balanced / Standard": "Balanced / Standard",
        "Soft / Low Ghost": "Soft / Low Ghost",
        "Sharp Test": "Sharp Test",
        "Sharp / High Detail": "Sharp / High Detail",
        "Content Aware / Highest Quality": "Content Aware / Highest Quality",
        "On": "On",
        "Anaglyph:": "Anaglyph:",
        "Cross Eyed": "Cross Eyed",
        "Advanced Stereo": "Advanced Stereo",
        "Advanced Device Options": "Advanced Options",
        "Depth Resolution:": "Depth Resolution:",
        "Anti-aliasing:": "Anti-aliasing:",
        "Depth Pop:": "Depth Pop:",
        "Foreground Pop:": "Foreground Pop:",
        "Midground Pop:": "Midground Pop:",
        "Background Pop:": "Background Pop:",
        "FP16": "FP16",
        "Inference Acceleration:": "Acceleration:",
        "Recompile TensorRT": "Recompile TensorRT",
        "Recompile CoreML": "Recompile CoreML",
        "Recompile OpenVINO": "Recompile OpenVINO",
        "Stop": "Stop",
        "Computing Device:": "Computing Device:",
        "Reset": "Reset",
        "Run": "Run",
        "Set Language:": "Set Language:",
        "Error": "Error",
        "Warning": "Warning",
        "Saved": "Run Desktop2Stereo",
        "PyYAML not installed, cannot save YAML file.": "PyYAML not installed, cannot save YAML file.",
        "Settings saved to settings.yaml": "Settings saved to settings.yaml",
        "Failed to save settings.yaml:": "Failed to save settings.yaml:",
        "Could not retrieve monitor list.\nFalling back to indexes 1 and 2.": "Could not retrieve monitor list.\nFalling back to indexes 1 and 2.",
        "Loaded settings.yaml at startup": "Loaded settings.yaml at startup",
        "Running": "Running... (Hold ESC 3s to Stop)",
        "Stopped": "Stopped.",
        "Countdown": "Settings saved to settings.yaml, starting...",
        "A thread already running!": "A thread already running!",
        "No windows found": "No windows found",
        "err_refresh_window": "Failed to refresh window list: {}",
        "Selected input window:": "Selected input window:",
        "Selected input monitor:": "Selected input monitor:",
        "Run Mode:": "Run Mode:",
        "Local Viewer": "Local Viewer",
        "Legacy Streamer": "Legacy Streamer",
        "MJPEG Streamer": "MJPEG Streamer",
        "RTMP Streamer": "RTMP Streamer",
        "Stream Protocol:": "Stream Protocol:",
        "Stream Key": "Stream Key:",
        "Stereo Mix": "Stereo Mix:",
        "CRF": "CRF:",
        "Audio Delay": "Audio Delay (s):",
        "Lossless Scaling Support": "LSFG",
        "3D Monitor": "3D Monitor",
        "OpenXR Link": "OpenXR Link",
        "Headset Model:": "Headset Model:",
        "XR Preview Window": "XR Preview Window",
        "VSync": "VSync",
        "Capture FPS:": "Capture FPS:",
        "Render Policy:": "Render Policy:",
        "Render Scale:": "4K Render Scale:",
        "Render Fixed Size:": "Fixed Size:",
        "Render Pixel Cap:": "Pixel Cap:",
        "Render Min Side:": "Min Side:",
        "Render Align:": "Align:",
        "Native": "Native",
        "Scaled": "Scaled",
        "Fixed": "Fixed",
        "Dynamic": "Dynamic",
        "Upscaler:": "Upscaler:",
        "Upscaler Sharpness:": "Sharpness:",
        "Auto": "Auto",
        "Off": "Off",
        "Streamer Port:": "Streamer Port:",
        "Streamer URL": "Streamer URL:",
       "Preview":"Preview",
        "Stream Quality:": "Stream Quality:",
        "Host": "Host:",
        "Invalid port number (1-65535)": "Invalid port number (must be between 1-65535)",
        "Invalid port number": "Port must be a number",
        "Please select a window before running in Window capture mode": "Please select a window before running in Window capture mode",
        "The selected window no longer exists. Please refresh and select a valid window.": "The selected window no longer exists. Please refresh and select a valid window.",
        "Failed to stop process on exit:": "Failed to stop process on exit:",
        "Failed to stop process:": "Failed to stop process:",
        "Failed to run process:": "Failed to run process:",
        "Failed to load settings.yaml:": "Failed to load settings.yaml:",
        "Opening URL in browser": "Opening URL in browser",
        "Controller:": "Controller:",
        "Environment:": "Room:",
        "Capture Tool:": "Capture Tool:",
        "Fill 16:9": "16:9",
        "Fix Viewer Aspect": "Fix Aspect",
        "Stereo Output:": "Stereo Output:",
        "Theme:": "Theme:",
        "torch.compile": "torch.compile",
        "TensorRT": "TensorRT",
        "CoreML": "CoreML",
        "OpenVINO": "OpenVINO",
        "tooltip_window": "Select a window to capture",
        "tooltip_depth_model": "Depth estimation model",
        "tooltip_model_size": "Model backbone size",
        "tooltip_depth_res": "Depth map resolution",
        "tooltip_convergence": "Zero-parallax screen plane. Raise it in 0.05 steps when foreground objects pop out too much or show ghosting; lower it when the whole scene feels too flat or sits behind the screen.",
        "tooltip_dynamic_convergence_strength": "Dynamic convergence strength. 0.00 keeps manual Convergence; values above 0.00 enable automatic convergence and follow the measured depth target.",
        "tooltip_depth_strength": "Overall stereo depth intensity. Range is 0.00-0.50 in 0.05 steps. Use Standard / 0.25 as the baseline; raise it only when the scene feels flat, and lower it when foreground objects show ghosts, edges tear, or viewing feels uncomfortable.",
        "tooltip_depth_quick": "Quick fixed depth presets for everyday use: Soft, Standard, or Enhanced",
        "tooltip_stereo_preset": "Stereo preset. Traditional is fastest, Cinema uses quality_4k, Game uses fast_plus, and Image uses hq_4k; selecting a preset loads its advanced parameters.",
        "tooltip_stereo_quality": "Internal stereo synthesis backend derived from Stereo Mode.",
        "tooltip_parallax_budget": "Maximum stereo parallax budget resolved from render size. Comfort is safest, Standard is the default, Strong and Extreme increase separation.",
        "tooltip_depth_separation": "Preset for Foreground, Midground, and Background Pop. Default uses 1.00/1.00/1.00; Standard uses 1.15/1.05/1.05; Strong uses 1.25/1.10/1.00; Weak uses 1.15/1.05/0.85.",
        "tooltip_max_shift": "Maximum horizontal shift as a ratio of image width; higher values increase stereo separation",
        "tooltip_temporal_strength": "Temporal smoothing strength for stereo output; higher values reduce flicker but can add lag",
        "tooltip_temporal": "Enable temporal stabilization between frames",
        "tooltip_scene_reset": "Scene-change threshold for resetting temporal history; lower values reset more often",
        "tooltip_auto_scene_reset": "Automatically reset temporal state when a scene cut is detected",
        "tooltip_edge_dilation": "Expands detected depth edges for occlusion handling",
        "tooltip_mask_feather": "Softens the occlusion fill mask; higher values reduce hard edge artifacts",
        "tooltip_edge_threshold": "Depth edge sensitivity; lower values detect more edges",
        "tooltip_hole_fill_mode": "Occlusion fill preset: Balanced / Standard keeps the realtime speed-detail balance, Soft / Low Ghost reduces edge ghosts, Sharp / High Detail keeps stronger detail for comparison, Content Aware / Highest Quality uses directional content-aware fill and is much slower.",
        "tooltip_anaglyph": "Color pair used when Display Mode is Anaglyph",
        "tooltip_cross_eyed": "Swap left and right eyes for cross-eyed viewing",
        "tooltip_advanced_stereo": "Show expert stereo/runtime parameters. Leave off for the simplified everyday UI.",
        "tooltip_advanced_device_options": "Show capture frame pacing, XR preview window, and image enhancement controls.",
        "tooltip_xr_preview": "Show the desktop XR preview window while running OpenXR Link.",
        "tooltip_depth_pop": "Centered depth curve: output = 0.5 + sign(depth - 0.5) * abs(depth - 0.5) ** (1 / (1 + Depth Pop)). Use 0 for no change.",
        "tooltip_foreground_pop": "Increase or reduce parallax shift for nearby objects, mainly people, hands, and tabletop foreground.",
        "tooltip_midground_pop": "Increase or reduce parallax shift for the main subject layer, mainly characters, vehicles, and common focus areas.",
        "tooltip_background_pop": "Increase or reduce parallax shift for distant background, mainly sky, walls, and far buildings.",
        "tooltip_antialiasing": "Depth-map smoothing level. Higher values reduce jagged depth edges and flicker, but can soften fine geometry; keep low for games/realtime, raise when object edges shimmer or produce broken stereo borders.",
        "tooltip_device": "Inference device",
        "tooltip_capture_tool": "Capture backend",
        "tooltip_run_mode": "Output mode",
        "tooltip_display_mode": "Stereo display format",
        "tooltip_xr_headset": "OpenXR screen preset. Each VR headset or AR glasses model applies its recommended viewing distance and 16:9 screen size when OpenXR Link starts.",
        "tooltip_vsync": "Synchronize the local viewer to the display refresh rate",
        "tooltip_target_fps": "Override internal frame pacing. Auto uses the detected display refresh rate.",
        "tooltip_render_policy": "Runtime render-size policy is fixed to 4K scale tiers.",
        "tooltip_render_scale": "Scale tier used only for 4K-class input. Output keeps the input aspect ratio; smaller input keeps its native size.",
        "tooltip_render_fixed_size": "Output size used when Render Policy is Fixed.",
        "tooltip_render_max_pixels": "Maximum output pixel count used by Dynamic render sizing.",
        "tooltip_render_min_dimension": "Minimum short-side dimension used by Dynamic render sizing.",
        "tooltip_render_align": "Output width and height alignment in pixels for runtime texture compatibility.",
        "tooltip_ctrl_model": "Controller model",
        "tooltip_env_model": "Room environment model",
        "tooltip_capture_mode": "Source: monitor or window",
        "tooltip_monitor": "Input monitor",
        "tooltip_stereo_monitor": "Stereo output monitor",
        "tooltip_lang": "Interface language",
        "tooltip_theme": "Color theme",
        "tooltip_stream_quality": "Encode quality",
        "tooltip_stream_proto": "Streaming protocol",
        "tooltip_audio": "Stereo mix device",
        "tooltip_stream_port": "Server port",
        "tooltip_stream_key": "Stream key",
        "tooltip_crf": "Quality factor (0-51)",
        "tooltip_audio_delay": "Audio offset (s)",
        "err_crf": "CRF must be between 0-51",
        "err_audio_delay": "Audio Delay must be between -10 and 10",
        "err_stream_key": "Stream Key can only contain letters, digits, underscore, hyphen, max 64 chars",
        "err_start_failed": "Start failed: {}",
        "esc_stop": "Hold ESC 3s — stopping!",
        "exited_with_code": "Exited with code {}",
        "failed_save_yaml": "Failed to save YAML: {}",
        "stereo_parameters_saved": "Stereo parameters saved",
        "invalid_url_scheme": "Invalid URL scheme: {}",
        "error_preview": "Failed to preview: {}",
        "url_copied": "URL copied to clipboard",
        "Log panel title": "Run Log",
        "Log panel running title": "Running - live log",
        "Log panel error title": "Issue detected - check logs",
        "Clear logs": "Clear",
        "Hide log panel link": "Hide log <-",
        "Show log panel link": "Show log ->",
        "Report issue": "Report bug",
        "Open log file": "Open log",
        "Opening log file": "Opening log file",
        "Bug report copied to clipboard!": "Bug report copied to clipboard!",
        "Starting Desktop2Stereo...": "Starting Desktop2Stereo {}...",
        "Runtime stopped": "Stopped",
        "Downloading model...": "Downloading AI model...",
        "Exporting ONNX...": "Exporting ONNX file...",
        "Building TensorRT engine...": "Building TensorRT engine (this may take a while)...",
        "Error occurred": "Error occurred",
        "Preparing Flet package...": "Preparing Flet desktop client...",
        "Startup preparation complete": "Startup preparation complete.",
        "Startup preparation failed: {}": "Startup preparation failed: {}",
    },
    "CN": {
        "Monitor": "输入屏幕",
        "Window": "输入窗口",
        "Refresh": "刷新",
        "Show FPS": "显示帧率",
        "Debug Mode": "调试模式",
        "Convergence:": "会聚位置:",
        "Dynamic Convergence:": "动态会聚:",
        "Display Mode:": "显示模式:",
        "Depth Model:": "深度模型:",
        "Depth Strength:": "深度强度:",
        "Depth Quick:": "深度选项:",
        "Soft": "柔和",
        "Standard": "标准",
        "Enhanced": "增强",
        "Stereo Mode:": "立体模式:",
        "Traditional / Fastest": "传统 / 速度快",
        "Cinema": "电影 / 偏均衡",
        "Game / Low Latency": "游戏 / 低延迟",
        "Image  / High Quality": "图片 / 高质量",
        "Debug / Export": "调试 / 导出",
        "Synthetic View:": "立体质量:",
        "Parallax Budget:": "视差预算:",
        "Depth Separation:": "前后分离：",
        "separation_default": "默认",
        "separation_standard": "标准",
        "separation_strong": "增强",
        "separation_weak": "减弱",
        "comfort": "舒适",
        "standard": "标准",
        "strong": "强",
        "extreme": "极强",
        "fast": "最低",
        "fast_plus": "中等",
        "quality_4k": "较高",
        "hq_4k": "最高",
        "Temporal Strength:": "时域强度:",
        "Temporal": "时域稳定",
        "Scene Threshold:": "场景阈值:",
        "Auto Scene Reset": "自动场景重置",
        "Edge Dilation:": "边缘扩张:",
        "Mask Feather:": "遮罩羽化:",
        "Edge Threshold:": "边缘阈值:",
        "Hole Fill Mode:": "补洞模式:",
        "Balanced": "均衡",
        "Balanced / Standard": "均衡 / 标准",
        "Soft / Low Ghost": "柔和 / 低重影",
        "Sharp Test": "锐利测试",
        "Sharp / High Detail": "锐利 / 高细节",
        "Content Aware / Highest Quality": "内容感知 / 最高质量",
        "On": "开启",
        "Anaglyph:": "红蓝模式:",
        "Cross Eyed": "交叉眼",
        "Advanced Stereo": "显示高级立体参数",
        "Advanced Device Options": "高级选项",
        "Depth Resolution:": "深度细节:",
        "Anti-aliasing:": "抗锯齿值:",
        "Depth Pop:": "深度弹出:",
        "Foreground Pop:": "前景视差:",
        "Midground Pop:": "中景视差:",
        "Background Pop:": "背景视差:",
        "FP16": "FP16",
        "Inference Acceleration:": "推理加速:",
        "Recompile TensorRT": "重译TensorRT",
        "Recompile CoreML": "重译CoreML",
        "Recompile OpenVINO": "重译OpenVINO",
        "Stop": "停止",
        "Computing Device:": "计算设备:",
        "Reset": "重置",
        "Run": "运行",
        "Set Language:": "设置语言:",
        "Error": "错误",
        "Warning": "警告",
        "Saved": "运行Desktop2Stereo",
        "PyYAML not installed, cannot save YAML file.": "未安装PyYAML，无法保存YAML文件。",
        "Settings saved to settings.yaml": "设置已保存到 settings.yaml",
        "Failed to save settings.yaml:": "保存 settings.yaml 失败：",
        "Could not retrieve monitor list.\nFalling back to indexes 1 and 2.": "无法获取显示器列表。\n回退到索引1和2。",
        "Loaded settings.yaml at startup": "启动时已加载 settings.yaml",
        "Running": "运行中...（长按ESC 3秒停止）",
        "Stopped": "已停止。",
        "Countdown": "设置已保存到 settings.yaml，启动中...",
        "A thread already running!": "一个进程已经运行！",
        "No windows found": "未找到窗口",
        "err_refresh_window": "刷新窗口列表失败：{}",
        "Selected input window:": "已选择输入窗口:",
        "Selected input monitor:": "已选择输入显示器 :",
        "Run Mode:": "运行模式:",
        "Local Viewer": "本地查看",
        "Legacy Streamer": "旧网络推流",
        "MJPEG Streamer": "MJPEG推流",
        "RTMP Streamer": "RTMP推流",
        "Stream Protocol:": "流协议:",
        "Stream Key": "推流密钥:",
        "Stereo Mix": "混音设备:",
        "CRF": "恒定质量:",
        "Audio Delay": "音频延迟 (秒):",
        "system": "系统",
        "blue": "蓝色",
        "green": "绿色",
        "red": "红色",
        "purple": "紫色",
        "orange": "橙色",
        "teal": "青色",
        "pink": "粉色",
        "grey": "灰色",
        "Lossless Scaling Support": "小黄鸭",
        "3D Monitor": "3D显示器",
        "OpenXR Link": "OpenXR串流",
        "Headset Model:": "头显型号:",
        "VSync": "垂直同步",
        "Capture FPS:": "捕获帧率:",
        "Render Policy:": "渲染策略:",
        "Render Scale:": "4K缩放档:",
        "Render Fixed Size:": "固定尺寸:",
        "Render Pixel Cap:": "像素上限:",
        "Render Min Side:": "最短边:",
        "Render Align:": "尺寸对齐:",
        "Native": "原生",
        "Scaled": "缩放",
        "Fixed": "固定",
        "Dynamic": "动态",
        "Upscaler:": "画面增强:",
        "Upscaler Sharpness:": "增强锐度:",
        "Auto": "自动",
        "Off": "关闭",
        "Streamer Port:": "推流端口:",
        "Streamer URL": "推流网址:",
        "Preview": "预览",
        "Stream Quality:": "推流质量:",
        "Host": "主机:",
        "Invalid port number (1-65535)": "端口号无效 (必须介于1-65535之间)",
        "Invalid port number": "端口必须是数字",
        "Please select a window before running in Window capture mode": "请在窗口捕获模式下选择一个窗口再运行",
        "The selected window no longer exists. Please refresh and select a valid window.": "所选窗口已不存在。请刷新并选择一个有效的窗口。",
        "Failed to stop process on exit:": "退出时停止进程失败：",
        "Failed to stop process:": "停止进程失败：",
        "Failed to run process:": "运行进程失败：",
        "Failed to load settings.yaml:": "加载 settings.yaml 失败：",
        "Opening URL in browser": "正在浏览器中打开网址",
        "Controller:": "手柄模型：",
        "Environment:": "房间模型：",
        "Capture Tool:": "捕获工具:",
        "Fill 16:9": "16:9",
        "Fix Viewer Aspect": "锁定比例",
        "Stereo Output:": "立体输出:",
        "Theme:": "主题颜色:",
        "torch.compile": "torch.compile",
        "TensorRT": "TensorRT",
        "CoreML": "CoreML",
        "OpenVINO": "OpenVINO",
        "tooltip_window": "选择要捕获的窗口",
        "tooltip_depth_model": "选择深度估计模型",
        "tooltip_model_size": "模型骨架大小",
        "tooltip_depth_res": "深度细节档位。建议使用最大 518，以获得更稳定的深度边缘、小物体结构和最好的立体细节；数值降低可减少推理耗时和显存占用，但更容易丢失轮廓层次。",
        "tooltip_convergence": "零视差屏幕平面。前景太突出或出现重影时，每次提高 0.05；画面整体太平、都贴在屏幕后方时，每次降低 0.05。",
        "tooltip_dynamic_convergence_strength": "动态会聚强度。0.00 表示使用手动会聚位置；大于 0.00 时启用动态会聚并跟随测得的深度目标。",
        "tooltip_depth_strength": "整体立体深度强度。范围 0.00-0.50，步进 0.05；建议以标准档 0.25 为基准，画面太平时再上调，前景重影、边缘撕裂或观看不舒服时下调。",
        "tooltip_depth_quick": "给普通用户使用的固定深度档位：柔和、标准、增强",
        "tooltip_stereo_preset": "立体预设模式。传统速度快，电影使用 quality_4k，游戏使用 fast_plus，图片使用 hq_4k；选择模式会加载对应高级参数。",
        "tooltip_stereo_quality": "内部立体合成后端，由立体模式自动决定。",
        "tooltip_parallax_budget": "根据渲染尺寸解析最大视差预算。舒适最稳，标准为默认，强和极强会增加立体分离。",
        "tooltip_depth_separation": "一键设置前景/中景/背景视差：默认为 1.00/1.00/1.00，标准为 1.15/1.05/1.05，增强为 1.25/1.10/1.00，减弱为 1.15/1.05/0.85。",
        "tooltip_max_shift": "水平位移占画面宽度的比例；越高立体分离越强",
        "tooltip_temporal_strength": "时域平滑强度；越高越稳定，但可能增加拖影或延迟",
        "tooltip_temporal": "启用帧间时域稳定",
        "tooltip_scene_reset": "场景变化重置阈值；越低越容易触发重置",
        "tooltip_auto_scene_reset": "检测到场景切换时自动重置时域历史",
        "tooltip_edge_dilation": "扩张深度边缘区域，用于遮挡和补洞处理",
        "tooltip_mask_feather": "柔化遮挡补洞遮罩；数值越高越能减轻硬边重影",
        "tooltip_edge_threshold": "深度边缘检测敏感度；越低检测到的边缘越多",
        "tooltip_hole_fill_mode": "遮挡补洞预设：均衡 / 标准保留实时速度和细节折中，柔和 / 低重影降低边缘重影，锐利 / 高细节保留更强细节用于对比，内容感知 / 最高质量使用方向内容感知补洞，速度会明显变慢。",
        "tooltip_anaglyph": "显示模式为红蓝/补色时使用的颜色组合",
        "tooltip_cross_eyed": "交换左右眼，用于交叉眼观看",
        "tooltip_advanced_stereo": "显示专家级立体和运行时参数；普通使用建议保持关闭。",
        "tooltip_advanced_device_options": "显示捕获帧率、XR画面预览窗口、本地垂直同步和画面增强选项。",
        "tooltip_xr_preview": "运行 OpenXR Link 时显示桌面 XR 画面预览窗口。",
        "tooltip_depth_pop": "居中深度曲线：output = 0.5 + sign(depth - 0.5) * abs(depth - 0.5) ** (1 / (1 + Depth Pop))。0 表示不改变深度曲线。",
        "tooltip_foreground_pop": "增强/减弱近处物体的位移，主要影响人物、手、桌面前景。",
        "tooltip_midground_pop": "增强/减弱画面主体层的位移，主要影响角色、车辆、常见焦点区域。",
        "tooltip_background_pop": "增强/减弱远处背景的位移，主要影响天空、墙面、远景建筑。",
        "tooltip_antialiasing": "深度图平滑级别。数值越高越能减少深度边缘锯齿和闪烁，但会软化细节；游戏和实时观看保持较低，物体边缘闪烁或立体边界破碎时再上调。",
        "tooltip_device": "计算设备",
        "tooltip_capture_tool": "捕获后端",
        "tooltip_run_mode": "输出模式",
        "tooltip_display_mode": "立体显示格式",
        "tooltip_xr_headset": "OpenXR 屏幕预设。选择 VR 头显或 AR 眼镜型号后，启动 OpenXR Link 时自动应用推荐观看距离和 16:9 屏幕尺寸。",
        "tooltip_vsync": "将本地查看窗口同步到显示器刷新率，关闭可用于帧率对比测试",
        "tooltip_target_fps": "覆盖内部帧率节流；自动使用检测到的显示器刷新率",
        "tooltip_render_policy": "运行时渲染尺寸策略固定为 4K 缩放档位。",
        "tooltip_render_scale": "仅 4K 级输入使用的缩放档位；输出保持输入宽高比，低于 4K 的输入保持原生尺寸。",
        "tooltip_render_fixed_size": "渲染策略为固定时使用的输出尺寸。",
        "tooltip_render_max_pixels": "动态渲染尺寸使用的最大输出像素数。",
        "tooltip_render_min_dimension": "动态渲染尺寸使用的最短边下限。",
        "tooltip_render_align": "运行时纹理兼容所需的输出宽高像素对齐。",
        "tooltip_ctrl_model": "手柄型号",
        "tooltip_env_model": "房间环境模型",
        "tooltip_capture_mode": "捕获源：屏幕或窗口",
        "tooltip_monitor": "输入显示器",
        "tooltip_stereo_monitor": "立体输出显示器",
        "tooltip_lang": "界面语言",
        "tooltip_theme": "主题颜色",
        "tooltip_stream_quality": "编码质量",
        "tooltip_stream_proto": "推流协议",
        "tooltip_audio": "混音设备",
        "tooltip_stream_port": "推流端口",
        "tooltip_stream_key": "推流密钥",
        "tooltip_crf": "质量因子 (0-51)",
        "tooltip_audio_delay": "音频偏移（秒）",
        "err_crf": "CRF 必须是 0-51 之间的整数",
        "err_audio_delay": "Audio Delay 必须是 -10 到 10 之间的数值",
        "err_stream_key": "Stream Key 只能包含字母、数字、下划线和连字符，最长 64 字符",
        "err_start_failed": "启动失败: {}",
        "esc_stop": "长按ESC 3秒停止",
        "exited_with_code": "退出码 {}",
        "failed_save_yaml": "保存 YAML 失败: {}",
        "stereo_parameters_saved": "立体参数已保存",
        "invalid_url_scheme": "无效 URL 协议: {}",
        "error_preview": "打开浏览器失败: {}",
        "url_copied": "已复制网址到剪贴板",
        "Log panel title": "运行日志",
        "Log panel running title": "运行中 - 实时日志",
        "Log panel error title": "检测到异常 - 请查看日志",
        "Clear logs": "清空",
        "Hide log panel link": "隐藏log<-",
        "Show log panel link": "显示log->",
        "Report issue": "反馈bug",
        "Open log file": "打开日志",
        "Opening log file": "正在打开日志",
        "Bug report copied to clipboard!": "异常反馈信息已复制到剪贴板！",
        "Starting Desktop2Stereo...": "正在启动 Desktop2Stereo {}...",
        "Runtime stopped": "已停止",
        "Downloading model...": "正在下载 AI 模型...",
        "Exporting ONNX...": "正在导出 ONNX 文件...",
        "Building TensorRT engine...": "正在编译 TensorRT 引擎（可能需要较长时间）...",
        "Error occurred": "出现异常",
        "Preparing Flet package...": "正在准备 Flet 桌面客户端...",
        "Startup preparation complete": "启动准备已完成。",
        "Startup preparation failed: {}": "启动准备失败: {}",
    }
}

LOCALE_ALIASES = MappingProxyType({
    "EN": "EN",
    "EN_US": "EN",
    "EN-US": "EN",
    "CN": "CN",
    "ZH": "CN",
    "ZH_CN": "CN",
    "ZH-CN": "CN",
    "ZH_HANS": "CN",
    "ZH-HANS": "CN",
})

SUPPORTED_LOCALES = tuple(MESSAGE_CATALOGS.keys())
UI_MESSAGES = MESSAGE_CATALOGS
UI_TEXTS = UI_MESSAGES


class CatalogTranslation(gettext.NullTranslations):
    def __init__(self, catalog, fallback=None):
        super().__init__()
        self._catalog = catalog
        self._fallback = fallback

    def gettext(self, message):
        if message in self._catalog:
            return self._catalog[message]
        if self._fallback is not None:
            return self._fallback.gettext(message)
        return message

    def ngettext(self, msgid1, msgid2, n):
        message = msgid1 if n == 1 else msgid2
        return self.gettext(message)


_EN_TRANSLATION = CatalogTranslation(MESSAGE_CATALOGS["EN"])
_LOCALE_TRANSLATIONS = {
    lang: CatalogTranslation(catalog, fallback=_EN_TRANSLATION if lang != "EN" else None)
    for lang, catalog in MESSAGE_CATALOGS.items()
}


def normalize_locale(locale):
    key = str(locale or DEFAULT_LOCALE).replace(" ", "_").upper()
    return LOCALE_ALIASES.get(key, key if key in MESSAGE_CATALOGS else DEFAULT_LOCALE)


def is_supported_locale(locale):
    return normalize_locale(locale) in MESSAGE_CATALOGS


def get_translation(locale=DEFAULT_LOCALE):
    return _LOCALE_TRANSLATIONS[normalize_locale(locale)]


def get_messages(locale=DEFAULT_LOCALE):
    return MESSAGE_CATALOGS[normalize_locale(locale)]


def gettext_for(locale, message):
    return get_translation(locale).gettext(message)

STEREO_QUALITY_KEYS = ("fast", "fast_plus", "quality_4k", "hq_4k")
PARALLAX_BUDGET_KEYS = ("comfort", "standard", "strong", "extreme")
DEPTH_SEPARATION_KEYS = ("default", "standard", "strong", "weak")
DEPTH_SEPARATION_LABELS = {
    "default": "separation_default",
    "standard": "separation_standard",
    "strong": "separation_strong",
    "weak": "separation_weak",
}


def stereo_quality_options(locale=DEFAULT_LOCALE):
    messages = get_messages(locale)
    return [messages[key] for key in STEREO_QUALITY_KEYS]


def stereo_quality_to_display(value, locale=DEFAULT_LOCALE):
    key = str(value or "quality_4k")
    messages = get_messages(locale)
    return messages.get(key, messages["quality_4k"])

def display_to_stereo_quality(value):
    text = str(value or "")
    for key in STEREO_QUALITY_KEYS:
        if text == key:
            return key
        for locale in SUPPORTED_LOCALES:
            if text == get_messages(locale).get(key):
                return key
    return "quality_4k"


def parallax_budget_options(locale=DEFAULT_LOCALE):
    messages = get_messages(locale)
    return [messages[key] for key in PARALLAX_BUDGET_KEYS]


def parallax_budget_to_display(value, locale=DEFAULT_LOCALE):
    key = str(value or "standard")
    messages = get_messages(locale)
    return messages.get(key, messages.get("standard", "Standard"))


def display_to_parallax_budget(value):
    text = str(value or "")
    for key in PARALLAX_BUDGET_KEYS:
        if text == key:
            return key
        for locale in SUPPORTED_LOCALES:
            if text == get_messages(locale).get(key):
                return key
    return "standard"


def depth_separation_options(locale=DEFAULT_LOCALE):
    messages = get_messages(locale)
    return [messages[DEPTH_SEPARATION_LABELS[key]] for key in DEPTH_SEPARATION_KEYS]


def depth_separation_to_display(value, locale=DEFAULT_LOCALE):
    key = str(value or "standard")
    messages = get_messages(locale)
    label = DEPTH_SEPARATION_LABELS.get(key, DEPTH_SEPARATION_LABELS["standard"])
    return messages.get(label, label)


def display_to_depth_separation(value):
    text = str(value or "")
    for key, label in DEPTH_SEPARATION_LABELS.items():
        if text == key or text == label:
            return key
        for locale in SUPPORTED_LOCALES:
            if text == get_messages(locale).get(label):
                return key
    return "standard"


HOLE_FILL_MODE_KEYS = ("balanced", "soft_low_ghost", "sharp_test", "quality")
HOLE_FILL_MODE_LABELS = {
    "balanced": "Balanced / Standard",
    "soft_low_ghost": "Soft / Low Ghost",
    "sharp_test": "Sharp / High Detail",
    "quality": "Content Aware / Highest Quality",
}
HOLE_FILL_MODE_LEGACY_LABELS = {
    "balanced": ("Balanced",),
    "soft_low_ghost": (),
    "sharp_test": ("Sharp Test",),
    "quality": ("Quality", "Content Aware", "Directional"),
}


def hole_fill_mode_options(locale=DEFAULT_LOCALE):
    messages = get_messages(locale)
    return [messages[HOLE_FILL_MODE_LABELS[key]] for key in HOLE_FILL_MODE_KEYS]


def hole_fill_mode_to_display(value, locale=DEFAULT_LOCALE):
    key = str(value or "balanced")
    messages = get_messages(locale)
    label = HOLE_FILL_MODE_LABELS.get(key, HOLE_FILL_MODE_LABELS["balanced"])
    return messages.get(label, label)


def display_to_hole_fill_mode(value):
    text = str(value or "")
    for key, label in HOLE_FILL_MODE_LABELS.items():
        labels = (label, *HOLE_FILL_MODE_LEGACY_LABELS.get(key, ()))
        if text == key or text in labels:
            return key
        for locale in SUPPORTED_LOCALES:
            messages = get_messages(locale)
            if any(text == messages.get(candidate) for candidate in labels):
                return key
    return "balanced"
