# 高画质缓冲输出技术可行性报告

## 结论

在当前 Desktop2Stereo 架构中，新增一个可选的“高画质缓冲输出”模式是可行的。

该模式不应覆盖现有实时模式，而应作为 GUI 中的独立勾选项启用。启用后，系统允许输出端落后捕获端一段可配置时间，用帧缓存和前后帧 lookahead 给深度估计、视差生成、遮挡补洞、时间稳定提供更充足的计算窗口。它适合看视频、电影、本地播放器、非交互式桌面内容；不适合游戏、鼠标精确操作、低延迟 VR 交互。

推荐定位：

```text
模式名称：高画质缓冲输出 / Quality Buffered Output
默认状态：关闭
目标场景：视频、影视、低交互本地内容
输出范围：OpenXR Link、本地预览/本地输出、3D 显示器输出
核心代价：增加可感知延迟
核心收益：更稳定的补洞、更少边缘闪烁、更高画面完整性
```

## 当前项目现状

当前 runtime 和 OpenXR 输出链路以“实时优先”为核心。

项目里的队列工具采用 latest-frame 语义：

```text
src/utils/queue_utils.py

put_latest(q, item)
-> queue 满时丢弃旧 item，再放入最新 item

drain_latest(q, first_item)
-> 清空队列中的 stale item，只返回最新帧
```

OpenXR viewer 侧也采用类似策略：

```text
src/xr_viewer/core_source_state.py

_poll_source_frame()
-> 非阻塞读取 depth_q
-> 如果一次读到多帧，只保留最后一帧
-> viewer_drop 统计被丢弃的旧帧
```

这说明现有架构的正确目标是“尽量显示最新画面”，而不是“保证每一帧都完整处理并按顺序播放”。因此，高画质缓冲输出不能直接改造现有实时队列语义，否则会破坏实时模式、OpenXR 交互体验和当前性能统计。

正确做法是在 runtime/presentation 边界新增一个可选分支：

```text
实时模式：
capture -> runtime latest-frame -> OpenXR / local output

高画质缓冲模式：
capture -> frame buffer window -> high quality synthesis -> delayed presentation -> OpenXR / local output
```

## 为什么缓冲能提升画质

单纯“延迟显示”只能吸收算力抖动，不能自动提升画质。真正有价值的是“延迟 + 帧窗口 + lookahead”。

实时补洞只能使用当前帧和历史帧：

```text
N-2, N-1, N
```

高画质缓冲可以在输出第 N 帧时使用未来帧：

```text
N-k ... N-1, N, N+1 ... N+m
```

这对 2D 转 3D 特别有价值，因为遮挡区域在当前帧不可见，但可能在前后帧中因为物体运动、镜头运动而露出。缓冲模式可以把这些信息用于：

- 视差拉伸后空洞区域修复。
- 人物、字幕、UI 边缘的稳定补洞。
- 镜头横移场景中的背景补全。
- 避免当前帧单独推断造成的边缘抖动。
- 降低 temporal flicker。

## 建议的功能边界

### GUI 控制

建议新增为明确的用户选择项：

```text
[ ] 高画质缓冲输出

缓冲延迟：
100 ms / 150 ms / 250 ms / 500 ms / 自定义

补洞质量：
标准 / 高 / 最高

音频同步：
自动 / 手动偏移
```

默认值建议：

```text
高画质缓冲输出：关闭
缓冲延迟：150 ms
补洞质量：高
音频同步：自动
音频偏移：0 ms
```

### 适用输出

该能力应做成通用输出能力，而不是 OpenXR 专属能力。

应支持：

- OpenXR Link。
- 本地模式输出。
- 3D 显示器/SBS 输出。
- 后续本地文件输出或录像输出。

不建议默认用于：

- 游戏。
- VR 控制器强交互。
- 鼠标拖拽、绘图、精确点击。
- 任何用户明确要求低延迟的场景。

## 推荐架构

建议在 capture 与 presentation 之间新增一个通用 buffered presentation layer。

```text
Capture
  -> FrameTimestamp
  -> Runtime input queue
  -> Stereo runtime / depth runtime
  -> QualityBufferedOutputController
       -> FrameWindow
       -> TemporalDepthStabilizer
       -> LookaheadHoleFiller
       -> PresentationScheduler
  -> Output Adapter
       -> OpenXR
       -> Local Viewer
       -> 3D Monitor
```

关键点：

- capture 帧必须携带单调时钟 timestamp。
- buffered controller 不应阻塞 capture 线程。
- frame window 应该有明确上限，避免缓存无限增长。
- 实时高画质缓冲严禁把画面帧落盘；硬盘缓存只允许用于离线转换，不允许用于 OpenXR/本地实时输出。
- 缓存优先级必须是 GPU resident buffer，其次才是系统内存 ring buffer。
- 如果 GPU 缓存和内存缓存都无法满足目标延迟/帧窗口，应直接判定高画质缓冲模式不可用，并回退或提示用户关闭该模式；不能退化成硬盘缓存。
- 输出按 presentation timestamp 调度，而不是“处理完立刻显示”。
- 当前实时模式继续使用 latest-frame 队列。
- 高画质模式内部可以按顺序处理帧，但对外仍要有 backpressure 和降级策略。

## 补洞算法重评估

启用缓冲后，应重新评估补洞算法，而不是只复用实时模式的最快策略。

建议分层实现：

### 1. 当前帧 edge-aware 补洞

保留当前已存在或类似的边缘感知补洞策略，用于低成本修补小洞。

目标：

- 保持主体边缘干净。
- 避免颜色跨边界污染。
- 对字幕、窗口边缘、人物轮廓做保护。

### 2. Layered DIBR

将前景、背景、遮挡边界分层处理，避免单层视差 warp 导致边缘撕裂。

目标：

- 提升大视差场景的稳定性。
- 减少前景边缘拉伸。
- 为后续 temporal/lookahead 补洞提供 mask。

### 3. Temporal history fill

使用历史帧寻找可用背景颜色和纹理。

目标：

- 稳定慢速运动场景。
- 降低补洞区域闪烁。
- 对重复纹理、桌面窗口、视频背景更友好。

### 4. Future-frame lookahead fill

这是缓冲模式的主要画质收益来源。

输出第 N 帧时，可读取 N+1、N+2 等未来帧的背景信息，填补当前帧因为视差生成暴露出来的遮挡区域。

目标：

- 处理镜头横移和物体运动造成的空洞。
- 提升人物边缘、车辆边缘、室内场景的背景连续性。
- 降低实时模式里只能“猜”的区域比例。

### 5. Multi-scale push-pull / diffusion fallback

对历史帧和未来帧都无法可靠恢复的区域，使用多尺度扩散或 push-pull 作为兜底。

目标：

- 避免黑洞、透明洞、硬边界残留。
- 保证任何输入都能输出完整画面。
- 对画质模式可接受更高计算开销。

### 6. Confidence blend

补洞不能只追求“填满”，还要保留置信度。

建议每个候选来源带 confidence：

```text
current edge-aware fill
history fill
future fill
push-pull fallback
```

最终按 mask、depth edge、motion consistency、temporal stability 做融合。

## 延迟与缓存策略

这里的缓存是实时内存/显存缓存，不是离线文件缓存。高画质缓冲输出面向 OpenXR、本地窗口和 3D 显示器实时播放，不能为了保帧把画面写到硬盘。硬盘 I/O 会引入不可控延迟、抖动和寿命问题，也会破坏实时输出的可预测性。

缓存优先级：

```text
1. GPU resident buffer：优先保存 GPU tensor / texture / depth / mask / confidence 等必要中间结果。
2. System memory ring buffer：仅在显存不足或跨设备输出需要时作为降级缓存。
3. Disk cache：实时模式禁止使用；只允许离线转换、depth cache、最终视频渲染使用。
```

判定规则：

```text
GPU buffer 可用 -> 启用高画质缓冲。
GPU buffer 不足但内存 ring buffer 可稳定满足目标延迟 -> 降级启用，并在 GUI 标注。
GPU 与内存都无法稳定满足 -> 高画质缓冲启动失败，提示降低延迟/分辨率/补洞质量，或关闭该模式。
绝不自动退化为硬盘帧缓存。
```

建议从固定延迟开始，后续再做自适应延迟。

第一阶段：

```text
delay_ms = 用户配置值
frame_window = ceil(delay_ms / frame_interval) + lookahead_margin
```

例如：

```text
60 FPS, 150 ms 延迟
-> 约 9 帧基础缓存
-> 可使用 N+1 到 N+3 的 lookahead
```

第二阶段可以增加自适应策略：

- 算法耗时升高时自动增加缓冲。
- 输出掉帧时降低补洞等级。
- 用户切换到交互场景时提示关闭或自动降级。

必须设置硬上限：

```text
最大视频延迟：建议 500 ms
最大 frame window：建议 30 帧以内
超限策略：丢弃最旧帧，保留时间连续性
```

## 音频同步可行性

高画质缓冲输出会引入视频延迟，因此必须规划音频同步。否则看视频时会出现明显音画不同步。

同步关系建议：

```text
video_delay_ms = quality_buffer_delay_ms + processing_delay_ms
audio_delay_ms = video_delay_ms + user_audio_offset_ms
```

其中：

- `quality_buffer_delay_ms` 是用户选择的缓冲延迟。
- `processing_delay_ms` 是 runtime 测量到的实际处理延迟。
- `user_audio_offset_ms` 用于手动校准设备差异。

用户已经明确要求：新音频链路不要复用 RTMP 现有采集。

这是合理的。当前 RTMP 音频路径更接近“FFmpeg 推流参数拼装”，适合 RTMP 输出，但不适合作为 OpenXR、本地输出、3D 显示器输出的通用音频时钟层。

推荐新增独立音频子系统：

```text
AudioCapture
  -> AudioFrame(timestamp, pcm)
  -> AudioClock
  -> AudioDelayBuffer
  -> AudioOutput
```

RTMP 后续可以反过来消费这个通用音频层，但高画质缓冲输出不应依赖 RTMP 模块。

## 音频采集方案调查

### Windows WASAPI loopback

优先级最高。

Microsoft 官方 WASAPI loopback 支持捕获正在由 render endpoint 播放的音频流，并且不依赖用户启用 `Stereo Mix` 这类硬件 loopback 设备。官方文档还说明，硬件 loopback 设备名称不统一，例如 `Stereo Mix`、`Waveout Mix`、`What You Hear`，而 WASAPI loopback 本身可以避免这类用户配置问题。

适合作为 Windows 主实现：

```text
默认扬声器输出
-> WASAPI loopback capture
-> timestamped PCM
-> delay buffer
-> output sync
```

风险：

- 需要处理不同采样率、声道数和系统混音格式。
- DRM/受保护内容可能无法被 loopback 捕获。
- Python 层封装选择需要实测。

参考：<https://learn.microsoft.com/en-us/windows/win32/coreaudio/loopback-recording>

### sounddevice / PortAudio WASAPI

项目当前已有 `sounddevice` 依赖，因此它是最低依赖成本的候选。

`sounddevice` 提供 `WasapiSettings`，可设置 WASAPI exclusive、auto_convert、explicit_sample_format 等平台参数。它适合快速做设备枚举、输入输出验证和原型测试。

但需要注意：文档中的 platform-specific settings 说明的是 WASAPI 参数，不等同于高层 loopback API 已经完整封装。是否能稳定捕获系统输出，需要在本项目内做实际 spike。

参考：<https://python-sounddevice.readthedocs.io/en/latest/api/platform-specific-settings.html>

### pyminiaudio

`pyminiaudio` 暴露 miniaudio 后端，API 中包含 `WASAPI`、`COREAUDIO`、`PULSEAUDIO` 等 backend 枚举，并提供 loopback 支持检测能力。它适合作为独立音频引擎候选，尤其是希望未来跨平台统一音频采集/播放时。

优势：

- 轻量。
- 回调式音频模型适合 ring buffer。
- 有 backend 能力检测。

风险：

- 需要新增依赖。
- 需要验证 Windows WASAPI loopback 的稳定性、延迟和打包体积。

参考：<https://github.com/irmen/pyminiaudio>

### SoundCard

`SoundCard` 是跨平台音频库，支持 Windows/WASAPI、macOS/CoreAudio、Linux/PulseAudio，并提供 `include_loopback` 相关接口。它适合做候选调研。

但其文档也列出 Windows/WASAPI 已知问题，例如单声道录制异常、部分情况下 blocksize 被忽略、可能 underrun。因此不建议直接拍板为主路线，除非实测结果明显优于其它方案。

参考：<https://soundcard.readthedocs.io/en/latest/>

## 推荐音频路线

短期建议：

```text
Windows:
  优先调研 WASAPI loopback
  先用 sounddevice 做最小 spike
  如果 loopback 不稳定，再评估 pyminiaudio 或原生 WASAPI binding

macOS:
  先列为后续调研
  CoreAudio 系统音频捕获涉及权限和系统版本差异

Linux:
  后续按 PipeWire/PulseAudio 路线调研
```

实现原则：

- 音频帧必须带 timestamp。
- 音频延迟用 ring buffer，不用简单 `sleep()`。
- GUI 暴露自动同步和手动偏移。
- 任何音频采集失败都不能影响视频实时模式启动。
- 受保护内容无法采集时要给出明确状态提示。

## 本地模式输出

高画质缓冲输出应支持本地模式。它不仅是 OpenXR 功能。

本地输出有两个价值：

1. 作为开发和调试路径，可以更容易观察补洞质量。
2. 对非 VR 用户，本地播放器/3D 显示器也能受益于高画质缓冲。

本地模式建议支持：

- 本地实时窗口延迟播放。
- SBS 本地输出。
- 未来可选本地文件输出。
- GUI 中显示当前缓冲帧数、视频延迟、音频偏移、补洞策略。

本地模式的调试指标应比 OpenXR 更完整：

```text
capture_ts
runtime_start_ts
runtime_done_ts
presentation_ts
buffer_depth_frames
video_delay_ms
audio_delay_ms
hole_fill_strategy
lookahead_frames_used
fallback_area_ratio
```

## 风险评估

### 延迟风险

用户会明显感知到输入延迟。因此必须默认关闭，并在 GUI 文案中明确适合视频/影视，不适合游戏/交互。

### 内存风险

4K 帧缓存占用很高。必须限制 frame window，并尽量缓存 GPU tensor、GPU texture 或压缩后的必要中间结果，而不是无上限保存完整 CPU numpy 帧。系统内存只能作为实时 ring buffer 降级路径，不能扩展成硬盘缓存。

### 硬盘缓存风险

实时高画质缓冲不允许使用硬盘缓存。硬盘缓存适合离线转换、depth cache、抽帧和最终视频编码，但不适合 OpenXR/本地实时输出。只要实现需要把实时画面帧写入硬盘才能维持缓冲，就应判定该实时方案失败。

### GPU 显存风险

如果保存多帧 depth、left/right eye、mask、confidence map，显存会快速增长。需要按补洞阶段释放临时对象，并提供低显存降级策略。

### 同步风险

音频同步不能依赖实时线程 sleep。必须用 timestamp 和 ring buffer，否则会随着帧率抖动逐步漂移。

### 质量风险

future-frame lookahead 可能在快速运动、切镜头、字幕滚动时引入错误背景。需要 shot-change / scene-cut 检测和 confidence 降级。

### 架构风险

如果直接修改现有 latest-frame 队列，会破坏实时模式。必须将缓冲作为独立分支插入，并保留现有实时路径。

## 分阶段落地计划

### 阶段 1：设计与开关

- 增加 GUI 勾选项：高画质缓冲输出。
- 增加 settings 字段：
  - `quality_buffered_output_enabled`
  - `quality_buffer_delay_ms`
  - `quality_buffer_hole_fill_quality`
  - `quality_buffer_audio_sync_enabled`
  - `quality_buffer_audio_offset_ms`
- 默认关闭。
- 文档标注适用场景和延迟代价。

### 阶段 2：视频缓冲骨架

- 新增 `QualityBufferedOutputController`。
- 按 timestamp 缓存 runtime result。
- 实现固定延迟 presentation scheduler。
- OpenXR、本地输出共用同一缓冲层。
- 不改变默认实时模式。

验收：

- 关闭开关时行为与当前一致。
- 开启后可稳定延迟输出。
- GUI 显示当前延迟和缓存帧数。

### 阶段 3：高画质补洞

- 接入 layered DIBR。
- 接入 temporal history fill。
- 接入 future-frame lookahead fill。
- 增加 confidence blend。
- 增加场景切换检测和 fallback。

验收：

- 视频横移场景空洞更少。
- 人物/字幕边缘闪烁下降。
- 大面积空洞不出现黑洞或硬边。

### 阶段 4：独立音频链路

- 调研并 spike Windows WASAPI loopback。
- 比较 sounddevice、pyminiaudio、SoundCard。
- 新增 `AudioCapture / AudioDelayBuffer / AudioClock`。
- GUI 暴露音频同步状态和手动偏移。

验收：

- 视频延迟 150/250/500 ms 时，音频可同步延迟。
- 采集失败时视频模式仍可运行，并显示清晰错误。
- 不依赖 RTMP 音频采集。

### 阶段 5：质量评估与自动策略

- 增加补洞面积统计。
- 增加 lookahead 使用率统计。
- 增加 GPU 显存占用监控。
- 增加自适应降级：
  - 显存不足时减少 lookahead。
  - 帧率不足时降低补洞质量。
  - 场景切换时重置 temporal buffer。

## 建议的验收标准

功能验收：

- GUI 有明确开关，默认关闭。
- OpenXR 和本地输出都能启用。
- 关闭后实时路径行为不变。
- 开启后输出延迟稳定且可观察。

画质验收：

- 横向镜头运动视频中，遮挡空洞少于实时模式。
- 人物边缘、字幕边缘、窗口边缘闪烁少于实时模式。
- 快速切镜头不产生明显错误残影。

性能验收：

- 150 ms 延迟档可在目标设备上连续运行。
- 缓冲帧数有上限。
- 显存不足时可降级而不是崩溃。

音频验收：

- 音频延迟随视频缓冲自动调整。
- 用户可手动微调 offset。
- 采集失败有明确提示。

## 最终建议

建议推进该方案，但要严格分阶段实施。

第一版不要急于实现最复杂的 future-frame 补洞。应先完成可选开关、视频缓冲骨架、本地/OpenXR 双输出、延迟统计。确认 presentation 层稳定后，再逐步引入 lookahead 补洞和独立音频链路。

这条路线与当前实时模式不冲突。实时模式继续服务低延迟场景，高画质缓冲模式服务视频和影视场景。两者并存，比试图用一个模式同时满足低延迟和最高画质更稳妥。
