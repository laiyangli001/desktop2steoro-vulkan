# 并行推理调度与跨后端实现方案

## 目标

在不改变现有单路实时模式的前提下，增加可选的双路推理实验模式。该模式将调度层与推理后端解耦，支持 NVIDIA CUDA/TensorRT 和 AMD ROCm/HIP/MIGraphX/ONNX Runtime，并保证 OpenXR/Vulkan 输出仍由单一 presenter 线程提交。

并行模式的目标不是承诺帧率翻倍，而是验证 GPU 尚有计算余量时，多个独立执行上下文能否提高 `processed_fps`。如果后端不支持安全并发，必须自动退回单 worker。

## 总体架构

```text
Capture
  -> FrameScheduler
       -> bounded pending queue (最多 2 个 in-flight frame)
       -> Worker 0
            -> backend execution context 0
            -> backend stream 0
       -> Worker 1
            -> backend execution context 1
            -> backend stream 1
       -> frame_id reorder buffer
       -> temporal state manager
       -> serial output queue
  -> OpenXR/Vulkan presenter
```

调度层只处理帧生命周期、顺序、容量和统计，不直接调用 CUDA、HIP、TensorRT 或 MIGraphX API。后端执行层负责创建上下文、流、输入输出缓冲和完成事件。

## 模块职责

### `FrameScheduler`

- 为每个输入帧分配单调递增的 `frame_id`。
- 将帧放入有界 pending queue。
- 最多允许两个 in-flight frame。
- 根据后端能力选择 1 或 2 个 worker。
- 收集 worker 完成结果并交给 reorder buffer。
- 队列满时优先丢弃尚未开始处理的中间帧。
- 记录输入、处理、重排、丢帧和输出统计。

### `InferenceWorker`

- 持有一个独立的 `BackendExecutionContext`。
- 持有该 worker 专属的临时 tensor、输出缓冲和 temporal state。
- 不访问其他 worker 的上下文或可变状态。
- 完成后返回 `FrameResult(frame_id, timestamp, payload, metadata)`。

### `FrameReorderBuffer`

- 按 `frame_id` 保存已完成但尚未轮到输出的结果。
- 只有下一个期望的 `frame_id` 到达时才提交结果。
- 等待超过延迟预算时，允许丢弃缺失帧并跳到最小可用的后续帧。
- 记录 `reorder_wait` 和 `frame_reorder_drop`。

### `TemporalStateManager`

- 每个 worker 使用独立的历史状态，禁止多个 worker 共享可变 temporal state。
- 如果算法要求严格顺序的历史状态，则在 reorder 后由 presenter-side state manager 顺序更新。
- 场景切换、分辨率变化、配置重建时清空所有 worker 状态。

### `OutputSubmitter`

- 由单独的 presenter 线程拥有 OpenXR swapchain、Vulkan graphics queue 和外部图像 lease。
- 负责 acquire、projection submit、quad layer submit、release。
- 推理 worker 不得直接调用 OpenXR/Vulkan presenter API。

## 后端抽象

建议定义以下后端接口，具体名称可按现有代码风格调整：

```python
class BackendExecutionFactory(Protocol):
    def capability(self) -> BackendConcurrencyCapability: ...
    def create_worker(self, worker_index: int) -> "BackendExecutionContext": ...


class BackendExecutionContext(Protocol):
    def submit(self, frame: RuntimeInput) -> BackendFuture: ...
    def poll(self, future: BackendFuture) -> bool: ...
    def collect(self, future: BackendFuture) -> BackendResult: ...
    def close(self) -> None: ...
```

`BackendConcurrencyCapability` 至少包含：

```text
supported: bool
max_workers: int
requires独立_context: bool
supports_async_stream: bool
supports_external_sync: bool
fallback_reason: string
```

### NVIDIA

```text
worker 0 -> CUDA stream 0 + TensorRT execution context 0
worker 1 -> CUDA stream 1 + TensorRT execution context 1
```

- 每个 TensorRT context 必须有独立输入输出缓冲。
- 每个 stream 使用独立完成 event。
- 不得在并行路径调用全局 `torch.cuda.synchronize()`。
- CUDA Graph 只有在每个 worker 拥有独立 graph 和静态缓冲时才能启用。

### AMD ROCm

```text
worker 0 -> HIP/PyTorch stream 0 + MIGraphX/ONNX session 0
worker 1 -> HIP/PyTorch stream 1 + MIGraphX/ONNX session 1
```

- MIGraphX 或 ONNX Runtime 必须确认 session 可并发执行。
- 如果 provider 只支持单 session，能力探测返回 `max_workers=1`。
- HIP/Vulkan external synchronization 需要独立验证，不能复用 CUDA semaphore 假设。
- 显存和利用率优先使用 ROCm SMI 或 provider 可用的统计接口。

## 帧顺序与丢帧策略

每个 `FrameResult` 必须携带：

```text
frame_id
capture_timestamp
runtime_start_timestamp
runtime_done_timestamp
worker_index
backend_name
```

规则：

1. pending queue 最大容量为 2 个 in-flight frame，默认不允许无限增长。
2. 新帧到达且容量已满时，优先丢弃尚未开始处理的中间帧。
3. 已提交到后端的任务不强行取消，等待完成或在超时后丢弃其结果。
4. reorder buffer 超过最大深度时，丢弃最旧的未提交结果并记录原因。
5. temporal 模式不能输出跨越场景切换的旧状态；发生 scene cut 时清空 reorder buffer 和 worker state。
6. 后端不支持并发、创建第二 worker 失败或显存不足时，自动回退单 worker。

建议硬上限：

```text
max_inflight = 2
max_reorder_depth = 2
max_buffer_delay_ms = 500~2000
```

## 统计与日志

必须区分“提交频率”和“真实生成频率”：

```text
input_fps
processed_fps
present_fps
parallel_enabled
active_workers
pending_depth
frame_reorder_wait
frame_reorder_drop
frame_inflight_drop
backend_fallback
```

推理性能指标：

```text
rt_depth_model
rt_gpu_total
gpu_utilization
vram_used
vram_total
```

日志示例：

```text
[ParallelInference]
enabled=1 workers=2 backend=tensorrt_native
input_fps=60.0 processed_fps=31.2 present_fps=72.0
pending_depth=2 reorder_wait=0.8 reorder_drop=0 inflight_drop=4
rt_depth_model=28.4ms rt_gpu_total=39.7ms gpu_utilization=94% vram=7.1/12.0GB
```

`present_fps` 不得把重复显示误报为 `processed_fps`。如果双 worker 没有提高处理吞吐，必须保留真实数据并自动回退，不得伪造性能提升。

## 与现有 OpenXR/Vulkan 的集成边界

- `process_openxr_frame()` 应拆成可独立调度的“输入/深度阶段”和“立体输出阶段”。
- 深度阶段可以由 FrameScheduler 分发到多个 worker。
- 需要顺序 temporal 状态的立体输出阶段，应在 reorder 后统一执行，或为每个 worker 使用隔离 state。
- OpenXR `wait_frame/begin_frame/end_frame`、swapchain acquire/wait/release 和 Vulkan graphics submit 保持在 presenter 线程。
- worker 只返回可消费的 GPU/CPU 资源和同步 token，不直接提交 OpenXR layer。
- CUDA、HIP 与 Vulkan 的 semaphore、image layout、queue ownership 必须由各 backend adapter 明确声明。

## 降级与安全策略

启动并行模式时依次执行：

1. 探测后端是否支持独立执行上下文。
2. 创建 worker 0。
3. 尝试创建 worker 1 和独立 stream/session/buffer。
4. 任一步失败则关闭 worker 1，继续单 worker 运行，并记录 `backend_fallback`。
5. 运行中出现设备丢失、同步错误、显存不足或连续超时，立即清空未提交结果并回退单 worker。

不得因为并行实验导致默认单路模式无法启动，也不得使用硬盘缓存来掩盖 pending queue 增长。

## 分阶段实现

### 阶段 1：调度骨架

- 新增 `FrameScheduler`、`FrameResult`、`FrameReorderBuffer`。
- 保持默认单 worker。
- 加入有界 pending、frame_id、重排和丢帧统计。

### 阶段 2：NVIDIA 双 worker

- 为 TensorRT context 和 CUDA stream 建立 worker adapter。
- 为每个 worker 分配独立缓冲和完成 event。
- 仅在 GUI `Parallel Inference` 开启时使用双 worker。

### 阶段 3：AMD ROCm adapter

- 增加 MIGraphX/ONNX Runtime 能力探测。
- 验证多 session 或 HIP stream 并发。
- 增加 ROCm SMI 显存/利用率统计。

### 阶段 4：OpenXR/Vulkan 输出接入

- 将重排后的结果交给现有 presenter。
- 保持 Vulkan 单提交者和现有 projection/quad 功能。
- 验证外部同步、图像 layout 和资源 release。

## 验收标准

- 关闭开关时，与当前单 worker 输出一致。
- 开启后日志明确显示 `parallel_enabled` 和 `active_workers`。
- `frame_id` 不乱序；发生丢帧时有计数和原因。
- pending 和 reorder buffer 永不超过硬上限。
- temporal 状态不跨 worker 互相污染。
- NVIDIA 无法并发时自动回退单 worker。
- AMD 无法安全创建第二 session/stream 时自动回退单 worker。
- OpenXR/Vulkan 仍由单一 presenter 提交，左右眼和资源 release 正常。
- 性能报告同时包含 `input_fps`、`processed_fps`、`present_fps`、`rt_depth_model`、`rt_gpu_total`、GPU 利用率、显存、pending、重排和丢帧数据。

## 当前实现状态（2026-08）

- NVIDIA native TensorRT 已创建最多两个 execution context、CUDA stream、独立输出缓冲和完成 event。
- OpenXR pipeline 已接入有界 `ThreadPoolExecutor` 深度调度：worker 只调用 `predict_openxr_depth()`，输出阶段仍在 pipeline/presenter 线程执行。TensorRT engine 可能在首帧输入尺寸确定后才创建，因此 pipeline 会在准备首帧后重新探测 `pipeline_slot_count`；检测到两个 slot 时延迟创建两个 worker，避免启动期错误降级为单路。
- 每个任务分配单调 `frame_id`，按提交顺序收集；pending 上限为 2，队列满时只取消尚未启动的新任务。
- GUI 的 `Parallel Inference` 位于高级立体参数之后，默认开启但可手动关闭。开关关闭、provider 少于两个 slot、启用 `profile_sync` 或后端能力不足时保持原单 worker 路径；当前不改变补洞、temporal 或 OpenXR layer 行为。
- `parallel_inference_workers`、`parallel_inference_pending`、`parallel_inference_dropped` 已附加到 runtime debug 数据并汇总到 `FPSBreakdown`。有效双路运行必须同时显示：`rt_parallel=1 rt_parallel_workers=2 rt_pending_limit=2`，并出现交替的 `rt_depth_slot=0/2` 与 `rt_depth_slot=1/2`。
- RTX 2060 实机动态内容对比：开启后处理/SBS 帧率较单路增加约 6~10 FPS；这是吞吐收益，不承诺翻倍，后续补洞、立体合成和 presenter 仍可能成为瓶颈。
