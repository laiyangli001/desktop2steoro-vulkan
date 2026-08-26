  请基于当前项目代码及 [19-cross-platform-capture-inference-stereo-backend-matrix.md](desktop2stereo-vulkan/docs/19-cross-platform-capture-inference-stereo-backend-matrix.md) 第 10 节“当前实现与目标架构的主要缺口”，逐项核查并补齐尚未完成的架构能力。

  macOS原生链路暂不实现，相关功能已有其他项目负责。本次必须排除：

  - CoreML深度推理。
  - MPSGraph推理桥。
  - CVPixelBuffer/IOSurface到MTLTexture的零回读桥。
  - Metal立体合成。
  - Intel Mac相关适配。
  - 不修改 `17-macos-zero-copy-capture-inference-survey.md`。
  - 除非为了避免跨平台导入错误，否则不要修改macOS专用代码。

  ## 一、开始前检查

  1. 检查当前分支、工作区状态和已有未提交修改，保留用户现有改动。
  2. 阅读第10节列出的缺口，并逐项对照当前代码、测试、依赖和规格文档。
  3. 为每项缺口标记：
     - 已实现并验证。
     - 部分实现。
     - 尚未实现。
     - 需要真实硬件验证。
     - 本次排除的macOS功能。
  4. 不得把规划目标、CI编译成功或能力探针通过描述成真机功能已经验证。

  ## 二、本次需要补齐的功能

  ### 1. Windows通用DirectML闭环

  建立并验证以下目标链路：

  ```text
  Desktop Duplication / Windows Graphics Capture
  → D3D11 Texture
  → D3D11/D3D12共享资源或一次GPU复制
  → DirectML推理
  → Vulkan立体合成
  ```

  要求：

  - DirectML只作为推理后端，不得命名为捕获模式。
  - 优先共享GPU资源；无法共享时允许一次GPU内部复制。
  - 禁止静默回读CPU。
  - 检测模型不支持算子、隐式CPU fallback及设备初始化失败。
  - DirectML失败时明确回退CPU。

  ### 2. Windows Graphics Capture GPU资源输出

  改造通用 `WindowsCapture`，使其能够保留并输出D3D11纹理，而不是默认转换成CPU NumPy图像。

  同时保留现有CPU兼容路径，并通过统一帧契约明确记录：

  - 资源类型和格式。
  - Adapter LUID。
  - `gpu_to_cpu`。
  - `gpu_copy_count`。
  - `zero_copy`。
  - 资源所有权和生命周期。

  `DesktopDuplication`继续定位为跨厂商的DXGI/D3D11显示器捕获工具，不得声称原生支持单窗口捕获。

  ### 3. NVIDIA推理自动回退

  NVIDIA推理只保留以下顺序：

  ```text
  原生TensorRT
  → PyTorch CUDA
  → DirectML（仅Windows）
  → CPU
  ```

  删除自动链中的ORT TensorRT和ONNX CUDA。Linux不支持DirectML时应自动跳过，实际顺序为：

  ```text
  原生TensorRT
  → PyTorch CUDA
  → CPU
  ```

  ### 4. 立体合成自动选择

  Windows和Linux目标顺序统一为：

  ```text
  Vulkan
  → OpenGL
  → Torch/CPU
  ```

  核查现有Triton、Vulkan和Torch选择逻辑。若Triton仍有保留价值，应明确它是内部优化还是兼容实现，不得与深度推理后端混淆。

  补齐真实的OpenGL合成能力探测、初始化失败处理和回退日志；如果当前缺少完整OpenGL实现，应先报告缺失组件和实施方案，再进行修改。

  ### 5. Intel Windows闭环

  补齐并验证以下已有组件之间的连接：

  - Desktop Duplication。
  - D3D11 Texture。
  - OpenVINO GPU RemoteTensor。
  - Vulkan/D3D11共享资源。
  - D3D11/oneVPL最终输出。

  必须校验所有设备的Adapter LUID。任何设备不一致、LUID无效、格式不兼容或同步条件不完整的情况，都必须拒绝共享资源路径并明确回退。

  没有Intel真机时，只能完成代码、自动化测试、探针和远程编译验证；真机运行、4K长时间稳定性及严格零回读必须继续标记为“待验证”。

  ### 6. Linux捕获路径

  核查当前MSS + Xlib实现，并设计或实现Linux原生GPU捕获路径。

  重点考虑：

  - X11和Wayland差异。
  - PipeWire。
  - DMA-BUF。
  - DRM render node和PCI BDF。
  - Vulkan device UUID。
  - 显式及隐式同步。
  - 显示器捕获与窗口捕获的能力边界。

  如果当前环境无法验证Wayland、PipeWire或DMA-BUF，不得推测为已支持，应保留MSS CPU路径作为兼容回退并标记新路径“待验证”。

  ### 7. 其他及国产Windows GPU

  建立DirectML和Vulkan能力探测，不得仅凭DX12或Vulkan API可用就判定完整路径受支持。

  至少检测：

  - DirectML设备创建。
  - 模型算子支持情况。
  - 是否发生隐式CPU fallback。
  - D3D11/D3D12共享资源能力。
  - Vulkan外部内存导入。
  - Adapter一致性。
  - GPU复制次数。

  对未经真机测试的GPU统一标记“待验证”，并为后续白名单、黑名单和驱动兼容记录预留结构。

  ### 8. 统一可观测性

  所有平台路径统一输出以下诊断信息：

  - 操作系统和GPU厂商。
  - 捕获模式和实际捕获工具。
  - 深度推理后端。
  - 立体合成后端。
  - Adapter LUID、UUID、PCI BDF或等价设备身份。
  - 输入输出资源类型、格式和分辨率。
  - `gpu_to_cpu`。
  - `gpu_copy_count`。
  - `zero_copy`。
  - 每一级失败和回退原因。

  只有完整满足零CPU回读条件时才能报告 `zero_copy=true`。

  ### 9. GUI简化

  GUI只保留：

  - 捕获模式：显示器、窗口。
  - 捕获工具：自动及当前平台实际支持的捕获工具。

  不增加深度推理和立体合成后端下拉框。两类后端必须按照平台自动选择顺序依次尝试。

  GUI只读显示：

  - 实际启用的推理后端。
  - 实际启用的合成后端。
  - 是否发生回退。
  - 回退原因。
  - CPU回读、GPU复制次数及零回读状态。

  ### 10. 自动化与硬件回归矩阵

  增加可自动执行的能力探测和回归测试，覆盖：

  - Windows NVIDIA、AMD、Intel和其他DirectML GPU。
  - Linux NVIDIA、AMD、Intel及其他GPU。
  - CPU兼容路径。
  - 后端初始化失败和逐级回退。
  - Adapter不一致。
  - 资源格式或共享句柄不兼容。
  - 禁止误报零CPU回读。

  无法在当前环境完成的硬件测试，应生成明确的测试入口、预期结果和诊断日志要求，不得伪造测试通过结果。

  ## 三、实施约束

  - 优先复用现有捕获、推理、Vulkan、D3D11和资源契约代码。
  - 不新增会让用户混乱的厂商后端GUI选项。
  - 不把Vulkan作为默认深度推理后端。
  - 不删除CPU兼容回退。
  - 不静默切换后端。
  - 修改前检查影响范围，修改后检查完整Git差异。
  - Python文件运行 `py_compile`。
  - 执行相关单元测试、能力探针和 `git diff --check`。
  - 只修改本次缺口补齐直接相关的文件，不覆盖或清理用户已有修改。

  ## 四、交付结果

  完成后提供：

  1. 第10节每项缺口的完成状态对照表。
  2. 实际修改的代码和文档文件列表。
  3. 最终自动选择及回退顺序。
  4. 新增或修改的测试及结果。
  5. 仍需真实硬件验证的项目。
  6. 已知限制和后续工作。
  7. 同步更新 `19-cross-platform-capture-inference-stereo-backend-matrix.md`，使文档与最终代码一致。

  本次不要提交或推送，完成修改和验证后等待确认。
