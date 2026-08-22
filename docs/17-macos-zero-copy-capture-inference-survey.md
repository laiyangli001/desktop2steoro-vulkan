# macOS ScreenCaptureKit 到 MPS/CoreML 零拷贝实现调查报告

## 结论

macOS 原生技术栈支持一条不经过 CPU `numpy` 的捕获到推理路线：

```text
ScreenCaptureKit
-> CVPixelBuffer / IOSurface
-> CVMetalTextureCacheCreateTextureFromImage
-> MTLTexture
-> Metal / MPS preprocess
-> MPSGraph 或 CoreML inference
```

但本项目当前没有实现这条链路。当前实现仍是：

```text
ScreenCaptureKit / CoreGraphics
-> CPU 可读 buffer
-> numpy
-> torch tensor
-> .to("mps")
-> PyTorch MPS 推理
```

也就是说，目前 macOS 是 MPS GPU 推理，不是捕获到推理的 GPU 零拷贝。

## 项目现状

相关文件：

```text
src/capture/backends/macos_screencapturekit.py
src/capture/backends/macos_coregraphics.py
src/stereo_runtime/providers/apple/pytorch_mps.py
```

当前 `macos_screencapturekit.py` 使用 PyObjC 调 ScreenCaptureKit，但帧处理走：

```text
CMSampleBuffer
-> CVPixelBufferGetBaseAddress
-> numpy ndarray
```

`macos_coregraphics.py` 也是 CPU / numpy 路线。

Apple provider 目前只有 PyTorch MPS provider：

```text
pytorch_mps.py
```

它把模型和 tensor 放到 `mps` 设备上执行，但输入来自前面 CPU/numpy 管线后再 `.to("mps")`。

当前没有看到真正的 CoreML depth provider，也没有看到 `MTLTexture`、`CVMetalTextureCache`、`MPSGraph`、`MLModel` 原生桥接实现。

## 每段技术成熟度

### ScreenCaptureKit 捕获

成熟。ScreenCaptureKit 是 Apple 官方屏幕/窗口捕获 API，用于替代更老的截图方式。它输出的 sample buffer 可以携带 `CVPixelBuffer`。

当前项目已经通过 PyObjC 加载并使用 ScreenCaptureKit，因此“Python 调用 macOS 捕获 API”这一步已经成立。

### CVPixelBuffer / IOSurface

成熟。`CVPixelBuffer` 可以由 IOSurface-backed memory 支撑，是视频、图像和 GPU 互操作的基础对象。

对于零拷贝路线，关键不是读取 `CVPixelBufferGetBaseAddress`，而是避免 lock base address 后读 CPU 内存，改为把 pixel buffer 转成 Metal texture。

### CVPixelBuffer 到 MTLTexture

成熟。官方推荐入口是：

```text
CVMetalTextureCacheCreateTextureFromImage
```

它可以从 `CVPixelBuffer` 创建 `CVMetalTexture`，再取出 `MTLTexture`。

目标链路：

```text
CVPixelBuffer
-> CVMetalTextureCacheCreateTextureFromImage
-> CVMetalTextureGetTexture
-> MTLTexture
```

这一步仍需要原生生命周期管理：texture cache、pixel format、plane index、width/height、frame 生命周期。

### Metal / MPS 预处理

成熟。模型输入通常不是屏幕原始 BGRA texture，而是 resize、normalize、layout 后的 tensor。推荐用 Metal compute shader 或 MPS/MPSGraph 做：

```text
BGRA/RGBA texture sample
resize
normalize
NHWC/NCHW layout transform
FP16/FP32 tensor output
```

这一步不是 CPU 拷贝，但通常会有一次 GPU shader/compute 写入。应称为“零 CPU 回读”，不要宣称“完全无 GPU 拷贝”。

### CoreML / MPS 推理

成熟，但工程路径不同：

```text
CoreML
= 高层模型运行时，适合部署 .mlmodel/.mlpackage

MPS / MPSGraph
= 更低层 Metal GPU 计算图/算子路线，控制更强

PyTorch MPS
= Python/PyTorch 生态路线，易集成，但不等于捕获零拷贝
```

如果目标是最少改动，PyTorch MPS 继续保留即可。如果目标是捕获 texture 到推理零 CPU 回读，CoreML 或 MPSGraph 需要原生 bridge。

## Python 调用方式

Python 已经有成熟方式调用 macOS Objective-C framework：

```text
PyObjC
```

项目当前已经使用 PyObjC：

```python
objc.loadBundle("ScreenCaptureKit", ...)
import ScreenCaptureKit as SCK
```

因此可以继续用 Python 调原生 API。但完整零拷贝 pipeline 不建议全写在 Python/PyObjC 里，原因是：

- Metal object 生命周期复杂。
- command queue / command buffer / fence 同步需要严谨管理。
- shader dispatch 和 texture/tensor binding 用 PyObjC 写会很笨重。
- 性能路径不应频繁跨 Python/Objective-C 边界。

推荐形态：

```text
Python 薄调用
-> Objective-C++ / Swift / Rust native bridge
-> bridge 内部管理 ScreenCaptureKit + Metal + MPS/CoreML
```

Python 层只暴露类似：

```python
bridge.start_capture(...)
depth = bridge.next_depth(timeout_ms=...)
bridge.stop()
```

## 推荐实现路线

## 验证前提

这条路线可以先在 Windows 上做方案设计和接口拆分，但不能完整验证。真正验证至少需要一台 macOS 机器，因为以下环节都依赖本机 Apple framework、权限和 GPU runtime：

```text
ScreenCaptureKit 权限和 run loop
CVPixelBuffer / IOSurface 生命周期
MTLTexture 引用和释放时机
Metal command queue / command buffer 同步
CoreML / MPSGraph 在目标芯片上的算子和性能
打包、签名、macOS 版本兼容
```

如果目标是可用的零 CPU 回读 demo，建议用 Apple Silicon Mac 验证。Intel Mac 可以做部分 API 验证，但性能、ANE、CoreML 后端行为和最终目标不同。

### 里程碑 1：保留现状，标注非零拷贝

目标：避免误导。

工作项：

- 文档明确当前 macOS MPS 是 GPU 推理，不是捕获零拷贝。
- 日志或 provider info 可标记：`capture_zero_copy=False`。
- 不改现有 PyTorch MPS 路线。

### 里程碑 2：PyObjC 验证 CVPixelBuffer 到 MTLTexture

目标：证明捕获帧能不读 CPU base address，转换成 `MTLTexture`。

工作项：

- 从 ScreenCaptureKit callback 拿 `CMSampleBuffer`。
- 获取 `CVPixelBuffer`。
- 创建 `CVMetalTextureCache`。
- 调 `CVMetalTextureCacheCreateTextureFromImage`。
- 取 `MTLTexture`。
- 输出 texture width/height/pixel format 诊断。

该阶段只做验证，不接模型。

### 里程碑 3：Metal GPU 预处理 demo

目标：把 `MTLTexture` 转成模型输入 buffer。

工作项：

- 写最小 Metal compute shader。
- 输入 BGRA/RGBA texture。
- 输出 FP16/FP32 tensor buffer。
- 实现 resize / normalize / layout transform 中的最小可用子集。
- 与 CPU 预处理结果比较误差。

### 里程碑 4：CoreML 或 MPSGraph 推理桥

目标：原生端完成推理，Python 只拿结果。

可选方案：

```text
A. CoreML：转换模型到 mlpackage，使用 CoreML runtime
B. MPSGraph：手写/加载计算图，控制更强但工作量更大
C. 继续 PyTorch MPS：最省事，但 texture 零拷贝输入很难完整闭环
```

推荐先走 CoreML，因为 Apple 生态部署更成熟。

第一阶段固定使用 Distill Depth / Distill-Any-Depth Base，不把 InfiniDepth 放进首版范围。

```text
Distill Depth / Distill-Any-Depth Base
-> PyTorch / ONNX
-> CoreML mlpackage
-> CoreML 推理验证
```

原因：

- Distill Depth 类模型结构更常规，PyTorch -> ONNX -> CoreML 成功率更高。
- InfiniDepth 结构更复杂，包含 DINOv3 相关块、implicit decoder、动态 shape 风险，ONNX 阶段已经出现 TracerWarning，CoreML 转换风险更高。
- 零拷贝链路先验证捕获、Metal preprocess、CoreML runtime 是否闭环；模型难度不应一开始成为主阻塞。

这里的“Distill Depth 更简单”只解决模型转换难度，不等于零拷贝 pipeline 已经完成。完整闭环还需要把捕获帧保持在 `CVPixelBuffer` / `MTLTexture` 路径里，并在原生层完成预处理、推理和输出同步。

InfiniDepth 转 CoreML 可能单独卡几天。若 CoreML 不支持关键算子，才考虑 MPSGraph/Metal 自定义推理。

### 里程碑 5：封装 native bridge

目标：提供项目可调用的最小 Python API。

建议 API：

```python
class MacOSMetalDepthBridge:
    def start(self, target): ...
    def next_depth(self, timeout_ms: int): ...
    def stop(self): ...
```

桥内部管理：

```text
ScreenCaptureKit stream
CVMetalTextureCache
MTLDevice / MTLCommandQueue
Metal preprocess pipeline
CoreML / MPSGraph model
同步和资源池
```

## Distill Depth 时间表

目标是先做一个 Distill Depth 版 macOS 原生零 CPU 回读原型：

```text
ScreenCaptureKit
-> CVPixelBuffer / IOSurface
-> MTLTexture
-> Metal preprocess
-> CoreML Distill Depth
-> depth output
-> Python bridge
```

| 阶段 | 目标 | 交付物 | 预估 |
|---|---|---|---|
| 0 | 准备 Mac 验证环境 | 可运行项目、捕获权限、Xcode/Command Line Tools、Python 环境 | 0.5 - 1 天 |
| 1 | Distill Depth 转 CoreML | `.mlpackage`，固定输入尺寸，单张图片输出 depth | 1 - 2 天 |
| 2 | CoreML 单帧推理接入 | 原生或 Python 测试入口，输入图片得到 depth buffer | 1 - 2 天 |
| 3 | ScreenCaptureKit 原生捕获 | 不读 CPU base address，拿到 `CVPixelBuffer` / `IOSurface` | 1 - 2 天 |
| 4 | `CVPixelBuffer` 转 `MTLTexture` | `CVMetalTextureCacheCreateTextureFromImage` 验证，输出 texture 诊断 | 1 - 2 天 |
| 5 | Metal 预处理 | resize、normalize、layout，输出 CoreML 可吃的输入 buffer | 2 - 4 天 |
| 6 | Capture -> Metal -> CoreML 闭环 | 实时帧直接进入 CoreML，输出 depth，避免 CPU 图像回读 | 2 - 4 天 |
| 7 | Python bridge 接入项目 | `bridge.start()` / `bridge.next_depth()` / `bridge.stop()`，接入现有 runtime | 1 - 2 天 |
| 8 | 稳定性整理 | 资源生命周期、同步、错误处理、权限提示、打包说明 | 1 - 3 天 |

合计：`10.5 - 22` 个 AI 工作日。这个估算假设有真实 Apple Silicon Mac 可以连续验证，并且 Distill Depth 转 CoreML 没有关键算子阻塞。

最短可演示版本可以压缩到 `5 - 8` 个 AI 工作日，但只建议作为 demo：

```text
固定显示器捕获
固定输入尺寸
固定模型
只输出 depth 诊断图
不完整接入现有 GUI / OpenXR pipeline
```

InfiniDepth 不建议放进第一阶段。它的 CoreML 转换风险应单独评估，可能额外消耗数天到更久，取决于算子、动态 shape 和精度差异。

## 不建议现在做的事

- 不建议把完整 Metal/MPS/CoreML pipeline 全写成 PyObjC Python 代码。
- 不建议把 PyTorch MPS 路线称为零拷贝。
- 不建议为了“未来零拷贝”重写当前可运行的 macOS capture provider。
- 不建议在没有模型转换验证前承诺 CoreML 性能。
- 不建议在第一阶段用 InfiniDepth 验证 CoreML；先用 Distill Depth / Distill-Any-Depth Base 跑通链路。

## 命名建议

当前路线：

```text
pytorch_mps
```

未来原生零 CPU 回读路线可命名为：

```text
macos_metal_coreml
macos_screencapturekit_metal
macos_metal_mpsgraph
```

不要叫 `xpu`，`XPU` 在项目里是 Intel PyTorch XPU，不是 macOS。

## 风险

- ScreenCaptureKit 权限和窗口/显示器捕获策略会影响可用性。
- CVPixelBuffer pixel format 可能和模型预处理期望不同。
- CoreML 模型转换可能遇到算子、动态 shape、精度问题；InfiniDepth 风险高于 Distill Depth。
- MPSGraph/Metal 路线开发成本高，但控制力最好。
- 原生 bridge 打包要处理 macOS 架构、签名、framework 链接。

## 参考资料

- ScreenCaptureKit: https://developer.apple.com/documentation/screencapturekit
- CoreVideo CVMetalTextureCache: https://developer.apple.com/documentation/corevideo/cvmetaltexturecache
- CVMetalTextureCacheCreateTextureFromImage: https://developer.apple.com/documentation/corevideo/cvmetaltexturecachecreatetexturefromimage
- Metal Performance Shaders MPSImage: https://developer.apple.com/documentation/metalperformanceshaders/mpsimage
- Core ML: https://developer.apple.com/documentation/coreml
- PyObjC: https://pyobjc.readthedocs.io/
