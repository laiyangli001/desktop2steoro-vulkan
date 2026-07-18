# 第一批迁移清单

## 迁移原则

本仓库从`4k-stereo-synthesis-lab`复制可复用文件。源仓库中的文件不移动、不删除，新仓库也不通过`sys.path`或运行时导入依赖源仓库。

## 已复制

- `samples/`全部14个文件。
- `src/capture/`中的Python Capture实现，排除依赖Rust Native的DXGI Duplication路径。
- `src/gui/`中的Python GUI代码，排除内置Flet压缩包和本地客户端缓存。
- `src/stereo_runtime/`全部Git跟踪文件，包括Provider、模型实现、TensorRT、ROCm/MIGraphX、Triton和立体合成优化。
- `src/utils/`全部Python工具。
- `src/streaming/`中非Legacy模块。
- `src/xr_viewer/controllers/`和`src/xr_viewer/environments/`全部资产，GLB/HDR继续使用Git LFS。
- `src/xr_viewer/`中平台无关的输入、姿态、帧门控、时间和交互模块。
- Filament Bridge源码和版本清单迁移到`native/filament/`，关联GitHub Actions工作流同步迁移；多平台产物输出到`src/xr_viewer/native/`。
- 27个Capture、推理、模型和GPU合成相关脚本。
- 46个与已迁移模块相关的测试文件；旧启动器、旧渲染器和D3D11诊断断言不迁移。

## 明确未复制

- Panda3D及其Probe、Runtime和测试。
- D3D11 OpenXR、WGL/NV_DX和D3D互操作代码。
- 旧OpenGL Viewer、PBO/CUDA-GL上传器和相关视觉回归脚本。
- 包含ModernGL/D3D11布局的旧自研glTF渲染器；glTF场景由Filament Bridge接管。
- `src/capture/dxgi/native/`中的Rust扩展源码。
- 旧Filament OpenGL Bridge预编译DLL/SO/dylib和编译中间文件。
- `__pycache__`、日志、下载模型、内置Python环境、构建目录和本地输出。

## 已实现并通过头显实测

1. Python Vulkan Instance、Device、Graphics Queue、Command Pool、Fence和队列同步基础层。
2. 基于`XR_KHR_vulkan_enable2`的Python OpenXR Vulkan Session。
3. 双眼Vulkan交换链、Projection Layer和纯色清屏帧提交入口。
4. 独立能力探针与`src/tools/openxr_vulkan_smoke.py`实测工具。
5. Virtual Desktop OpenXR Runtime、RTX 3090、双眼3648x3648交换链的300帧实机验证。

## 待实现

1. Vulkan资源分配器、Compute Stereo Graph和SPIR-V Shader。
2. Filament Vulkan Render Target Bridge的CI编译和Python运行时联调。
3. Capture/Inference GPU资源到Vulkan的external-memory或一次GPU copy路径。
4. 新运行时装配、GUI配置Schema和OpenGL Fallback。
