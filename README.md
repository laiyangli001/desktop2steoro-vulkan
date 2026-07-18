# Desktop2Stereo Vulkan

Desktop2Stereo的Python Vulkan迁移项目。项目保持原工程的`src/`目录和Python编程方式，默认图形主路径迁移为Vulkan，OpenGL仅作为隔离Fallback。

## 架构原则

- Capture、AI推理、立体合成调度、Vulkan、OpenXR、GUI和输出均使用Python源码实现。
- 保留已验证的WindowsCaptureCUDA、WindowsCaptureROCm、TensorRT、ROCm/MIGraphX和Triton优化代码。
- GPU实时路径不进行CPU NumPy/PIL逐帧往返。
- Filament DLL Bridge是唯一允许的项目自有C/C++组件。
- 不迁入Panda3D、D3D11 OpenXR、WGL/CUDA-GL Bridge和旧OpenGL上传器。

## 当前状态

仓库已完成第一批复制迁移和目标目录初始化。Capture、Inference、Stereo、GUI、资产及平台无关OpenXR代码已迁入；Vulkan Device、Compute Graph、OpenXR Vulkan Session和新版Filament Bridge仍按规格分阶段实现。

## 启动

```powershell
python src/main.py --probe
```

默认无参数启动只显示当前迁移状态，不会进入旧图形运行时。GUI源码已复制用于后续迁移，但在完成新配置Schema和Vulkan运行时装配前不从新入口启动。

## 规范

- [实时系统规格](docs/01-Realtime-2d-to-3d-specification.md)
- [工程设计规范](docs/02-desktop2stereo-engineering-design-specification.md)
- [Vulkan迁移技术报告](docs/03-d2s_vulkan_migration_technical_report.md)
- [迁移清单](docs/MIGRATION_MANIFEST.md)
