# Vulkan shaders

本目录是产品运行时唯一的 Vulkan Shader 目录，包含 GLSL 源码、运行时直接加载的 SPIR-V 二进制以及 `manifest.json`。

使用 `scripts/compile_shaders.ps1` 可在本目录原位重新编译 `.spv`。GLSL 与对应 SPIR-V 必须一起提交，确保复制或发布整个 `src/` 后不再依赖仓库根目录的资源。
