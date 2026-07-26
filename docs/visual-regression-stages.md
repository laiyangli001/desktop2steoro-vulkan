# 立体显示阶段视觉回归

该工具用于定位“文字边缘变模糊”首次出现的阶段。它使用固定的 RGB/Depth 输入，生成捕捉、深度、Vulkan storage-image 输出以及 CUDA/Triton 参考图，并生成差异热图。

## 运行

使用项目自带 Python：

```powershell
src\python3\python.exe scripts\visual_regression_stages.py `
  --rgb samples\gui.png `
  --depth <与当前实机相同的深度图> `
  --output-dir $env:TEMP\d2s-visual-regression
```

如果没有深度图，工具会生成测试用 proxy depth；这只能检查采样和输出路径，不能代表实机深度效果。

CUDA/Triton 可用时默认会运行 CUDA 参考路径；只检查 Vulkan 时加 `--skip-cuda`。如果已有 CUDA 阶段截图，可以使用：

```powershell
src\python3\python.exe scripts\visual_regression_stages.py `
  --rgb <rgb.png> `
  --depth <depth.png> `
  --cuda-reference-dir <目录> `
  --output-dir <输出目录>
```

`--cuda-reference-dir` 中放置 `left_eye.png` 和 `right_eye.png`。

## 输出

- `00_capture_rgb.png`：捕捉输入。
- `01_raw_depth.png`、`02_prepared_depth.png`：深度处理前后。
- `03_vulkan_left_eye.png`、`03_vulkan_right_eye.png`：生产同款 `d2s_stereo_layered_output` storage-image shader 的读回结果。
- `03_vulkan_output_left_eye.png`、`03_vulkan_output_right_eye.png`：OpenXR 实机首帧中，进入 Filament 前的生产 Vulkan output image。
- `06_openxr_projection_left_eye.png`、`06_openxr_projection_right_eye.png`：OpenXR projection swapchain 的最终渲染结果。
- `04_cuda_left_eye.png`、`04_cuda_right_eye.png`：CUDA/Triton 参考结果。
- `05_diff_vulkan_vs_cuda_*_heatmap.png`：左右眼差异热图。
- `visual_regression_contact_sheet.png`：所有阶段缩略图总览。
- `visual_regression_manifest.json`：尺寸、后端、颜色契约、读回路径和误差指标。

必须确认 manifest 中包含：

```text
visual_regression_shader=d2s_stereo_layered_output
visual_regression_readback=temporary_host_image
```

如果出现 `visual_regression_exact_output_error`，说明设备无法执行独立 storage-image 诊断，工具使用的是 buffer shader 回退结果，不能据此判断 Filament 导入阶段。

手动工具的临时 host image 读回只用于诊断，不会改变正式运行时的零拷贝路径。OpenXR
自动截图只在首个可提交帧完成后执行一次，后续帧不再读回。

## OpenXR 实机自动截图

默认运行 OpenXR 时不会抓取运行截图，也不会执行首帧 Vulkan 读回。需要阶段证据时，
显式设置 `D2S_OPENXR_RGB_DEPTH_DUMP_DIR` 后再启动程序，运行时才会对首个可提交帧
抓取一次。输出目录就是该变量指定的目录：

`.ci-artifacts/visual-regression/auto-<timestamp>-<pid>-<nonce>/`

除输入和深度阶段外，还会从同一帧保存：

- `03_vulkan_output_left_eye.png`、`03_vulkan_output_right_eye.png`：生产 Vulkan
  storage image 在进入 Filament 前的内容。
- `06_openxr_projection_left_eye.png`、`06_openxr_projection_right_eye.png`：Filament
  完成渲染后的 OpenXR projection swapchain 内容。
- `visual_regression_runtime_manifest.json`：帧号、尺寸、颜色空间和图像原点。
- `visual_regression_contact_sheet.png`：自动生成的阶段缩略图对比。

截图只执行一次，并且发生在正常渲染完成后；显式启用时，读回会增加首帧诊断开销。
也可以直接运行本目录中的阶段回归脚本生成离线证据。
