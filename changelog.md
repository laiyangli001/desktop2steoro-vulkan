# Desktop2Stereo Vulkan 项目日志

本文件记录项目重大更新和每日工作收尾。新记录按日期倒序追加；每个工作日结束时更新“已实现”“验证结果”“未决事项”和“下一项内容”。

## 2026-07-18

### 已实现

- 建立独立项目`desktop2steoro-vulkan`，保持原项目的Python源码组织方式，不在运行时依赖原仓库。
- 迁移可复用的Capture、AI推理、Stereo、GUI、OpenXR平台无关模块、Samples、测试和工具；原项目文件保持不变。
- 迁移`native/filament`及Windows、Linux、macOS多平台GitHub Actions构建流程，统一产物目录为`src/xr_viewer/native/`。
- 确立Vulkan为主图形路径、OpenGL为隔离Fallback，不迁入旧Panda3D、D3D11 OpenXR、WGL/CUDA-GL Bridge和旧OpenGL上传链路。
- 实现Python Vulkan基础层，包括Instance、物理设备选择、Device、Graphics Queue、Command Pool、Command Buffer、Fence、图像布局转换、清屏提交和资源释放。
- 实现基于`XR_KHR_vulkan_enable2`的Python OpenXR Vulkan Phase 1，包括运行时选定物理设备、Session、双眼交换链、Projection Layer、事件处理和纯色帧提交。
- 新增`src/tools/openxr_vulkan_smoke.py`，用于头显环境下独立验证双眼Vulkan交换链。
- 更新`src/requirements.txt`，明确`pyopenxr==1.1.5301`和`vulkan==1.3.275.1`为Vulkan/OpenXR主路径依赖，PyOpenGL归入Fallback依赖。
- 修正pyopenxr Composition Layer提交方式，使用`ctypes.pointer(layer)`满足`FrameEndInfo.layers`的Base Header指针约定。
- 更新能力探针、README和迁移清单，使Phase 1状态与头显实测结果一致。

### 验证结果

- Filament Vulkan Bridge的Windows、Linux和macOS GitHub Actions构建已通过（run `29650016647`）。
- 新增手动发布工作流，可从成功的三平台 CI 运行中下载 DLL、so、dylib，打包为 GitHub Release 资产并生成 SHA-256 校验文件。
- 更新 Filament Bridge CI：三平台构建完成后自动将 DLL、so、dylib 下载到`src/xr_viewer/native/`并提交到`main`，不再只保留为临时 Actions artifact。
- 本机Vulkan探针识别到NVIDIA GeForce RTX 3090、Vulkan 1.4.341和Graphics Queue Family 0。
- Virtual Desktop OpenXR Runtime可加载，并声明支持`XR_KHR_vulkan_enable2`。
- Vulkan/OpenXR新增及迁移状态测试共13项通过。
- 最终全量测试394项全部通过；期间既有Hugging Face Provider测试曾因外部站点SSL EOF短暂失败，网络恢复后复测通过。
- `py_compile`和`git diff --check`通过。
- 未连接头显时，Smoke入口按设计返回`FormFactorUnavailableError`并完成资源清理。
- 连接头显后成功创建双眼3648x3648 Vulkan交换链，并完成300/300帧提交。
- 用户确认头显内稳定显示深蓝色双眼画面，无OpenXR调用顺序、Vulkan同步或资源释放错误。
- 开始实现Filament Vulkan Render Target Bridge：新增VulkanSharedContext接入、OpenXR VkImage外部SwapChain、Python ctypes封装和跨平台构建配置。
- Bridge明确借用Python/OpenXR所有Vulkan对象，不创建或销毁OpenXR资源；结束帧前使用Filament `flushAndWait`完成GPU同步。
- 将Bridge以显式配置方式接入`OpenXrVulkanPresenter`：左右眼分别绑定外部OpenXR VkImage，帧内传递acquire index，关闭顺序先Bridge后OpenXR交换链；未配置Bridge时保持原有Vulkan清屏路径。
- 在Filament Bridge内建立`Scene`、`Camera`和`View`，加载GLB后将实体加入场景，并在每帧调用`Renderer::render`；三平台CI重新编译通过，最新二进制已自动回写`src/xr_viewer/native/`。
- 增加每眼OpenXR Camera同步：Python根据View pose计算look-at参数，根据View FOV计算垂直视场角和aspect，并通过C ABI更新Filament Camera；三平台新Bridge构建和19项聚焦测试通过。
- 扩展`openxr_vulkan_smoke.py`支持显式指定`--filament-bridge`和`--filament-glb`，默认仍保持纯Vulkan清屏模式；README补充Bedroom环境GLB的Filament头显测试命令。
- 修复Filament 1.74外部SwapChain的`FixedCapacityVector`容量初始化，并保存平台层ExternalSwapChain句柄；Windows RTX 3090头显实测无GLB 60/60帧、QUEST控制器GLB 120/120帧、Artemis `environment3.glb` 120/120帧通过。

### 未决事项

- CodeGraph数据库被当前MCP进程占用，本轮无法重建索引；代码和测试不受影响。
- 既有Hugging Face Provider测试依赖外部站点可达性，需要后续消除测试对网络状态的依赖。
- Filament Bridge的真实场景渲染尚未验证；当前Python封装只覆盖Bridge ABI和生命周期，不接管OpenXR acquire/release。
- Artemis和QUEST GLB已完成头显帧提交实测，但尚未取得用户对头显内实际模型画面的视觉确认；FOV同步使用对称等效投影，OpenXR非对称左右/上下切偏移仍需使用自定义投影矩阵精确处理。
- Bedroom `environment.glb` 在Filament `load_glb`阶段解析失败，文件头和GLB声明长度一致，需后续用glTF Validator定位其扩展或资源兼容性问题。

### 下一项内容

下一项：请确认头显内是否看到了Artemis场景和QUEST模型；随后修复Bedroom GLB兼容性，并将非对称OpenXR投影矩阵接入Filament Camera。
