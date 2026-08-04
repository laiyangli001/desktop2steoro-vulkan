# Vulkan Native OVR 提交路径：待实现与验证方案

状态：待实现、待实机验证

## 1. 目标

在 VDXR 内增加独立的 Vulkan Native OVR 路径，同时完整保留现有 D3D11 OVR 路径：

```text
Vulkan OpenXR 应用 → Vulkan Native OVR SwapChain → ovr_EndFrame
D3D11 OpenXR 应用 → D3D11 OVR SwapChain       → ovr_EndFrame
```

本方案不替换 `ovr_CreateTextureSwapChainDX`，也不删除任何现有 D3D11 提交接口。Vulkan Native 路径只有在运行时能力验证通过后才启用，否则继续使用现有 Vulkan→D3D11 互操作路径。

## 2. 当前事实

VDXR 已经支持 Vulkan OpenXR 前端，但当前 Vulkan 路径仍然把资源交给 D3D11 OVR 后端：

- `virtualdesktop-openxr/vulkan_interop.cpp:421` 将 Vulkan 初始化描述为与 D3D11 backend 的互操作。
- `virtualdesktop-openxr/vulkan_interop.cpp:472` 创建 OVR 使用的 submission device。
- `virtualdesktop-openxr/vulkan_interop.cpp:479` 建立 Vulkan 与 D3D11 fence/timeline 同步。
- `virtualdesktop-openxr/vulkan_interop.cpp:835` 明确将 Vulkan queue 同步到“D3D11 context used by OVR”。
- `virtualdesktop-openxr/swapchain.cpp:384` 当前使用 `ovr_CreateTextureSwapChainDX`。
- `virtualdesktop-openxr/d3d11_native.cpp:793` 的慢路径仍通过 `ovr_CreateTextureSwapChainDX` 创建内部 OVR swapchain。

因此，当前“应用使用 Vulkan”不等于“VD 使用 Vulkan 原生 OVR 提交”。

## 3. 已发现的 Vulkan OVR API

VDXR 的 OVR shim 已包装以下 LibOVR Vulkan API：

```text
ovr_GetSessionPhysicalDeviceVk
ovr_SetSynchronizationQueueVk
ovr_CreateTextureSwapChainVk
ovr_GetTextureSwapChainBufferVk
ovr_CreateMirrorTextureWithOptionsVk
ovr_GetMirrorTextureBufferVk
```

对应代码位于 `virtualdesktop-openxr/OVR_CAPIShim.c:1304-1365`。shim 使用 `GetProcAddress` 从实际加载的 LibOVR DLL 动态解析函数，解析逻辑位于 `virtualdesktop-openxr/OVR_CAPIShim.c:620-637`。

这只能证明 LibOVR ABI 中存在 Vulkan API，不能单独证明当前 Virtual Desktop 运行时真正实现并支持这些调用。必须完成运行时探测和提交测试。

## 4. 如何确认 Virtual Desktop 提供原生 Vulkan 提交

### 4.1 符号探测

在 VDXR 创建 OVR session 后，对实际加载的 LibOVR 模块调用 `GetProcAddress`，记录以下符号是否存在：

```text
ovr_GetSessionPhysicalDeviceVk
ovr_SetSynchronizationQueueVk
ovr_CreateTextureSwapChainVk
ovr_GetTextureSwapChainBufferVk
ovr_CommitTextureSwapChain
```

符号存在是必要条件，不是充分条件。

### 4.2 设备和队列验证

使用当前 OpenXR Vulkan session 的对象进行验证：

1. `ovr_GetSessionPhysicalDeviceVk` 返回成功。
2. 返回的 `VkPhysicalDevice` 与 OpenXR 使用的物理设备一致。
3. `ovr_SetSynchronizationQueueVk` 接收当前 `VkQueue` 并返回成功。

任一步失败，都不能启用 Vulkan Native 路径。

### 4.3 SwapChain 验证

使用实际投影格式和尺寸创建最小 Vulkan OVR swapchain：

```text
ovr_CreateTextureSwapChainVk
ovr_GetTextureSwapChainLength
ovr_GetTextureSwapChainBufferVk
写入 Vulkan 图像
ovr_CommitTextureSwapChain
ovr_EndFrame
```

需要确认：

- 返回的 `VkImage` 有效；
- 图像格式、尺寸、array layer 与请求一致；
- Vulkan layout transition 和 queue ownership 不触发错误；
- `ovr_CommitTextureSwapChain` 返回成功；
- `ovr_EndFrame` 返回成功且头显有画面。

### 4.4 红绿双眼验证

使用两个独立的 `array_size=1` Vulkan swapchain：

```text
左眼写纯红
右眼写纯绿
提交同一个 ovr_EndFrame
```

预期结果：左眼红色、右眼绿色。该测试同时验证 Vulkan 原生 swapchain、左右眼 layer 绑定和 VD 最终合成，不受普通场景内容影响。

## 5. 推荐实现结构

### 5.1 保留现有 D3D11 路径

以下接口和逻辑必须保留：

- `ovr_CreateTextureSwapChainDX`
- `ovr_GetTextureSwapChainBufferDX`
- D3D11 slow-path swapchain
- D3D11 Resolve、复制、预处理和 fallback
- 当前 Vulkan→D3D11 互操作路径

Vulkan Native 能力探测失败、创建失败或提交失败时，自动回退到当前路径。

### 5.2 新增 Vulkan Native 路径

建议在 `Swapchain` 中增加独立的 Vulkan OVR 资源字段，不复用 D3D11 图像容器：

```text
ovrTextureSwapChain nativeVkOvrSwapchain
vector<VkImage> nativeVkImages
bool nativeVkActive
```

实现分工：

1. `vulkan_native.cpp`：初始化、能力探测、Vulkan OVR swapchain 创建和销毁。
2. `swapchain.cpp`：Vulkan session 根据能力选择 DX 或 VK 创建路径。
3. Vulkan swapchain image 枚举：直接返回 `ovr_GetTextureSwapChainBufferVk` 的 `VkImage`。
4. Vulkan release/resolve：执行必要的 layout transition，然后调用通用 `ovr_CommitTextureSwapChain`。
5. `frame.cpp`：继续使用现有 OVR layer 组装和 `ovr_EndFrame`，只替换 layer 使用的 swapchain 来源。

### 5.3 必须绕开的 D3D11 假设

Native Vulkan 路径不能直接复用以下 D3D11 假设：

- `ID3D11Texture2D` 图像容器；
- D3D11 SRV/RTV/UAV 创建；
- D3D11 Resolve 和复制 shader；
- D3D11 submission fence；
- 仅支持 `ovrTextureMisc_DX_Typeless` 和 `ovrTextureBind_DX_*` 的资源描述。

这些处理必须使用 Vulkan 版本，或在 Vulkan Native 路径中直接跳过不需要的预处理。

## 6. 能力选择规则

```text
Vulkan session
  ├─ Vulkan OVR 符号缺失       → 现有 Vulkan→D3D11
  ├─ 设备/队列验证失败         → 现有 Vulkan→D3D11
  ├─ Vulkan OVR swapchain 创建失败 → 现有 Vulkan→D3D11
  ├─ 红绿提交测试失败           → 现有 Vulkan→D3D11
  └─ 全部验证成功               → Vulkan Native OVR

D3D11 session                  → 保持现有 D3D11 OVR
```

默认不强制启用 Native Vulkan。建议先增加诊断开关，例如 `D2S_VDXR_VULKAN_NATIVE=1`，只在显式开启时实验；确认稳定后再考虑自动选择。

## 7. 验收标准

- D3D11 应用行为与修改前一致。
- Vulkan 应用在 Native 关闭时继续走现有路径。
- Native 开启且探测成功时，不调用 `ovr_CreateTextureSwapChainDX` 创建 Vulkan 投影资源。
- Vulkan 红绿双眼测试稳定显示左红右绿。
- 普通投影层、Quad Layer、深度层至少完成投影层验证。
- 运行一段时间无 `VK_ERROR_DEVICE_LOST`、access violation 或 `ovr_EndFrame` 失败。
- 能在日志中明确记录：符号探测、设备验证、swapchain 类型、回退原因和最终提交路径。

## 8. 当前结论

“只修改 VDXR 内部文件”在技术上可行，但前提是 Virtual Desktop 实际加载的 LibOVR 实现了 Vulkan OVR API。当前源码已经给出了 Vulkan API 包装入口，但还没有证明 VD 运行时支持原生 Vulkan OVR 提交。

下一步应先实现**只读能力探测 + 最小 Vulkan swapchain + 红绿双眼提交测试**，验证成功后再迁移完整 Projection、Quad 和深度路径。D3D11 接口不删除、不替换，始终作为兼容和故障回退路径。
