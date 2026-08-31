# 跨平台原生启动器实现计划

## 1. 目标

为 Desktop2Stereo 增加可直接运行的跨平台原生启动器。用户运行启动器即可完成启动画面、授权检查、运行时定位和 Python/Flet GUI 启动，不再依赖手动执行 `.bat` 文件。

本计划与 `13-cross-platform-licensing-launcher-implementation-plan.md` 对接，但不假定其中尚未实现的授权功能已经存在。

## 2. 总体启动流程

```text
原生启动器
  → 显示无边框透明 Splash
  → 初始化许可证检查接口
  → 定位项目目录、运行时和 Python
  → 启动 desktop2stereo 主程序
  → 等待 GUI 完整初始化信号
  → 关闭 Splash
```

主程序异常退出、许可证失败或运行时缺失时，启动器必须关闭 Splash，并显示明确的原生错误信息。

## 3. 平台实现

### 3.1 Windows

- 使用独立 Win32 原生启动器（`.exe`）。
- 使用 `WS_EX_LAYERED` 和 `UpdateLayeredWindow` 显示逐像素 Alpha PNG。
- 启动器负责解析自身所在目录，支持路径包含空格和非 ASCII 字符。
- 支持 Windows 10/11，处理主显示器定位和多显示器尺寸。
- `.bat` 保留为开发、诊断和兼容入口，不再作为用户必需入口。

### 3.2 Linux

- 提供独立可执行启动器。
- 首期兼容常见 X11 环境。
- 为 Wayland 保留平台接口；透明窗口能力由当前合成器决定，无法确认时标记为待验证。
- 可选使用 GTK 或 Qt 原生窗口，平台代码不得依赖 Python。

### 3.3 macOS

- 提供 `.app` 或独立可执行启动器。
- 使用 AppKit `NSWindow` 创建无边框窗口、透明背景和图片视图。
- 分别支持 Intel Mac、Apple Silicon，并提供 Universal 构建方案。
- 不将 Vulkan/MoltenVK 作为启动画面实现依赖。

## 4. Splash 设计

- 使用 `src/desktop2stereo/d2s_blur.png`。
- 启动器进程显示图片，不依赖 Python、Flet、PySide、PyQt 或 tkinter。
- 无边框、透明背景、居中显示。
- 按主显示器分辨率计算窗口大小，目标面积约为显示器面积的 1/4。
- 保持图片原始宽高比。
- 图片缺失时显示可读的纯色回退或错误提示。
- Splash 不接受业务输入；主程序准备完成后立即关闭。

## 5. 主程序衔接

- 统一启动 `desktop2stereo` 主入口，并正确传递项目根目录、配置目录、日志目录和命令行参数。
- 支持当前 GUI/GUI2 启动菜单选择及最后选择记忆。
- 使用 `gui_ready.flag`、退出码或本地 IPC 作为握手机制；不得仅以 Splash 已显示作为 GUI 就绪条件。
- GUI 完整挂载、必要的初始配置加载和启动准备完成后，主程序写入就绪信号。
- 启动器必须监控子进程；子进程异常退出时关闭 Splash 并报告错误。
- 启动器退出后不得残留窗口、锁文件或后台进程。

## 6. 授权扩展接口

启动器需要预留与 `13-cross-platform-licensing-launcher-implementation-plan.md` 的稳定接口：

- 启动阶段提供许可证检查入口。
- 区分检查中、授权有效、授权无效、离线宽限和检查失败状态。
- 授权失败显示原生错误窗口，不静默跳过或伪装为成功。
- 为在线激活、设备绑定、离线授权和版本限制预留扩展点。
- 授权模块与 Splash、进程管理、GUI 解耦。
- 禁止硬编码密钥、Token 或其他私有凭据。
- 尚未实现的授权行为必须标记为 TODO/待实现。

## 7. 工程目录建议

```text
native/
  launcher/
    common/
    windows/
    linux/
    macos/
  splash/
    assets/
```

公共协议、版本信息、资源定位和错误码放在 `common`；窗口创建、进程启动和平台授权桥接放在对应平台目录。

## 8. 构建与发布

- 提供 Windows、Linux、macOS 的 Debug/Release 构建脚本或明确构建说明。
- 启动器发布包必须包含所需原生动态库和资源，不依赖开发机上的编译器、SDK 或 Python。
- Windows 明确 MSVC 或 MinGW 构建链；Linux 明确 GTK/Qt 版本和 X11/Wayland 依赖；macOS 明确 Intel/Apple Silicon 架构。
- 发布目录中提供直接可运行的启动器，并保留 Python/Flet 运行时和项目资源的相对路径约定。
- 启动器应支持从任意当前工作目录启动。

## 9. 实施阶段

1. 梳理当前 Windows `.bat`、统一主入口和 `gui_ready.flag` 生命周期。
2. 定义跨平台启动协议、错误码、授权接口和资源定位规则。
3. 实现 Windows 原生 Splash 与子进程监控。
4. 实现 Linux 原生 Splash，先验证 X11，再保留 Wayland 适配接口。
5. 实现 macOS AppKit Splash 与双架构构建。
6. 接入 GUI/GUI2 选择、授权接口和异常关闭流程。
7. 生成各平台发布目录和用户可直接运行的启动器。

## 10. 验证标准

- 直接运行各平台启动器即可启动 GUI，无需手动运行 `.bat`。
- Python 解释器加载前已经显示 Splash。
- 透明 PNG 区域不会被固定黑色背景覆盖。
- GUI 完成初始化后 Splash 正常关闭。
- 主程序异常退出、路径错误、运行时缺失和授权失败均能正确清理 Splash。
- GUI 与 GUI2 均可启动，且启动菜单选择保持一致。
- 多显示器环境下 Splash 显示在主显示器并正确计算尺寸。
- 路径包含空格或非 ASCII 字符时仍可启动。
- 完成 Python `py_compile`、原生启动器构建测试、相关单元测试、Markdown 检查和 `git diff --check`。

## 11. 当前边界

- 本计划只定义启动器和启动画面实现，不改变立体推理、捕获、合成或推流业务逻辑。
- Linux Wayland 透明窗口、macOS Universal 打包和许可证在线服务在实现前必须进行实际环境验证。
- Qt 可以作为原生跨平台实现方案，但不得以 Qt Python 绑定作为启动前置依赖；如采用 Qt，应随启动器发布所需原生运行库。
