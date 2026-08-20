# 跨平台串流运行时打包

网络串流运行时目录统一为 `streaming/rtmp/`，包含：

- `ffmpeg/bin/ffmpeg`（Windows 为 `ffmpeg.exe`）
- `mediamtx/mediamtx`（Windows 为 `mediamtx.exe`）
- `mediamtx.yml`（项目实际使用的最终配置）
- `mediamtx/mediamtx.yml`（MediaMTX 官方模板，仅用于首次生成最终配置）
- `runtime-manifest.json`（平台与压缩包映射）

发布包按目标平台分别放入对应的 FFmpeg 与 MediaMTX 二进制，不要混用不同操作系统的文件。可以运行 `python scripts/download_streaming_runtime.py --system Linux` 或 `--system Darwin` 自动下载并安装当前架构的运行时；脚本只复制可执行文件和 `mediamtx.yml`，不会覆盖现有配置。FFmpeg 应包含所需的硬件编码器；程序启动时仍会按实际分辨率探测，失败后自动回退软件编码。

首次启动时，程序根据 `runtime-manifest.json` 检测当前系统和架构，只解压对应的 FFmpeg、MediaMTX 压缩包；官方模板只在根目录 `mediamtx.yml` 不存在时复制一次。以后升级或重新解压不会覆盖根目录中的项目配置。

## 官方下载地址

MediaMTX：

- 最新发布页：<https://github.com/bluenviron/mediamtx/releases/latest>
- Windows x64：`mediamtx_v*_windows_amd64.zip`
- Linux x64：`mediamtx_v*_linux_amd64.tar.gz`
- Linux ARM64：`mediamtx_v*_linux_arm64.tar.gz`
- macOS Intel：`mediamtx_v*_darwin_amd64.tar.gz`
- macOS Apple Silicon：`mediamtx_v*_darwin_arm64.tar.gz`

FFmpeg：

- Windows x64：<https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip>
- Linux x64 静态版：<https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz>
- Linux ARM64 静态版：<https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-arm64-static.tar.xz>
- macOS Intel / Apple Silicon：<https://evermeet.cx/ffmpeg/>

可通过环境变量覆盖安装目录：

- `D2S_STREAMING_RUNTIME_DIR`：运行时根目录
- `D2S_FFMPEG_PATH`：直接指定 FFmpeg
- `D2S_MEDIAMTX_PATH`：直接指定 MediaMTX
- `D2S_MEDIAMTX_CONFIG`：直接指定 MediaMTX 配置

平台音频输入：Windows 使用 `dshow`，Linux 使用 PulseAudio，macOS 使用 `avfoundation`。macOS 的音频设备值应为 FFmpeg 设备索引，例如 `2`。

`ffmpeg/rtmp.bat` 不是运行时依赖。它是旧的 Windows 手工命令脚本；当前 FFmpeg 参数由 `src/streaming/direct_sbs.py` 统一生成，跨平台运行不会调用该文件。升级或重新解压运行时不会影响实际推流命令。

MediaMTX 配置使用运行目录根部的 `mediamtx.yml`。项目自定义的端口通过 `MTX_*ADDRESS` 环境变量传递，HLS 安全上限 `hlsSegmentMaxSize: 256M` 等固定兼容项保留在最终配置中。升级 MediaMTX 时，应以新版本默认配置更新 `mediamtx/mediamtx.yml`，再将需要的项目兼容项合并到根部 `mediamtx.yml`；不要把完整默认配置复制到 `settings.yaml`。
