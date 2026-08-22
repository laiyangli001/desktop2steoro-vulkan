# 在 GitHub 建立统一 FFmpeg 构建与发布项目

本文说明如何建立独立的 GitHub 仓库，完全使用 GitHub Actions 的远程 Runner 为 Desktop2Stereo 构建、验证和发布 FFmpeg。本地电脑不安装编译器、MSYS2、Vulkan SDK 或各编码库的开发环境，只负责提交脚本、触发远程构建，以及下载构建结果。目标不是生成一个能在所有系统运行的二进制文件，而是使用同一份版本清单、构建脚本、验证规则和 GitHub Release，稳定产出各平台对应的软件包。

> FFmpeg 官方只发布源码，官网列出的 Windows、Linux 和 macOS 可执行文件来自不同第三方。因此，项目自己构建是统一版本和功能集最可靠的方式。参考：[FFmpeg 下载页](https://ffmpeg.org/download.html)。

## 目录

- [目标与边界](#目标与边界)
- [建议的发布矩阵](#建议的发布矩阵)
- [创建 GitHub 仓库](#创建-github-仓库)
- [固定源码和依赖版本](#固定源码和依赖版本)
- [配置编译功能](#配置编译功能)
- [实现构建脚本](#实现构建脚本)
- [实现 GitHub Actions](#实现-github-actions)
- [远程触发构建并下载到本地](#远程触发构建并下载到本地)
- [验证构建结果](#验证构建结果)
- [发布产物和运行时清单](#发布产物和运行时清单)
- [接入 Desktop2Stereo](#接入-desktop2stereo)
- [许可证与再分发要求](#许可证与再分发要求)
- [常见问题](#常见问题)
- [验收清单](#验收清单)

## 目标与边界

建议新建仓库：

```text
desktop2stereo-ffmpeg-builds
```

它应提供以下能力：

1. 所有平台使用同一个 FFmpeg 版本和同一组核心功能。
2. Windows、Linux、macOS 的二进制都从同一个 GitHub Release 下载。
3. 每个依赖使用固定版本，不在正式发布时动态获取 `latest`。
4. 每个压缩包带 SHA-256、构建配置、许可证和版本信息。
5. CI 验证“功能是否编译进去”；真实 GPU 主机验证“硬件编码是否能运行”。
6. 自动生成供 Desktop2Stereo 使用的 `runtime-manifest.json`。
7. 本地电脑不承担编译，只下载 GitHub 生成并校验过的压缩包。

下面两点必须区分：

- `--enable-vulkan` 表示 FFmpeg 编译了 Vulkan 支持，不等于任意显卡都能执行 Vulkan Video 编码。
- GitHub 托管 Runner 通常没有可用的 NVIDIA、AMD 或 Intel 编码 GPU，因此只能完成编译和静态能力检查，不能替代真实硬件测试。
- 本地仍需要执行一次目标显卡真机测试，但只运行下载好的 FFmpeg，不需要安装任何构建环境。

## 建议的发布矩阵

| 平台键 | 构建环境 | 主要硬件编码能力 | Vulkan Video | 压缩格式 |
| --- | --- | --- | --- | --- |
| `Windows-amd64` | Windows Server 2022/MSYS2 | NVENC、AMF、QSV/oneVPL | 编译启用，真机验证 | ZIP |
| `Linux-amd64` | Ubuntu 24.04 | NVENC、AMF、QSV/oneVPL | 编译启用，真机验证 | TAR.XZ |
| `Linux-arm64` | 原生 ARM64 Runner 或交叉编译容器 | 依设备而定 | 编译启用，设备验证 | TAR.XZ |
| `Darwin-amd64` | macOS 13 | VideoToolbox | 不作为强制要求 | ZIP |
| `Darwin-arm64` | macOS 14/15 | VideoToolbox | 不作为强制要求 | ZIP |

macOS 不应把 Vulkan Video 当作验收条件。MoltenVK 不能被当成完整的 Vulkan Video 编码后端，macOS 应优先使用系统 VideoToolbox。

MediaMTX 不需要跟随 FFmpeg 一起编译。继续从 MediaMTX 官方 Release 获取，并在 Desktop2Stereo 的运行时清单中独立管理版本和校验值。

## 创建 GitHub 仓库

### 1. 建立空仓库

在 GitHub 创建 `desktop2stereo-ffmpeg-builds`，建议先设为私有仓库完成验证，确认许可证和发布内容无误后再决定是否公开。

仓库可以直接通过 GitHub 网页建立。需要编辑和提交脚本时，只要安装 Git，不需要安装任何 FFmpeg 编译依赖：

```powershell
git clone https://github.com/OWNER/desktop2stereo-ffmpeg-builds.git
Set-Location desktop2stereo-ffmpeg-builds
```

也可以完全使用 GitHub 网页上传初始文件。无论使用哪种方式，Windows、Linux 和 macOS 的编译都只在 GitHub Actions 中执行。

### 2. 建立目录结构

```text
desktop2stereo-ffmpeg-builds/
├─ .github/
│  └─ workflows/
│     ├─ build.yml
│     └─ release.yml
├─ config/
│  ├─ versions.env
│  └─ targets.json
├─ scripts/
│  ├─ build-common.sh
│  ├─ build-windows.sh
│  ├─ build-linux.sh
│  ├─ build-macos.sh
│  ├─ package.sh
│  ├─ verify-runtime.py
│  └─ generate-manifest.py
├─ patches/
├─ licenses/
├─ README.md
└─ LICENSE
```

仓库只保存脚本、补丁和版本配置，不要提交已编译的大文件。二进制文件由 Actions 上传到 Artifacts 或 GitHub Releases。

## 固定源码和依赖版本

`config/versions.env` 是唯一版本来源。示例：

```bash
FFMPEG_VERSION=9.0.1
FFMPEG_TAG=n9.0.1

X264_REF=<固定提交或正式标签>
X265_REF=<固定标签>
OPUS_VERSION=<固定版本>
SRT_VERSION=<固定版本>
NV_CODEC_HEADERS_REF=<固定标签>
AMF_HEADERS_REF=<固定标签>
ONEVPL_VERSION=<固定版本>
VULKAN_HEADERS_VERSION=<固定版本>
VULKAN_LOADER_VERSION=<固定版本>

BUILD_REVISION=1
```

实施要求：

- FFmpeg 正式包优先使用带签名的发布源码，并验证官方签名或 SHA-256。
- Git 仓库依赖必须固定到标签或完整 commit SHA。
- 下载的源码归档必须保存 SHA-256；校验失败立即停止构建。
- 升级任何依赖后，将 `BUILD_REVISION` 加一。
- 不允许正式 Release 从浮动的 `master`、`main` 或 `latest` 构建。

如果必须使用 FFmpeg 开发分支，应固定完整提交，并将提交号写入包名和 `build-info.json`。

## 配置编译功能

用户列出的参数可以出现在同一个 FFmpeg 配置中，但不同操作系统需要使用不同的参数子集。

### 公共功能

Windows 和 Linux 的核心配置建议为：

```bash
COMMON_CONFIG=(
  --enable-gpl
  --enable-libx264
  --enable-libx265
  --enable-libopus
  --enable-libsrt
  --enable-vulkan
)
```

硬件后端配置：

```bash
HW_CONFIG=(
  --enable-nvenc
  --enable-amf
  --enable-libvpl
)
```

完整调用示意：

```bash
./configure \
  --prefix="$PREFIX" \
  --pkg-config-flags=--static \
  --extra-cflags="-I$PREFIX/include" \
  --extra-ldflags="-L$PREFIX/lib" \
  "${COMMON_CONFIG[@]}" \
  "${HW_CONFIG[@]}"
```

构建参数中删除 `--enable-nonfree`，公开构建和 Release 均禁止启用。它不是“增加更多免费编码器”的普通开关；启用后，FFmpeg 会把组合产物标记为不可再分发。

建议在 CI 增加硬性检查，避免后续维护时误加回来：

```bash
if grep -R --fixed-strings -- '--enable-nonfree' scripts config; then
  echo 'ERROR: public FFmpeg builds must not enable nonfree components' >&2
  exit 1
fi
```

### macOS 功能

macOS 使用系统框架，建议配置为：

```bash
./configure \
  --prefix="$PREFIX" \
  --enable-gpl \
  --enable-libx264 \
  --enable-libx265 \
  --enable-libopus \
  --enable-libsrt \
  --enable-videotoolbox \
  --enable-audiotoolbox
```

macOS 构建不强制加入 `--enable-nvenc`、`--enable-amf`、`--enable-libvpl` 或 Vulkan Video 编码器。

### 编码后端和用途

| 后端 | 目标系统 | 预期编码器示例 | 说明 |
| --- | --- | --- | --- |
| Vulkan Video | Windows、Linux | `h264_vulkan`、`hevc_vulkan` | 需要 FFmpeg、Vulkan Loader、驱动和设备同时支持 |
| NVENC | Windows、Linux/NVIDIA | `h264_nvenc`、`hevc_nvenc` | 编译依赖 `nv-codec-headers`，运行依赖 NVIDIA 驱动 |
| AMF | Windows，部分 Linux/AMD | `h264_amf`、`hevc_amf` | 编译需要 AMF Headers，运行依赖 AMD 驱动 |
| oneVPL/QSV | Windows、Linux/Intel | `h264_qsv`、`hevc_qsv` | 编译依赖 oneVPL，运行依赖 Intel 驱动 |
| VideoToolbox | macOS | `h264_videotoolbox`、`hevc_videotoolbox` | Apple 平台首选 |
| libx264/libx265 | 全平台 | `libx264`、`libx265` | CPU 回退路径；使 FFmpeg 构建适用 GPL |

## 实现构建脚本

### 1. 公共脚本职责

`scripts/build-common.sh` 应完成：

1. 读取 `config/versions.env`。
2. 下载源码并验证签名或 SHA-256。
3. 按固定顺序构建 x264、x265、Opus、SRT 和平台硬件头文件。
4. 生成 FFmpeg `configure` 参数数组。
5. 保存 `ffbuild/config.log`、`ffmpeg -buildconf` 和依赖版本。
6. 安装到独立的 `$PREFIX`，不得依赖 Runner 上碰巧存在的库。

所有脚本应开启严格错误处理：

```bash
#!/usr/bin/env bash
set -euo pipefail
```

### 2. Windows 构建

远程使用 GitHub 的 `windows-2022` Runner 和 MSYS2/MinGW64。Actions 工作流负责临时安装环境，Bash 脚本完成依赖和 FFmpeg 编译；任务结束后 Runner 被释放，不会改变本地电脑。

关键依赖包括：

- `mingw-w64-x86_64-gcc`
- `mingw-w64-x86_64-nasm`
- `mingw-w64-x86_64-yasm`
- `mingw-w64-x86_64-cmake`
- `mingw-w64-x86_64-meson`
- `mingw-w64-x86_64-ninja`
- `mingw-w64-x86_64-pkgconf`
- Vulkan Headers/Loader
- `nv-codec-headers`、AMF Headers、oneVPL

不要只安装头文件就宣称硬件编码可用。头文件解决编译问题，驱动和 GPU 决定运行问题。

### 3. Linux 构建

在 GitHub 的 Ubuntu Runner 内使用固定容器构建，避免 Runner 镜像更新改变依赖：

```yaml
container:
  image: ubuntu:24.04
```

Linux amd64 可以先完成。Linux arm64 有两种方案：

1. 使用 GitHub 可用的原生 ARM64 Runner或自托管 ARM64 Runner；
2. 使用交叉编译工具链，并在真实 ARM64 设备上执行最终验证。

不要仅凭 x86 主机“编译成功”判定 ARM64 包可发布。

### 4. macOS 构建

分别在 GitHub 的 Intel 和 Apple Silicon Runner 上构建。不要用一个通用包覆盖两个架构，除非后续明确增加 `lipo` 合并和所有依赖的 Universal Binary 构建流程。

macOS 包至少验证：

```bash
ffmpeg -hide_banner -encoders | grep videotoolbox
ffmpeg -hide_banner -protocols | grep srt
```

### 5. 产物目录

每个平台安装完成后整理为相同结构：

```text
package-root/
├─ ffmpeg/
│  └─ bin/
│     ├─ ffmpeg[.exe]
│     └─ ffprobe[.exe]
├─ licenses/
├─ build-info.json
├─ configure.txt
├─ dependency-versions.txt
└─ SOURCE-OFFER.md
```

`build-info.json` 至少记录：

```json
{
  "ffmpeg_version": "9.0.1",
  "ffmpeg_ref": "n9.0.1",
  "build_revision": 1,
  "platform": "Windows-amd64",
  "compiler": "gcc",
  "license_profile": "GPL",
  "nonfree": false
}
```

## 实现 GitHub Actions

下面是构建工作流骨架。实际依赖安装放进各平台脚本，避免 YAML 膨胀。

```yaml
name: Build FFmpeg

on:
  workflow_dispatch:
  pull_request:
  push:
    branches: [main]

permissions:
  contents: read

jobs:
  build:
    strategy:
      fail-fast: false
      matrix:
        include:
          - target: Windows-amd64
            os: windows-2022
            script: scripts/build-windows.sh
          - target: Linux-amd64
            os: ubuntu-24.04
            script: scripts/build-linux.sh
          - target: Darwin-amd64
            os: macos-13
            script: scripts/build-macos.sh
          - target: Darwin-arm64
            os: macos-14
            script: scripts/build-macos.sh

    runs-on: ${{ matrix.os }}

    steps:
      - uses: actions/checkout@v4

      - name: Build on Windows
        if: runner.os == 'Windows'
        shell: msys2 {0}
        run: ${{ matrix.script }} "${{ matrix.target }}"

      - name: Build on Unix
        if: runner.os != 'Windows'
        shell: bash
        run: ${{ matrix.script }} "${{ matrix.target }}"

      - name: Verify package
        shell: bash
        run: python3 scripts/verify-runtime.py "dist/${{ matrix.target }}"

      - uses: actions/upload-artifact@v4
        with:
          name: ffmpeg-${{ matrix.target }}
          path: dist/*
          if-no-files-found: error
```

注意：Windows 的 `shell: msys2 {0}` 需要先增加官方 MSYS2 setup Action。正式仓库应将第三方 Action 固定到完整 commit SHA，降低供应链变更风险。

发布工作流建议只接受版本标签：

```text
ffmpeg-v9.0.1-d2s.1
```

`release.yml` 的职责：

1. 由标签触发或手动触发。
2. 下载所有已验证的构建 Artifact。
3. 生成 `sha256sums.txt`。
4. 生成 `runtime-manifest.json`。
5. 同时打包对应源码、补丁、配置参数和依赖源码说明。
6. 创建 GitHub Release 并上传全部文件。

## 远程触发构建并下载到本地

本节是日常实际操作流程。本地不执行 `configure`、`make`、CMake、Meson 或 Ninja。

### 1. 提交构建配置

修改版本或脚本后，只提交到 GitHub：

```powershell
git add config scripts .github
git commit -m "build: update FFmpeg runtime"
git push origin main
```

推送到 `main` 会自动启动 `build.yml`。也可以打开 GitHub 仓库的 **Actions → Build FFmpeg → Run workflow**，手动选择分支并运行。

### 2. 查看远程构建

在 Actions 页面确认矩阵中的各平台任务状态：

```text
Windows-amd64  success
Linux-amd64    success
Darwin-amd64   success
Darwin-arm64   success
Linux-arm64    success 或明确标记暂不支持
```

任一目标失败时，在 GitHub 网页查看该任务日志。修复构建脚本后重新推送，不要转为在本地补装工具链编译。

### 3. 下载临时 Artifact

测试构建完成后，在对应 Actions Run 页面底部的 **Artifacts** 区域下载目标平台压缩包。Windows 开发机通常只需先下载：

```text
ffmpeg-Windows-amd64
```

如果本地安装了 GitHub CLI，也可以使用命令下载；GitHub CLI 只是仓库客户端，不是编译环境：

```powershell
gh run list --repo OWNER/desktop2stereo-ffmpeg-builds --workflow build.yml --limit 5
gh run download RUN_ID `
  --repo OWNER/desktop2stereo-ffmpeg-builds `
  --name ffmpeg-Windows-amd64 `
  --dir .\downloads\ffmpeg-Windows-amd64
```

Artifact 适合测试，不作为 Desktop2Stereo 的长期固定下载地址。

### 4. 发布正式 Release

所有平台的静态验证通过，并且目标 GPU 真机测试通过后，推送发布标签：

```powershell
git tag ffmpeg-v9.0.1-d2s.1
git push origin ffmpeg-v9.0.1-d2s.1
```

`release.yml` 在 GitHub 远程汇总各平台产物、生成 SHA-256 和运行时清单，再上传到 GitHub Release。正式包可通过 GitHub 网页下载，也可执行：

```powershell
gh release download ffmpeg-v9.0.1-d2s.1 `
  --repo OWNER/desktop2stereo-ffmpeg-builds `
  --pattern "*windows-amd64.zip" `
  --pattern "runtime-manifest.json" `
  --pattern "sha256sums.txt" `
  --dir .\downloads\ffmpeg-v9.0.1-d2s.1
```

### 5. 下载后在本地做运行验证

本地只解压和运行已构建的文件。以 Windows 为例：

```powershell
Expand-Archive `
  -LiteralPath .\downloads\ffmpeg-v9.0.1-d2s.1\d2s-ffmpeg-9.0.1-d2s.1-windows-amd64.zip `
  -DestinationPath .\downloads\ffmpeg-test

.\downloads\ffmpeg-test\ffmpeg\bin\ffmpeg.exe -version
.\downloads\ffmpeg-test\ffmpeg\bin\ffmpeg.exe -buildconf
.\downloads\ffmpeg-test\ffmpeg\bin\ffmpeg.exe -encoders
```

随后在本机显卡上执行本文的 GPU 真机测试。这里依赖现有显卡驱动，但不需要 Visual Studio、MSYS2、Vulkan SDK 或编码器头文件。

### 6. 自动下载到 Desktop2Stereo

最终不应要求用户手动复制 FFmpeg。Desktop2Stereo 的 `download_runtime.py` 读取 Release 中的 `runtime-manifest.json`，按当前系统下载对应文件、校验 SHA-256、解压验证，再安装到运行时目录。这样发布新版本后，本地只执行下载和安装流程。

## 验证构建结果

### 1. CI 静态验证

`scripts/verify-runtime.py` 应运行并解析：

```text
ffmpeg -version
ffmpeg -buildconf
ffmpeg -encoders
ffmpeg -hwaccels
ffmpeg -protocols
ffmpeg -filters
```

通用强制能力：

- `libx264`
- `libx265`
- `libopus`
- `srt`

Windows/Linux 强制检查编译项：

- `vulkan` 硬件设备或相关过滤支持
- `h264_vulkan`、`hevc_vulkan`，若选定 FFmpeg 版本提供这些编码器
- `h264_nvenc`、`hevc_nvenc`
- `h264_amf`、`hevc_amf`
- `h264_qsv`、`hevc_qsv`

macOS 强制检查：

- `h264_videotoolbox`
- `hevc_videotoolbox`

如果某个目标因上游或平台限制无法提供某项能力，应在 `targets.json` 中明确标记为可选，不能让验证脚本静默跳过。

### 2. CPU 冒烟测试

CI 可以执行不依赖真实 GPU 的短视频测试：

```bash
ffmpeg -hide_banner -y \
  -f lavfi -i testsrc2=size=1280x720:rate=30 \
  -f lavfi -i sine=frequency=1000:sample_rate=48000 \
  -t 3 -c:v libx264 -preset ultrafast -c:a libopus \
  smoke-test.mkv

ffprobe -v error -show_streams smoke-test.mkv
```

### 3. GPU 真机验证

硬件测试应在 NVIDIA、AMD、Intel 主机上分别执行，并保存日志。Vulkan Video 示例：

```bash
ffmpeg -hide_banner \
  -init_hw_device vulkan=vk:0 \
  -filter_hw_device vk \
  -f lavfi -i testsrc2=size=1920x1080:rate=60 \
  -vf format=nv12,hwupload \
  -t 10 -c:v h264_vulkan -b:v 20M vulkan-test.mp4
```

该命令失败不一定表示编译错误，还可能是驱动没有 Vulkan Video Encode 扩展、设备队列不支持编码或像素格式不被后端接受。真机报告必须同时保存：

```text
ffmpeg -version
ffmpeg -buildconf
ffmpeg -encoders
ffmpeg -init_hw_device list
vulkaninfo --summary
完整编码日志
```

Desktop2Stereo 当前优先需要 Windows/NVIDIA 路径，因此第一阶段验收顺序建议为：

1. Windows amd64 + NVENC。
2. Windows amd64 + Vulkan Video。
3. Linux amd64 + NVENC/Vulkan Video。
4. Windows AMD/Intel 和 macOS。
5. Linux arm64。

## 发布产物和运行时清单

推荐包名：

```text
d2s-ffmpeg-9.0.1-d2s.1-windows-amd64.zip
d2s-ffmpeg-9.0.1-d2s.1-linux-amd64.tar.xz
d2s-ffmpeg-9.0.1-d2s.1-linux-arm64.tar.xz
d2s-ffmpeg-9.0.1-d2s.1-darwin-amd64.zip
d2s-ffmpeg-9.0.1-d2s.1-darwin-arm64.zip
ffmpeg-9.0.1-d2s.1-sources.tar.xz
runtime-manifest.json
sha256sums.txt
```

建议将 Desktop2Stereo 的运行时清单升级为 schema 2，示例：

```json
{
  "schema_version": 2,
  "ffmpeg": {
    "version": "9.0.1",
    "build_revision": 1,
    "release_tag": "ffmpeg-v9.0.1-d2s.1"
  },
  "runtimes": {
    "Windows-amd64": {
      "ffmpeg_url": "https://github.com/OWNER/desktop2stereo-ffmpeg-builds/releases/download/ffmpeg-v9.0.1-d2s.1/d2s-ffmpeg-9.0.1-d2s.1-windows-amd64.zip",
      "ffmpeg_sha256": "<64位SHA-256>",
      "ffmpeg_executable": "ffmpeg/bin/ffmpeg.exe",
      "required_features": [
        "libx264",
        "libx265",
        "libopus",
        "srt",
        "h264_nvenc"
      ],
      "optional_features": [
        "h264_vulkan",
        "hevc_vulkan",
        "h264_amf",
        "h264_qsv"
      ]
    }
  }
}
```

`OWNER` 替换为实际 GitHub 组织或用户名。清单中的校验值必须由 Release 工作流根据最终上传文件计算，不能手工填写。

## 接入 Desktop2Stereo

当前项目的接入点是：

```text
src/desktop2stereo/streaming/rtmp/runtime-manifest.json
src/desktop2stereo/streaming/rtmp/download_runtime.py
```

迁移步骤：

1. 新 FFmpeg 仓库先发布一个预发布版本。
2. 下载脚本改为读取 `ffmpeg_url` 和 `ffmpeg_sha256`。
3. 下载后先验证 SHA-256，再解压到临时目录。
4. 在临时目录执行 `ffmpeg -version` 和功能探测。
5. 验证通过后原子替换正式运行时目录；失败则保留旧版本。
6. 启动推流前探测实际编码器，按 Vulkan Video、厂商硬件编码、CPU 回退策略选择后端。
7. Windows/NVIDIA 的高级网络推流在 Vulkan 路径尚未通过真机验收前，继续使用稳定的 `h264_nvenc`。
8. MediaMTX 版本、配置和下载逻辑保持独立，不随 FFmpeg 包覆盖。

建议保留一版旧 FFmpeg 作为回滚包。新构建通过高级网络推流、GPU 推流、音频回环、WebRTC 头显播放测试后，再删除第三方旧压缩包。

## 许可证与再分发要求

本方案包含 `--enable-gpl`、libx264 和 libx265，因此发布的是 GPL 配置的 FFmpeg，不是纯 LGPL 配置。

至少完成以下事项：

1. 不启用 `--enable-nonfree`。
2. 在每个 Release 同时提供与二进制精确对应的 FFmpeg 源码。
3. 提供所有补丁、完整 configure 参数和依赖版本。
4. 保留 FFmpeg 以及各依赖的许可证文本。
5. 对静态链接进入产物的依赖，逐项核对其许可证和源码提供义务。
6. 在应用关于页面、下载页面和发行说明中注明使用 FFmpeg。
7. 商业发布前单独评估 H.264/H.265 在目标地区的专利许可问题。

FFmpeg 官方许可证说明明确指出：启用 GPL 组件后 GPL 适用于整个 FFmpeg 构建；其合规清单还要求分发与二进制对应的源码、构建方式，并建议源码与二进制放在同一服务器。参考：[FFmpeg Legal](https://ffmpeg.org/legal.html)。

> 本节是工程合规提示，不构成法律意见。

## 常见问题

### FFmpeg 版本很新，为什么没有 `h264_vulkan`？

版本号不能证明编译功能。检查：

```bash
ffmpeg -buildconf
ffmpeg -encoders | grep vulkan
ffmpeg -hwaccels
```

构建机缺少 Vulkan Headers/Loader、编码器依赖未满足或配置阶段自动禁用，都会导致最终二进制没有 Vulkan 编码器。

### `--enable-vulkan` 成功，为什么真机仍不能编码？

编译成功只说明 FFmpeg 找到了开发依赖。运行还要求显卡、驱动、Vulkan Loader、编码队列、编码格式和输入像素格式全部匹配。

### GitHub Actions 为什么不能完成硬件编码测试？

托管 Runner 不是目标 GPU 测试机。Actions 负责可复现构建、CPU 冒烟测试、二进制能力检查和打包；硬件编码必须由自托管 GPU Runner 或发布前真机测试完成。

### 能不能同时得到“全静态、所有后端、所有平台”构建？

不应把它设为第一阶段目标。某些系统框架、驱动加载库和许可证组合不适合完全静态链接。Desktop2Stereo 只需一个可独立部署、目录结构固定、依赖随包完整提供的 CLI 运行时。

### 为什么不能使用 `--enable-nonfree`？

它会让组合后的 FFmpeg 产物不可再分发，和上传公共 GitHub Release 的目标冲突。只有确实需要某个 nonfree 依赖、且产物只在本机内部使用时才单独评估，不能混入公开发行流水线。

### 为什么不直接 Fork FFmpeg 主仓库？

如果没有修改 FFmpeg 源码，不必维护长期 Fork。构建仓库下载固定的官方源码标签，并保存项目自己的脚本和补丁即可。只有确实需要上游尚未合并的代码时，才维护最小补丁或临时 Fork。

## 验收清单

### 仓库

- [ ] 所有源码和依赖版本已固定。
- [ ] 所有下载都验证签名或 SHA-256。
- [ ] 所有编译任务均在 GitHub Actions 远程 Runner 中完成。
- [ ] 本地电脑不需要 MSYS2、编译器、Vulkan SDK 或编解码开发包。
- [ ] 构建脚本不依赖开发者电脑的隐式环境。
- [ ] Pull Request 构建不能访问 Release 密钥。
- [ ] 第三方 Actions 已固定版本，正式阶段固定到 commit SHA。

### 二进制

- [ ] 五个平台键均生成预期压缩包，或明确记录尚未支持的平台。
- [ ] `ffmpeg -version`、`-buildconf`、`-encoders`、`-protocols` 验证通过。
- [ ] libx264、libx265、libopus、SRT 均存在。
- [ ] Windows/Linux 的 Vulkan、NVENC、AMF、QSV 按目标清单验证。
- [ ] macOS 的 VideoToolbox 验证通过。
- [ ] CPU 音视频冒烟文件可由 ffprobe 正确读取。
- [ ] NVIDIA、AMD、Intel 的可用后端经过对应真机测试。

### 发布

- [ ] 每个文件都有 SHA-256。
- [ ] Release 包含源码包、补丁、配置和许可证。
- [ ] `runtime-manifest.json` 由工作流自动生成。
- [ ] GitHub Release 是 Desktop2Stereo 的统一 FFmpeg 下载源。
- [ ] Desktop2Stereo 下载后先校验、后安装，并支持回滚。
- [ ] 高级网络推流和 GPU 推流分别完成 4K、音频、WebRTC 头显播放测试。

完成这些项目后，Desktop2Stereo 就不再依赖三个不同网站提供的 FFmpeg 包；版本升级、功能验证、下载地址和故障回滚都由同一套 GitHub 流程控制。
