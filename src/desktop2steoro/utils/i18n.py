"""Shared i18n module for Desktop2Stereo. Accessible from both GUI and subprocess.
MESSAGES is read-only — never modify at runtime.
"""
from __future__ import annotations
import logging
import os
from typing import Any

# ── translation data (read-only) ──

MESSAGES: dict[str, dict[str, str]] = {
    "EN": {
        "Preparing environment": "📦 Preparing environment...",
        "Checking model cache": "🔍 Checking model cache...",
        "Downloading model": "⬇️ Downloading AI model {model}...",
        "Downloading model (first)": "⬇️ Downloading AI model {model} (first time, may take several minutes)...",
        "Exporting ONNX": "⚙️ Exporting ONNX file: {filename}",
        "Building TensorRT": "🔧 Building TensorRT engine (this may take a while)...",
        "Starting capture": "🚀 Starting capture...",
        "Loading model": "📤 Loading model {model}...",
        "Running": "✅ Running",
        "Stopped": "⏹ Stopped",
        "Error occurred": "❌ Error occurred",
        "Fatal error": "❌ Fatal error: {error}",
        "Shutting down": "🛑 Shutting down...",
        "Ready": "✅ Ready",
        "capture_started": "Capture started (monitor #{index})",
        "capture_stopped": "Capture stopped",
        "depth_loaded": "Depth model loaded: {model}",
        "trt_loaded": "TensorRT engine loaded: {path}",
    },
    "CN": {
        "Preparing environment": "📦 正在准备运行环境...",
        "Checking model cache": "🔍 正在检查模型缓存...",
        "Downloading model": "⬇️ 正在下载AI模型 {model}...",
        "Downloading model (first)": "⬇️ 首次下载AI模型 {model}，可能需要几分钟...",
        "Exporting ONNX": "⚙️ 正在导出ONNX文件：{filename}",
        "Building TensorRT": "🔧 正在编译TensorRT引擎（可能需要较长时间）...",
        "Starting capture": "🚀 正在启动采集...",
        "Loading model": "📤 正在加载模型 {model}...",
        "Running": "✅ 运行中",
        "Stopped": "⏹ 已停止",
        "Error occurred": "❌ 出现异常",
        "Fatal error": "❌ 致命错误：{error}",
        "Shutting down": "🛑 正在关闭...",
        "Ready": "✅ 准备就绪",
        "capture_started": "采集已启动（监视器 #{index}）",
        "capture_stopped": "采集已停止",
        "depth_loaded": "深度模型已加载：{model}",
        "trt_loaded": "TensorRT 引擎已加载：{path}",
    },
}

_SUPPORTED_LOCALES = ("EN", "CN")
_DEFAULT_LOCALE = "EN"


def t(key: str, locale: str | None = None, **kwargs: Any) -> str:
    """Translate key to locale with optional format args.

    降级链: 目标 locale → EN → 原始 key
    """
    locale = locale or _DEFAULT_LOCALE
    msg_map = MESSAGES.get(locale, MESSAGES[_DEFAULT_LOCALE])
    template = msg_map.get(key)
    if template is None:
        template = MESSAGES[_DEFAULT_LOCALE].get(key, key)
    if kwargs:
        try:
            return template.format(**kwargs)
        except (KeyError, ValueError):
            return f"{template} [unformatted: {kwargs}]"
    return template


def supported_locales() -> tuple[str, ...]:
    return _SUPPORTED_LOCALES


def is_supported_locale(code: str) -> bool:
    return code in _SUPPORTED_LOCALES


def _resolve_locale() -> str:
    """Resolve locale: env var → default EN."""
    return os.environ.get("DESKTOP2STEREO_LOCALE", _DEFAULT_LOCALE)


def status_log(key: str, level: int = logging.INFO, **kwargs: Any) -> None:
    """Log a translated status message via the 'status' logger.
    Locale is read from the DESKTOP2STEREO_LOCALE env var.
    """
    locale = _resolve_locale()
    msg = t(key, locale, **kwargs)
    logging.getLogger("status").log(level, msg)
