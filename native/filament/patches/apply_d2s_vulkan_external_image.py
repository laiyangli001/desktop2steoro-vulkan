#!/usr/bin/env python3
"""Apply the pinned Filament 1.74 D2S Vulkan external-image extension."""

from __future__ import annotations

import sys
from pathlib import Path


def replace_once(path: Path, old: str, new: str) -> None:
    text = path.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"expected one replacement in {path}: found {count}")
    path.write_text(text.replace(old, new), encoding="utf-8", newline="")


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit("usage: apply_d2s_vulkan_external_image.py FILAMENT_SOURCE")
    root = Path(sys.argv[1]).resolve()
    platform = root / "filament/backend/include/backend/Platform.h"
    vulkan_platform_h = root / "filament/backend/include/backend/platforms/VulkanPlatform.h"
    vulkan_platform_cpp = root / "filament/backend/src/vulkan/platform/VulkanPlatform.cpp"

    replace_once(
        platform,
        "    class ExternalImageHandle;\n\n    class ExternalImage {\n",
        """    class ExternalImageHandle;

    // Metadata for a borrowed application-owned Vulkan image.
    struct VulkanExternalImageData {
        uint64_t image = 0;
        int32_t format = 0;
        uint32_t width = 0;
        uint32_t height = 0;
    };

    class ExternalImage {
""",
    )
    replace_once(
        platform,
        "    protected:\n        virtual ~ExternalImage() noexcept;\n",
        """    protected:
        ExternalImage() noexcept = default;
        virtual ~ExternalImage() noexcept;

    public:
        virtual bool getVulkanImageData(VulkanExternalImageData& out) const noexcept {
            out = {};
            return false;
        }
""",
    )
    replace_once(
        vulkan_platform_h,
        """    // Note that the image metadata might change per-frame, hence we need a method for extracting
    // it.
    virtual ExternalImageMetadata extractExternalImageMetadata(ExternalImageHandleRef image) const {
        return {};
    }
""",
        """    // Creates a handle for a borrowed application-owned VkImage.
    static ExternalImageHandle createExternalImageFromVkImage(
            VkImage image, VkFormat format, uint32_t width, uint32_t height) noexcept;

    // Note that the image metadata might change per-frame, hence we need a method for extracting
    // it.
    virtual ExternalImageMetadata extractExternalImageMetadata(ExternalImageHandleRef image) const;
""",
    )
    replace_once(
        vulkan_platform_h,
        """    virtual ImageData createVkImageFromExternal(ExternalImageHandleRef image,
            uint32_t logicalWidth, uint32_t logicalHeight) const {
        return {};
    }
""",
        """    virtual ImageData createVkImageFromExternal(ExternalImageHandleRef image,
            uint32_t logicalWidth, uint32_t logicalHeight) const;
""",
    )

    marker = "VulkanPlatform::VulkanPlatform() = default;\n"
    cpp = vulkan_platform_cpp.read_text(encoding="utf-8")
    if cpp.count(marker) != 1:
        raise RuntimeError("VulkanPlatform constructor marker changed")
    helper = """class D2SVulkanExternalImage final : public Platform::ExternalImage {
public:
    explicit D2SVulkanExternalImage(Platform::VulkanExternalImageData data) noexcept
            : mData(data) {}

protected:
    bool getVulkanImageData(Platform::VulkanExternalImageData& out) const noexcept override {
        out = mData;
        return mData.image != 0 && mData.width != 0 && mData.height != 0;
    }

private:
    Platform::VulkanExternalImageData mData;
};

Platform::ExternalImageHandle VulkanPlatform::createExternalImageFromVkImage(
        VkImage image, VkFormat format, uint32_t width, uint32_t height) noexcept {
    if (image == VK_NULL_HANDLE || width == 0 || height == 0) {
        return {};
    }
    return Platform::ExternalImageHandle(new D2SVulkanExternalImage({
            reinterpret_cast<uint64_t>(image), static_cast<int32_t>(format), width, height}));
}

VulkanPlatform::ExternalImageMetadata VulkanPlatform::extractExternalImageMetadata(
        ExternalImageHandleRef image) const {
    Platform::VulkanExternalImageData data;
    if (!image || !image->getVulkanImageData(data)) {
        return {};
    }
    const bool srgb = data.format == static_cast<int32_t>(VK_FORMAT_R8G8B8A8_SRGB);
    return {
            .filamentFormat = srgb ? TextureFormat::SRGB8_A8 : TextureFormat::RGBA8,
            .filamentUsage = TextureUsage::SAMPLEABLE,
            .width = data.width,
            .height = data.height,
            .layers = 1,
            .samples = VK_SAMPLE_COUNT_1_BIT,
            .format = static_cast<VkFormat>(data.format),
            .externalFormat = 0,
            .usage = VK_IMAGE_USAGE_SAMPLED_BIT,
            .allocationSize = 0,
            .memoryTypeBits = 0,
            .ycbcrConversionComponents = {},
            .ycbcrModel = VK_SAMPLER_YCBCR_MODEL_CONVERSION_RGB_IDENTITY,
            .ycbcrRange = VK_SAMPLER_YCBCR_RANGE_ITU_FULL,
            .xChromaOffset = VK_CHROMA_LOCATION_COSITED_EVEN,
            .yChromaOffset = VK_CHROMA_LOCATION_COSITED_EVEN,
            .isStagingRequired = false,
            .isChromaConversionRequired = false};
}

VulkanPlatform::ImageData VulkanPlatform::createVkImageFromExternal(
        ExternalImageHandleRef image, uint32_t, uint32_t) const {
    Platform::VulkanExternalImageData data;
    if (!image || !image->getVulkanImageData(data)) {
        return {};
    }
    ImageData result;
    result.internal.image = reinterpret_cast<VkImage>(data.image);
    result.internal.memory = VK_NULL_HANDLE;
    return result;
}

"""
    vulkan_platform_cpp.write_text(
        cpp.replace(marker, helper + marker), encoding="utf-8", newline=""
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
