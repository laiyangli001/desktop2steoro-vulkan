#pragma once

#include <cstdint>

struct FilamentBridge;

void bridge_text_overlay_destroy(FilamentBridge* bridge);
int bridge_text_overlay_set_page_texture(
        FilamentBridge* bridge, uint32_t page,
        const uint8_t* rgba, uint32_t width, uint32_t height);
int bridge_text_overlay_set_page_vertices(
        FilamentBridge* bridge, uint32_t page,
        const float* vertices, uint32_t vertex_count,
        const uint16_t* indices, uint32_t index_count, int visible);
