#pragma once

#include "filament_bridge.h"

FilamentBridge* bridge_context_create(const FilamentBridgeVulkanCreateInfo* info);
void bridge_context_destroy(FilamentBridge* bridge);
int bridge_context_wait_for_idle(FilamentBridge* bridge);
const char* bridge_context_last_error(const FilamentBridge* bridge);
