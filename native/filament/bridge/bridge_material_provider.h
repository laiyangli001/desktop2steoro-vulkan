#pragma once

#include <gltfio/MaterialProvider.h>

namespace filament {
class Engine;
}

filament::gltfio::MaterialProvider* bridge_create_material_provider(
        filament::Engine* engine);
