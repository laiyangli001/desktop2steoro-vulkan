#version 450

layout(location = 0) in vec3 world_normal;
layout(location = 1) in vec3 base_color;
layout(location = 2) flat in int draw_kind;
layout(location = 3) in vec2 laser_uv;
layout(location = 4) in float laser_time;
layout(location = 5) noperspective in vec2 face_uv;
layout(location = 0) out vec4 output_color;

vec3 rainbow_color(float t) {
    if (t < 0.167) {
        return mix(vec3(0.0, 0.4, 1.0), vec3(0.0, 1.0, 1.0), t / 0.167);
    }
    if (t < 0.333) {
        return mix(vec3(0.0, 1.0, 1.0), vec3(0.0, 1.0, 0.0), (t - 0.167) / 0.166);
    }
    if (t < 0.5) {
        return mix(vec3(0.0, 1.0, 0.0), vec3(1.0, 1.0, 0.0), (t - 0.333) / 0.167);
    }
    if (t < 0.667) {
        return mix(vec3(1.0, 1.0, 0.0), vec3(1.0, 0.5, 0.0), (t - 0.5) / 0.167);
    }
    if (t < 0.833) {
        return mix(vec3(1.0, 0.5, 0.0), vec3(1.0, 0.0, 0.0), (t - 0.667) / 0.166);
    }
    return mix(vec3(1.0, 0.0, 0.0), vec3(0.0, 0.4, 1.0), (t - 0.833) / 0.167);
}

void main() {
    if (draw_kind == 1) {
        // Raw Vulkan UV interpolation has the opposite animated phase from
        // Filament's material UV convention. Reverse time to preserve the
        // legacy visible flow from the controller toward the tapered tip.
        float t = fract(laser_uv.y - laser_time * 0.4);
        output_color = vec4(rainbow_color(t), 1.0);
        return;
    }
    // The shared overlay render pass has no depth attachment. Reject cube
    // back faces explicitly so they cannot overwrite the visible front faces.
    if (!gl_FrontFacing) {
        discard;
    }
    vec2 edge_distance = min(face_uv, vec2(1.0) - face_uv);
    float nearest_edge = min(edge_distance.x, edge_distance.y);
    float antialias_width = max(fwidth(nearest_edge) * 1.5, 0.001);
    float interior = smoothstep(0.0, antialias_width, nearest_edge);
    output_color = vec4(mix(vec3(0.0), base_color, interior), 1.0);
}
