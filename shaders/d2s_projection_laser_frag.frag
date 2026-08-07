#version 450

layout(push_constant) uniform LaserView {
    mat4 view_projection;
} view;

layout(set = 0, binding = 0, std430) readonly buffer LaserState {
    mat4 model;
    float time;
} state;


layout(location = 0) in vec2 laser_uv;
layout(location = 0) out vec4 output_color;

void main() {
    float t = fract(laser_uv.y + state.time * 0.4);
    vec3 color;
    if (t < 0.167) {
        color = mix(vec3(0.0, 0.4, 1.0), vec3(0.0, 1.0, 1.0), t / 0.167);
    } else if (t < 0.333) {
        color = mix(vec3(0.0, 1.0, 1.0), vec3(0.0, 1.0, 0.0), (t - 0.167) / 0.166);
    } else if (t < 0.5) {
        color = mix(vec3(0.0, 1.0, 0.0), vec3(1.0, 1.0, 0.0), (t - 0.333) / 0.167);
    } else if (t < 0.667) {
        color = mix(vec3(1.0, 1.0, 0.0), vec3(1.0, 0.5, 0.0), (t - 0.5) / 0.167);
    } else if (t < 0.833) {
        color = mix(vec3(1.0, 0.5, 0.0), vec3(1.0, 0.0, 0.0), (t - 0.667) / 0.166);
    } else {
        color = mix(vec3(1.0, 0.0, 0.0), vec3(0.0, 0.4, 1.0), (t - 0.833) / 0.167);
    }
    output_color = vec4(color, 1.0);
}
