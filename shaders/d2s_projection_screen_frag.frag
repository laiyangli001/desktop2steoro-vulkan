#version 450

layout(set = 0, binding = 0) uniform sampler2D screen_texture;
layout(location = 0) in vec2 texture_uv;
layout(location = 0) out vec4 output_color;

void main() {
    output_color = vec4(texture(screen_texture, texture_uv).rgb, 1.0);
}
