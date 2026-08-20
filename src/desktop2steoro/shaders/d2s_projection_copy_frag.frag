#version 450

layout(set = 0, binding = 0) uniform sampler2D source_texture;
layout(location = 0) in vec2 texture_uv;
layout(location = 0) out vec4 output_color;

void main() {
    output_color = texture(source_texture, texture_uv);
}
