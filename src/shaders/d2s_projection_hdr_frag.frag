#version 450

layout(set = 0, binding = 0) uniform sampler2D source_texture;
layout(push_constant) uniform HdrParams {
    vec4 values;
} params;

layout(location = 0) in vec2 texture_uv;
layout(location = 0) out vec4 output_color;

void main() {
    vec4 source = texture(source_texture, texture_uv);
    vec3 exposed = max(source.rgb, vec3(0.0)) * exp2(params.values.x);
    output_color = vec4(clamp(exposed, 0.0, 1.0), clamp(source.a, 0.0, 1.0));
}
