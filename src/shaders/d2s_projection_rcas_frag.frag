#version 450

layout(set = 0, binding = 0) uniform sampler2D filtered_texture;
layout(push_constant) uniform RcasParams {
    vec4 texel_sharpness;
} params;

layout(location = 0) in vec2 texture_uv;
layout(location = 0) out vec4 output_color;

float luma(vec3 color) {
    return dot(color, vec3(0.299, 0.587, 0.114));
}

vec3 sample_color(vec2 offset) {
    return texture(filtered_texture, clamp(texture_uv + offset, vec2(0.0), vec2(1.0))).rgb;
}

void main() {
    vec2 texel = params.texel_sharpness.xy;
    vec3 b = sample_color(vec2(0.0, -texel.y));
    vec3 d = sample_color(vec2(-texel.x, 0.0));
    vec3 e = sample_color(vec2(0.0));
    vec3 f = sample_color(vec2(texel.x, 0.0));
    vec3 h = sample_color(vec2(0.0, texel.y));
    float b_luma = luma(b); float d_luma = luma(d); float e_luma = luma(e);
    float f_luma = luma(f); float h_luma = luma(h);
    float noise = 0.25 * (b_luma + d_luma + f_luma + h_luma) - e_luma;
    float l_max = max(max(max(b_luma, d_luma), max(e_luma, f_luma)), h_luma);
    float l_min = min(min(min(b_luma, d_luma), min(e_luma, f_luma)), h_luma);
    noise = clamp(abs(noise) / max(abs(l_max - l_min), 0.000001), 0.0, 1.0);
    noise = -0.5 * noise + 1.0;
    vec3 min4 = min(min(b, d), min(f, h));
    vec3 max4 = max(max(b, d), max(f, h));
    vec3 hit_min = min(min4, e) / max(4.0 * max4, vec3(0.000001));
    vec3 hit_max = (vec3(1.0) - max(max4, e)) /
        min(4.0 * min4 - vec3(4.0), vec3(-0.000001));
    float lobe = max(max(max(-hit_min, hit_max).r, max(-hit_min, hit_max).g), max(-hit_min, hit_max).b);
    float sharpness = clamp(params.texel_sharpness.z, 0.0, 1.0);
    float contrast = exp2(-2.0 * (1.0 - sharpness));
    lobe = max(-(0.25 - 1.0 / 16.0), min(lobe, 0.0)) * contrast * noise;
    float reciprocal_lobe = 1.0 / max(abs(4.0 * lobe + 1.0), 0.000001);
    output_color = vec4(clamp((lobe * (b + d + f + h) + e) * reciprocal_lobe,
        vec3(0.0), vec3(1.0)), 1.0);
}
