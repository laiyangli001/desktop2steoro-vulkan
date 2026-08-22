#version 450

layout(set = 0, binding = 0) uniform sampler2D source_texture;
layout(push_constant) uniform QualityParams {
    vec4 source_texel_mode;
} params;

layout(location = 0) in vec2 texture_uv;
layout(location = 0) out vec4 output_color;

float luma(vec3 color) {
    return dot(color, vec3(0.299, 0.587, 0.114));
}

float sinc(float value) {
    float magnitude = abs(value);
    if (magnitude < 1e-4) return 1.0;
    float pi_value = 3.14159265358979323846 * value;
    return sin(pi_value) / pi_value;
}

float lanczos2_weight(float value) {
    return abs(value) < 2.0 ? sinc(value) * sinc(value * 0.5) : 0.0;
}

vec3 easu_source(vec2 pixel, vec2 texel) {
    return texture(source_texture, clamp((pixel + vec2(0.5)) * texel, vec2(0.0), vec2(1.0))).rgb;
}

void easu_set(inout vec2 direction, inout float length_value, float weight,
        float a, float b, float c, float d, float e) {
    float direction_x = d - b;
    float length_x = 1.0 / max(max(abs(d - c), abs(c - b)), 1e-6);
    direction.x += direction_x * weight;
    length_x = clamp(abs(direction_x) * length_x, 0.0, 1.0);
    length_value += length_x * length_x * weight;
    float direction_y = e - a;
    float length_y = 1.0 / max(max(abs(e - c), abs(c - a)), 1e-6);
    direction.y += direction_y * weight;
    length_y = clamp(abs(direction_y) * length_y, 0.0, 1.0);
    length_value += length_y * length_y * weight;
}

void easu_tap(inout vec3 color, inout float weight_sum, vec2 offset,
        vec2 direction, vec2 length_value, float lobe, float clip_value, vec3 sample_color) {
    vec2 rotated = vec2(offset.x * direction.x + offset.y * direction.y,
            offset.x * -direction.y + offset.y * direction.x) * length_value;
    float distance_squared = min(dot(rotated, rotated), clip_value);
    float weight_b = 0.4 * distance_squared - 1.0;
    float weight_a = lobe * distance_squared - 1.0;
    weight_b = 1.5625 * weight_b * weight_b - 0.5625;
    float weight = weight_b * weight_a * weight_a;
    color += sample_color * weight;
    weight_sum += weight;
}

vec3 easu(vec2 uv, vec2 source_texel) {
    vec2 source_size = 1.0 / source_texel;
    vec2 output_size = source_size * vec2(max(params.source_texel_mode.w, 1.0));
    vec2 output_uv = (floor(uv * output_size) + vec2(0.5)) / output_size;
    vec2 source_position = output_uv * source_size - vec2(0.5);
    vec2 source_base = floor(source_position);
    vec2 pp = source_position - source_base;
    vec3 b = easu_source(source_base + vec2(0.0, -1.0), source_texel);
    vec3 c = easu_source(source_base + vec2(1.0, -1.0), source_texel);
    vec3 e = easu_source(source_base + vec2(-1.0, 0.0), source_texel);
    vec3 f = easu_source(source_base, source_texel);
    vec3 g = easu_source(source_base + vec2(1.0, 0.0), source_texel);
    vec3 h = easu_source(source_base + vec2(2.0, 0.0), source_texel);
    vec3 i = easu_source(source_base + vec2(-1.0, 1.0), source_texel);
    vec3 j = easu_source(source_base + vec2(0.0, 1.0), source_texel);
    vec3 k = easu_source(source_base + vec2(1.0, 1.0), source_texel);
    vec3 l = easu_source(source_base + vec2(2.0, 1.0), source_texel);
    vec3 n = easu_source(source_base + vec2(0.0, 2.0), source_texel);
    vec3 o = easu_source(source_base + vec2(1.0, 2.0), source_texel);
    vec2 direction = vec2(0.0); float length_value = 0.0;
    easu_set(direction, length_value, (1.0 - pp.x) * (1.0 - pp.y), luma(b), luma(e), luma(f), luma(g), luma(j));
    easu_set(direction, length_value, pp.x * (1.0 - pp.y), luma(c), luma(f), luma(g), luma(h), luma(k));
    easu_set(direction, length_value, (1.0 - pp.x) * pp.y, luma(f), luma(i), luma(j), luma(k), luma(n));
    easu_set(direction, length_value, pp.x * pp.y, luma(g), luma(j), luma(k), luma(l), luma(o));
    float direction_length = dot(direction, direction);
    direction = direction_length < 0.000030517578125 ? vec2(1.0, 0.0) : direction / sqrt(direction_length);
    length_value = 0.25 * length_value * length_value;
    float stretch = 1.0 / max(max(abs(direction.x), abs(direction.y)), 1e-6);
    vec2 length_squared = vec2(1.0 + (stretch - 1.0) * length_value, 1.0 - 0.5 * length_value);
    float lobe = 0.5 + (0.21 - 0.5) * length_value;
    float clip_value = 1.0 / max(lobe, 1e-6);
    vec3 min4 = min(min(f, g), min(j, k)); vec3 max4 = max(max(f, g), max(j, k));
    vec3 color = vec3(0.0); float weight_sum = 0.0;
    easu_tap(color, weight_sum, vec2(0.0, -1.0) - pp, direction, length_squared, lobe, clip_value, b);
    easu_tap(color, weight_sum, vec2(1.0, -1.0) - pp, direction, length_squared, lobe, clip_value, c);
    easu_tap(color, weight_sum, vec2(-1.0, 1.0) - pp, direction, length_squared, lobe, clip_value, i);
    easu_tap(color, weight_sum, vec2(0.0, 1.0) - pp, direction, length_squared, lobe, clip_value, j);
    easu_tap(color, weight_sum, vec2(0.0, 0.0) - pp, direction, length_squared, lobe, clip_value, f);
    easu_tap(color, weight_sum, vec2(-1.0, 0.0) - pp, direction, length_squared, lobe, clip_value, e);
    easu_tap(color, weight_sum, vec2(1.0, 1.0) - pp, direction, length_squared, lobe, clip_value, k);
    easu_tap(color, weight_sum, vec2(2.0, 1.0) - pp, direction, length_squared, lobe, clip_value, l);
    easu_tap(color, weight_sum, vec2(2.0, 0.0) - pp, direction, length_squared, lobe, clip_value, h);
    easu_tap(color, weight_sum, vec2(1.0, 0.0) - pp, direction, length_squared, lobe, clip_value, g);
    easu_tap(color, weight_sum, vec2(1.0, 2.0) - pp, direction, length_squared, lobe, clip_value, o);
    vec3 result = weight_sum <= 1e-6 ? f : color / weight_sum;
    return min(max4, max(min4, result));
}

vec3 lanczos2(vec2 uv, vec2 texel) {
    vec2 source_position = uv / texel - vec2(0.5);
    vec2 source_base = floor(source_position);
    vec3 color = vec3(0.0); float total = 0.0;
    for (int y = -1; y <= 2; ++y) for (int x = -1; x <= 2; ++x) {
        vec2 pixel = source_base + vec2(float(x), float(y));
        vec2 delta = source_position - pixel;
        float weight = lanczos2_weight(delta.x) * lanczos2_weight(delta.y);
        color += texture(source_texture, clamp((pixel + vec2(0.5)) * texel, vec2(0.0), vec2(1.0))).rgb * weight;
        total += weight;
    }
    return color / max(total, 1e-6);
}

void main() {
    vec2 texel = params.source_texel_mode.xy;
    float mode = params.source_texel_mode.z;
    vec3 color = mode > 1.5 ? easu(texture_uv, texel) : lanczos2(texture_uv, texel);
    output_color = vec4(color, 1.0);
}
