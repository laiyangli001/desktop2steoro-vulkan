#version 450
layout(push_constant) uniform PanoramaParams {
    vec4 fov_tangents;
    vec4 eye_orientation;
} params;
layout(location=0) out vec3 direction;

vec3 rotate_by_quaternion(vec3 value, vec4 quaternion) {
    vec3 q = quaternion.xyz;
    return value
        + 2.0 * cross(q, cross(q, value) + quaternion.w * value);
}

void main() {
    vec2 p = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2);
    vec2 ndc = p * 2.0 - 1.0;
    vec2 screen_uv = ndc * 0.5 + 0.5;
    vec3 view_direction = vec3(
        mix(params.fov_tangents.x, params.fov_tangents.y, screen_uv.x),
        mix(params.fov_tangents.w, params.fov_tangents.z, screen_uv.y),
        -1.0
    );
    direction = rotate_by_quaternion(
        view_direction, normalize(params.eye_orientation)
    );
    gl_Position = vec4(ndc, 0.0, 1.0);
}
