#version 450

layout(push_constant) uniform ScreenParams {
    mat4 view_projection;
    vec4 center;
    vec4 right;
    vec4 up;
    vec4 size_curve;
} params;

layout(location = 0) out vec2 texture_uv;

const int SEGMENTS = 48;

void main() {
    int column = gl_VertexIndex / 2;
    bool top = (gl_VertexIndex & 1) != 0;
    float u = float(column) / float(SEGMENTS);
    float local_x;
    float local_z = 0.0;
    float half_angle = params.size_curve.z;
    if (half_angle > 0.0) {
        float angle = mix(-half_angle, half_angle, u);
        float radius = params.size_curve.x / half_angle;
        local_x = radius * sin(angle);
        local_z = radius * (1.0 - cos(angle));
    } else {
        local_x = mix(-params.size_curve.x, params.size_curve.x, u);
    }
    float local_y = top ? params.size_curve.y : -params.size_curve.y;
    vec3 forward = normalize(cross(params.right.xyz, params.up.xyz));
    vec3 world_position = params.center.xyz
        + params.right.xyz * local_x
        + params.up.xyz * local_y
        + forward * local_z;
    gl_Position = params.view_projection * vec4(world_position, 1.0);
    texture_uv = vec2(u, top ? 0.0 : 1.0);
}
