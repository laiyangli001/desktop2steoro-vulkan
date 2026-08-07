#version 450

layout(push_constant) uniform LaserView {
    mat4 view_projection;
} view;

layout(set = 0, binding = 0, std430) readonly buffer LaserState {
    mat4 model;
    float time;
} state;

layout(location = 0) out vec2 laser_uv;

void main() {
    int vertex = gl_VertexIndex;
    vec3 position;
    vec2 uv;
    if (vertex < 6) {
        int corner = int[6](0, 1, 2, 1, 3, 2)[vertex];
        bool high_x = (corner & 1) != 0;
        bool high_y = (corner & 2) != 0;
        position = vec3(high_x ? 0.5 : -0.5, high_y ? 1.0 : 0.0, 0.0);
        uv = vec2(high_x ? 1.0 : 0.0, high_y ? 1.0 : 0.0);
    } else {
        int corner = int[6](0, 1, 2, 1, 3, 2)[vertex - 6];
        bool high_z = (corner & 1) != 0;
        bool high_y = (corner & 2) != 0;
        position = vec3(0.0, high_y ? 1.0 : 0.0, high_z ? 0.5 : -0.5);
        uv = vec2(high_z ? 1.0 : 0.0, high_y ? 1.0 : 0.0);
    }
    laser_uv = uv;
    gl_Position = view.view_projection * state.model * vec4(position, 1.0);
}
