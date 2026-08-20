#version 450

layout(push_constant) uniform ControllerView {
    mat4 view_projection;
} view;

layout(set = 0, binding = 0, std430) readonly buffer ControllerState {
    mat4 controller_model[2];
    mat4 laser_model[2];
    vec4 controller_visible;
    vec4 laser_state;
} state;

layout(location = 0) out vec3 world_normal;
layout(location = 1) out vec3 base_color;
layout(location = 2) flat out int draw_kind;
layout(location = 3) out vec2 laser_uv;
layout(location = 4) out float laser_time;
layout(location = 5) noperspective out vec2 face_uv;

const int triangle_corner[6] = int[6](0, 1, 2, 1, 3, 2);
const int CUBE_VERTEX_COUNT = 36;
const int CONTROLLER_VERTEX_COUNT = CUBE_VERTEX_COUNT * 2;
const vec3 face_normal[6] = vec3[6](
    vec3(1.0, 0.0, 0.0), vec3(-1.0, 0.0, 0.0),
    vec3(0.0, 1.0, 0.0), vec3(0.0, -1.0, 0.0),
    vec3(0.0, 0.0, 1.0), vec3(0.0, 0.0, -1.0)
);

vec3 face_position(int face, int corner) {
    float a = ((corner & 1) != 0) ? 1.0 : -1.0;
    float b = ((corner & 2) != 0) ? 1.0 : -1.0;
    if (face == 0) return vec3(1.0, a, b);
    if (face == 1) return vec3(-1.0, a, -b);
    if (face == 2) return vec3(a, 1.0, -b);
    if (face == 3) return vec3(a, -1.0, b);
    if (face == 4) return vec3(a, b, 1.0);
    return vec3(-a, b, -1.0);
}

void main() {
    draw_kind = gl_VertexIndex < CONTROLLER_VERTEX_COUNT ? 0 : 1;
    laser_uv = vec2(0.0);
    laser_time = state.laser_state.x;
    face_uv = vec2(0.5);
    if (draw_kind == 0) {
        int hand = gl_VertexIndex / CUBE_VERTEX_COUNT;
        int local_vertex = gl_VertexIndex % CUBE_VERTEX_COUNT;
        int face = local_vertex / 6;
        int corner = triangle_corner[local_vertex % 6];
        if (state.controller_visible[hand] < 0.5) {
            gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
            world_normal = vec3(0.0, 0.0, 1.0);
            base_color = vec3(0.0);
            return;
        }
        vec3 local_position = face_position(face, corner) * 0.040;
        vec4 world_position = state.controller_model[hand] * vec4(local_position, 1.0);
        world_normal = normalize(mat3(state.controller_model[hand]) * face_normal[face]);
        base_color = hand == 0 ? vec3(0.10, 0.65, 0.95) : vec3(1.00, 0.48, 0.08);
        face_uv = vec2(
            ((corner & 1) != 0) ? 1.0 : 0.0,
            ((corner & 2) != 0) ? 1.0 : 0.0
        );
        gl_Position = view.view_projection * world_position;
        return;
    }

    int laser_vertex = gl_VertexIndex - CONTROLLER_VERTEX_COUNT;
    int hand = laser_vertex / 12;
    int local_vertex = laser_vertex % 12;
    if (state.laser_state[hand + 1] < 0.5) {
        gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
        world_normal = vec3(0.0, 0.0, 1.0);
        base_color = vec3(0.0);
        return;
    }
    vec3 position;
    if (local_vertex < 6) {
        int corner = triangle_corner[local_vertex];
        bool high_x = (corner & 1) != 0;
        bool high_y = (corner & 2) != 0;
        float half_width = high_y ? (1.0 / 6.0) : 0.5;
        position = vec3(high_x ? half_width : -half_width, high_y ? 1.0 : 0.0, 0.0);
        laser_uv = vec2(high_x ? 1.0 : 0.0, high_y ? 1.0 : 0.0);
    } else {
        int corner = triangle_corner[local_vertex - 6];
        bool high_z = (corner & 1) != 0;
        bool high_y = (corner & 2) != 0;
        float half_width = high_y ? (1.0 / 6.0) : 0.5;
        position = vec3(0.0, high_y ? 1.0 : 0.0, high_z ? half_width : -half_width);
        laser_uv = vec2(high_z ? 1.0 : 0.0, high_y ? 1.0 : 0.0);
    }
    world_normal = vec3(0.0, 0.0, 1.0);
    base_color = vec3(1.0);
    gl_Position = view.view_projection * state.laser_model[hand] * vec4(position, 1.0);
}
