#version 450

layout(push_constant) uniform ScreenParams {
    mat4 view_projection;
    vec4 center;
    vec4 right;
    vec4 up;
    vec4 size_curve;
} screen;

layout(set = 0, binding = 1, std430) readonly buffer GlowState {
    vec4 head_mode;
    vec4 glow;
    vec4 frost0;
    vec4 frost1;
    vec4 veil;
    vec4 shell;
} state;

layout(location = 0) out vec2 texture_uv;
layout(location = 1) out vec2 effect_uv;

const int GLOW_SEGMENTS = 64;
const int GLOW_SHELL_SEGMENTS = 96;
const int SHELL_RADIAL_SEGMENTS = 48;
const int FLAT_DEPTH_STEPS = 8;
const int FLAT_EDGE_STEPS = 8;

vec3 screen_forward() {
    return normalize(cross(screen.right.xyz, screen.up.xyz));
}

vec3 screen_surface(float u, float v) {
    float half_width = screen.size_curve.x;
    float half_height = screen.size_curve.y;
    float local_x = (u - 0.5) * half_width * 2.0;
    float local_z = 0.0;
    if (screen.size_curve.z > 0.0) {
        float radius = half_width / screen.size_curve.z;
        float angle = mix(-screen.size_curve.z, screen.size_curve.z, u);
        local_x = radius * sin(angle);
        local_z = radius * (1.0 - cos(angle));
    }
    return screen.center.xyz
        + screen.right.xyz * local_x
        + screen.up.xyz * ((v - 0.5) * 2.0 * half_height)
        + screen_forward() * local_z;
}

vec3 extended_screen_surface(float local_x, float local_y) {
    float half_width = max(screen.size_curve.x, 0.00001);
    float half_height = max(screen.size_curve.y, 0.00001);
    float clamped_x = clamp(local_x, -half_width, half_width);
    float clamped_y = clamp(local_y, -half_height, half_height);
    float u = clamped_x / (half_width * 2.0) + 0.5;
    float v = clamped_y / (half_height * 2.0) + 0.5;
    vec3 result = screen_surface(u, v);
    vec3 tangent = screen.right.xyz;
    if (screen.size_curve.z > 0.0) {
        float angle = mix(-screen.size_curve.z, screen.size_curve.z, u);
        tangent = screen.right.xyz * cos(angle) + screen_forward() * sin(angle);
    }
    result += tangent * (local_x - clamped_x);
    result += screen.up.xyz * (local_y - clamped_y);
    return result;
}

vec3 frost_front(float u, float v) {
    vec3 local_head = state.head_mode.xyz - screen.center.xyz;
    float head_x = dot(local_head, screen.right.xyz);
    float head_y = dot(local_head, screen.up.xyz);
    float head_z = dot(local_head, screen_forward());
    float screen_distance = abs(head_z);
    float front_depth = max(max(head_z + 0.55, screen_distance + 0.35), 0.75);
    float front_half_width = max(screen.size_curve.x, abs(head_x) + 0.65);
    float front_half_height = max(screen.size_curve.y, abs(head_y) + 0.65);
    return screen.center.xyz
        + screen.right.xyz * ((u * 2.0 - 1.0) * front_half_width)
        + screen.up.xyz * ((v * 2.0 - 1.0) * front_half_height)
        + screen_forward() * front_depth;
}

void quad_corner(int vertex, out bool high_s, out bool high_t) {
    int corner = int[6](0, 1, 2, 1, 3, 2)[vertex % 6];
    high_s = (corner & 1) != 0;
    high_t = (corner & 2) != 0;
}

vec2 shell_edge_uv(int side, float along) {
    if (side == 0) return vec2(along, 1.0);
    if (side == 1) return vec2(1.0, 1.0 - along);
    if (side == 2) return vec2(1.0 - along, 0.0);
    return vec2(0.0, along);
}

vec3 spherical_interpolate(vec3 start, vec3 end, float amount) {
    float cosine = clamp(dot(start, end), -1.0, 1.0);
    float angle = acos(cosine);
    float sine = sin(angle);
    if (abs(sine) <= 0.00001) {
        return normalize(start * (1.0 - amount) + end * amount);
    }
    return normalize(
        start * (sin((1.0 - amount) * angle) / sine)
        + end * (sin(amount * angle) / sine)
    );
}

vec3 shell_surface(int side, float radial_t, float along, out vec2 source_uv) {
    source_uv = shell_edge_uv(side, along);
    vec3 source_position = screen_surface(source_uv.x, source_uv.y);
    vec3 shell_forward = screen.center.xyz - state.head_mode.xyz;
    if (length(shell_forward) <= 0.00001) {
        shell_forward = -screen_forward();
    } else {
        shell_forward = normalize(shell_forward);
    }
    vec3 shell_right = screen.right.xyz
        - shell_forward * dot(screen.right.xyz, shell_forward);
    if (length(shell_right) <= 0.00001) {
        shell_right = cross(shell_forward, screen.up.xyz);
    }
    shell_right = normalize(shell_right);
    vec3 shell_up = normalize(cross(shell_right, shell_forward));
    vec3 source_direction = source_position - state.head_mode.xyz;
    float source_distance = length(source_direction);
    if (source_distance <= 0.00001) {
        source_direction = shell_forward;
        source_distance = 0.0;
    } else {
        source_direction /= source_distance;
    }
    float rim_x = (source_uv.x - 0.5) * 2.0;
    float rim_y = (source_uv.y - 0.5) * 2.0;
    vec3 rim_direction = normalize(
        shell_right * rim_x + shell_up * rim_y + shell_forward * 0.02
    );
    vec3 direction = spherical_interpolate(source_direction, rim_direction, radial_t);
    float radius = max(state.shell.x, max(max(
        screen.size_curve.x * 2.0,
        screen.size_curve.y * 2.0
    ), 1.0) * 0.85);
    float height = max(state.shell.y, screen.size_curve.y * 2.0 * 1.8);
    float vertical_radius = max(height * 0.5, 1.0);
    float local_x = dot(direction, shell_right);
    float local_y = dot(direction, shell_up);
    float local_z = dot(direction, shell_forward);
    float inverse_distance_squared =
        local_x * local_x / (radius * radius)
        + local_y * local_y / (vertical_radius * vertical_radius)
        + local_z * local_z / (radius * radius);
    float intersection_distance = inversesqrt(max(inverse_distance_squared, 0.00000001));
    vec3 shell_target = state.head_mode.xyz + direction * intersection_distance;
    float shell_distance = length(shell_target - state.head_mode.xyz);
    float surface_distance = mix(source_distance, shell_distance, radial_t);
    return radial_t <= 0.0
        ? source_position
        : state.head_mode.xyz + direction * surface_distance;
}

void main() {
    int mode = int(round(state.head_mode.w));
    vec3 position = vec3(0.0);
    texture_uv = vec2(0.0);
    effect_uv = vec2(0.0);
    if (mode == 1 || mode == 2) {
        int column = gl_VertexIndex / 2;
        bool top = (gl_VertexIndex & 1) != 0;
        float u = float(column) / float(GLOW_SEGMENTS);
        float base_width = max(state.glow.y, 0.75);
        float screen_long = max(max(
            screen.size_curve.x * 2.0,
            screen.size_curve.y * 2.0
        ), 2.4);
        float distance_to_head = max(
            length(state.head_mode.xyz - screen.center.xyz),
            0.5
        );
        float range = base_width * (screen_long / 2.4)
            * (distance_to_head / 2.0) * 20.0;
        if (mode == 2) range *= 0.5;
        float glow_width = screen.size_curve.x * 2.0 + range * 2.0;
        float glow_height = screen.size_curve.y * 2.0 + range * 2.0;
        float local_x = (u - 0.5) * glow_width;
        float local_y = top ? glow_height * 0.5 : -glow_height * 0.5;
        position = extended_screen_surface(local_x, local_y);
        texture_uv = vec2(u, top ? 1.0 : 0.0);
    } else if (mode == 3 || mode == 4) {
        int quad = gl_VertexIndex / 6;
        bool high_s;
        bool high_t;
        quad_corner(gl_VertexIndex, high_s, high_t);
        float u;
        float v;
        float depth;
        if (screen.size_curve.z <= 0.0) {
            int side = quad / (FLAT_DEPTH_STEPS * FLAT_EDGE_STEPS);
            int within = quad % (FLAT_DEPTH_STEPS * FLAT_EDGE_STEPS);
            int depth_step = within / FLAT_EDGE_STEPS;
            int edge_step = within % FLAT_EDGE_STEPS;
            float edge_t = (float(edge_step) + (high_s ? 1.0 : 0.0))
                / float(FLAT_EDGE_STEPS);
            depth = (float(depth_step) + (high_t ? 1.0 : 0.0))
                / float(FLAT_DEPTH_STEPS);
            vec2 a = vec2[4](
                vec2(0.0, 0.0), vec2(1.0, 0.0),
                vec2(1.0, 1.0), vec2(0.0, 1.0)
            )[side];
            vec2 b = vec2[4](
                vec2(1.0, 0.0), vec2(1.0, 1.0),
                vec2(0.0, 1.0), vec2(0.0, 0.0)
            )[side];
            vec2 uv = mix(a, b, edge_t);
            u = uv.x;
            v = uv.y;
        } else {
            depth = high_t ? 1.0 : 0.0;
            if (quad < GLOW_SEGMENTS * 2) {
                bool top_wall = quad >= GLOW_SEGMENTS;
                int segment = quad % GLOW_SEGMENTS;
                u = (float(segment) + (high_s ? 1.0 : 0.0))
                    / float(GLOW_SEGMENTS);
                v = top_wall ? 1.0 : 0.0;
            } else {
                bool right_wall = quad == GLOW_SEGMENTS * 2 + 1;
                u = right_wall ? 1.0 : 0.0;
                v = high_s ? 1.0 : 0.0;
            }
        }
        vec3 rear = screen_surface(u, v);
        vec3 front = frost_front(u, v);
        position = mix(rear, front, depth);
        texture_uv = vec2(u, v);
        effect_uv = vec2(depth, 0.0);
    } else {
        int quad = gl_VertexIndex / 6;
        bool high_segment;
        bool high_radial;
        quad_corner(gl_VertexIndex, high_segment, high_radial);
        int quads_per_side = SHELL_RADIAL_SEGMENTS * GLOW_SHELL_SEGMENTS;
        int side = quad / quads_per_side;
        int within = quad % quads_per_side;
        int radial = within / GLOW_SHELL_SEGMENTS;
        int segment = within % GLOW_SHELL_SEGMENTS;
        float radial_t = (float(radial) + (high_radial ? 1.0 : 0.0))
            / float(SHELL_RADIAL_SEGMENTS);
        float along = (float(segment) + (high_segment ? 1.0 : 0.0))
            / float(GLOW_SHELL_SEGMENTS);
        position = shell_surface(side, radial_t, along, texture_uv);
        effect_uv = vec2(radial_t, 0.0);
    }
    gl_Position = screen.view_projection * vec4(position, 1.0);
}
