#version 450
#extension GL_EXT_multiview : require

layout(location = 0) flat out uint viewIndexFromVertex;

vec2 positions[3] = vec2[](
    vec2(-1.0, -1.0),
    vec2( 3.0, -1.0),
    vec2(-1.0,  3.0)
);

void main() {
    viewIndexFromVertex = gl_ViewIndex;
    gl_Position = vec4(positions[gl_VertexIndex], 0.0, 1.0);
}
