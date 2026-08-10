#version 450

layout(location = 0) out vec4 outColor;
layout(location = 0) flat in uint viewIndexFromVertex;
layout(set = 0, binding = 0) buffer ViewCounts {
    uint count[2];
} viewCounts;

void main() {
    if (gl_FragCoord.x < 1.0 && gl_FragCoord.y < 1.0) {
        atomicAdd(viewCounts.count[viewIndexFromVertex == 0 ? 0 : 1], 1);
    }
    outColor = viewIndexFromVertex == 0
        ? vec4(1.0, 0.0, 0.0, 1.0)
        : vec4(0.0, 1.0, 0.0, 1.0);
}
