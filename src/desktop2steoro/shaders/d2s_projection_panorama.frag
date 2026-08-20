#version 450
layout(set=0,binding=0) uniform sampler2D source_texture;
layout(location=0) in vec3 direction;
layout(location=0) out vec4 output_color;
const float PI=3.14159265359;
void main(){ vec3 d=normalize(direction); vec2 uv=vec2(atan(d.x,-d.z)/(2.0*PI)+0.5, 0.5-asin(clamp(d.y,-1.0,1.0))/PI); output_color=vec4(texture(source_texture,uv).rgb,1.0); }
