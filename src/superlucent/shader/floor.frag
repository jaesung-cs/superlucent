#version 450
#extension GL_ARB_separate_shader_objects : enable

layout (location = 0) in vec2 frag_position;
layout (location = 1) in vec2 frag_tex_coord;

layout (binding = 2) uniform sampler2D tex;

layout (location = 0) out vec4 out_color;

const float r1 = 10.f;
const float r2 = 15.f;

void main()
{
  const float r = length(frag_position);
  const float alpha = 1.f - smoothstep(r1, r2, r);

  out_color = vec4(texture(tex, frag_tex_coord).rgb, alpha);
}
