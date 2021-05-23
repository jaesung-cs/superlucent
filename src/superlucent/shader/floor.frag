#version 450
#extension GL_ARB_separate_shader_objects : enable

layout (location = 0) in vec2 frag_tex_coord;

layout (location = 0) out vec4 out_color;

void main()
{
  out_color = vec4(frag_tex_coord, 0.f, 1.f);
}
