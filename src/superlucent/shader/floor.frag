#version 450
#extension GL_ARB_separate_shader_objects : enable

layout (location = 0) in vec2 frag_tex_coord;

layout (binding = 2) uniform sampler2D tex;

layout (location = 0) out vec4 out_color;

void main()
{
  out_color = vec4(texture(tex, frag_tex_coord).rgb, 1.f);
}
