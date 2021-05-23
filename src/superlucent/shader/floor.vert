#version 450
#extension GL_ARB_separate_shader_objects : enable

layout (location = 0) in vec2 position;

layout (std140, binding = 0) uniform Camera
{
  mat4 projection;
  mat4 view;
  vec3 eye;
} camera;

layout (location = 0) out vec2 frag_tex_coord;

void main()
{
  gl_Position = camera.projection * camera.view * vec4(position, 0.f, 1.f);
  frag_tex_coord = position;
}
