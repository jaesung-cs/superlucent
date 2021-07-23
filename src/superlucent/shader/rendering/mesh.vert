#version 450
#extension GL_ARB_separate_shader_objects : enable

// Vertex
layout (location = 0) in vec3 position;
layout (location = 1) in vec3 normal;
layout (location = 2) in vec3 color;

layout (std140, binding = 0) uniform Camera
{
  mat4 projection;
  mat4 view;
  vec3 eye;
} camera;

layout (push_constant) uniform PushConstants
{
  mat4 model;
};

layout (location = 0) out vec3 frag_position;
layout (location = 1) out vec3 frag_normal;
layout (location = 2) out vec3 frag_color;

void main()
{
  const vec4 p = model * vec4(position, 1.f);
  gl_Position = camera.projection * camera.view * p;
  frag_position = p.xyz / p.w;
  frag_normal = normal;
  frag_color = color;
}
