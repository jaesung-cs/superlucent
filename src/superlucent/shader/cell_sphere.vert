#version 450
#extension GL_ARB_separate_shader_objects : enable

// Vertex
layout (location = 0) in vec3 position;
layout (location = 1) in vec3 normal;

// Instance
layout (location = 2) in vec3 center;
layout (location = 3) in vec3 color;

layout (std140, binding = 0) uniform Camera
{
  mat4 projection;
  mat4 view;
  vec3 eye;
} camera;

layout (push_constant) uniform PushConstants
{
  float radius;
} push_constants;

layout (location = 0) out vec3 frag_position;
layout (location = 1) out vec3 frag_normal;
layout (location = 2) out vec3 frag_color;

void main()
{
  frag_position = center + push_constants.radius * position;
  gl_Position = camera.projection * camera.view * vec4(frag_position, 1.f);
  frag_normal = normal;
  frag_color = color;
}
