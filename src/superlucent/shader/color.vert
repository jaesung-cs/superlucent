#version 450
#extension GL_ARB_separate_shader_objects : enable

// Vertex
layout (location = 0) in vec3 position;
layout (location = 1) in vec3 color;

layout (location = 0) out vec3 frag_color;

const vec3 positions[3] = vec3[](
  vec3(0.f, 0.f, 0.f),
  vec3(1.f, 0.f, 0.f),
  vec3(0.f, 1.f, 0.f)
);

const vec3 colors[3] = vec3[](
  vec3(1.f, 0.f, 0.f),
  vec3(0.f, 1.f, 0.f),
  vec3(0.f, 0.f, 1.f)
);

void main()
{
  // gl_Position = camera.projection * camera.view * model.model * vec4(position, 1.f);
  gl_Position = vec4(positions[gl_VertexIndex], 1.f);
  frag_color = colors[gl_VertexIndex];
}

/*
#version 450
#extension GL_ARB_separate_shader_objects : enable

// Vertex
layout (location = 0) in vec3 position;
layout (location = 1) in vec3 color;

layout (std140, binding = 0) uniform Camera
{
  mat4 projection;
  mat4 view;
  vec3 eye;
} camera;

layout (std140, binding = 1) uniform ModelUbo
{
  mat4 model;
  mat3 model_inverse_transpose;
} model;

layout (location = 0) out vec3 frag_color;

void main()
{
  gl_Position = camera.projection * camera.view * model.model * vec4(position, 1.f);
  frag_color = color;
}
*/
