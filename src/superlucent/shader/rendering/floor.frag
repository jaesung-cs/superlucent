#version 450
#extension GL_ARB_separate_shader_objects : enable

layout (location = 0) in vec2 frag_position;
layout (location = 1) in vec2 frag_tex_coord;

layout (std140, binding = 0) uniform Camera
{
  mat4 projection;
  mat4 view;
  vec3 eye;
} camera;

layout (binding = 2) uniform sampler2D tex;

struct Light
{
  vec3 position;
  vec3 ambient;
  vec3 diffuse;
  vec3 specular;
};

const int MAX_NUM_LIGHTS = 8;
layout (std140, binding = 1) uniform LightUbo
{
  Light directional_lights[MAX_NUM_LIGHTS];
  Light point_lights[MAX_NUM_LIGHTS];
} lights;

// TODO
layout (constant_id = 0) const uint num_directional_lights = 1U;
layout (constant_id = 1) const uint num_point_lights = 1U;

layout (location = 0) out vec4 out_color;

#include "../core/light.glsl"

void main()
{
  const float r1 = 10.f;
  const float r2 = 15.f;
  const float r = length(frag_position);
  const float alpha = 0.5f * (1.f - smoothstep(r1, r2, r));
 
  // Directional light
  const vec3 frag_normal = vec3(0.f, 0.f, 1.f);
  vec3 N = normalize(frag_normal);
  vec3 V = normalize(camera.eye - vec3(frag_position, 0.f));
   
  const vec3 diffuse_color = texture(tex, frag_tex_coord).rgb;
  const vec3 specular = vec3(0.1f, 0.1f, 0.1f);
  const float shininess = 1.f;

  vec3 total_color = vec3(0.f, 0.f, 0.f);
  for (int i = 0; i < num_directional_lights; i++)
  {
    vec3 light_color = compute_directional_light_color(lights.directional_lights[i], diffuse_color, specular, shininess, N, V);
    total_color += light_color;
  }
  
  for (int i = 0; i < num_point_lights; i++)
  {
    vec3 light_color = compute_point_light_color(lights.point_lights[i], diffuse_color, specular, shininess, vec3(frag_position, 0.f), N, V);
    total_color += light_color;
  }

  out_color = vec4(total_color, alpha);
}
