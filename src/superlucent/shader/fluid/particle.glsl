struct Particle
{
  vec4 prev_position;
  vec4 position;
  vec4 velocity;
  vec4 properties;
  vec4 external_force;
  vec4 color;
};

layout (binding = 1) buffer ParticleSsbo
{
  Particle particles[];
};
