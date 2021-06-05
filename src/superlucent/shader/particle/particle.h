struct Particle
{
  vec4 prev_position;
  vec4 position; // [px, py, pz, 0]
  vec4 velocity; // [vx, vy, vz, 0]
  vec4 properties; // [radius, mass, 0, 0]
  vec4 external_force;
};

layout(std140, binding = 0) buffer ParticleSsbo
{
  Particle particles[];
};
