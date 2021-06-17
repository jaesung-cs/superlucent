struct Particle
{
  vec4 prev_position;
  vec4 position; // [px, py, pz, 0]
  vec4 velocity; // [vx, vy, vz, 0]
  vec4 properties; // [mass, 0, 0, 0]
  vec4 external_force;
  vec4 color; // [r, g, b, 0]
};

layout (binding = 1) buffer ParticleSsbo
{
  Particle particles[];
};
