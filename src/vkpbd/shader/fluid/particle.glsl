#ifndef VKPBD_FLUID_PARTICLE_GLSL_
#define VKPBD_FLUID_PARTICLE_GLSL_

struct Particle
{
  vec4 position; // [px, py, pz, 0]
  vec4 velocity; // [vx, vy, vz, 0]
  vec4 properties; // [invMass, 0, 0, 0]
  vec4 external_force;
  vec4 color; // [r, g, b, 0]
};

layout(std140, binding = 0) buffer InParticleSsbo
{
  Particle in_particles[];
};

layout(std140, binding = 1) buffer OutParticleSsbo
{
  Particle out_particles[];
};

#endif // VKPBD_FLUID_PARTICLE_GLSL_
