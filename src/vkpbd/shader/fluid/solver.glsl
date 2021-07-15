#ifndef VKPBD_FLUID_SOLVER_GLSL_
#define VKPBD_FLUID_SOLVER_GLSL_

struct SolverParticle
{
  float density;
  float lambda;
  vec2 pad0;

  vec3 dx;
  float pad1;
};

layout (binding = 4) buffer SolverSsbo
{
  SolverParticle particles[];
} solver;

#endif // VKPBD_FLUID_SOLVER_GLSL_
