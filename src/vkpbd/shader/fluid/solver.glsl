#ifndef VKPBD_FLUID_SOLVER_GLSL_
#define VKPBD_FLUID_SOLVER_GLSL_

layout (binding = 4) buffer SolverSsbo
{
  int buf[];
} solver;

#endif // VKPBD_FLUID_SOLVER_GLSL_
