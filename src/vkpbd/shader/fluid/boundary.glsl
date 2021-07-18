#ifndef VKPBD_FLUID_BOUNDARY_GLSL_
#define VKPBD_FLUID_BOUNDARY_GLSL_

layout(binding = 3) buffer BoundaryPsiSsbo
{
  float volume[];
} boundary;

#endif // VKPBD_FLUID_BOUNDARY_GLSL_
