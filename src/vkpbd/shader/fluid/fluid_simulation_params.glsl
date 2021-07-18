#ifndef VKPBD_FLUID_FLUID_SIMULATION_PARAMS_GLSL_
#define VKPBD_FLUID_FLUID_SIMULATION_PARAMS_GLSL_

layout (binding = 6) uniform FluidSimulationParamsUbo
{
	float dt;
	int num_particles;
  float alpha;
  float wall_offset;

  float radius;
  int max_num_neighbors;
  vec2 kernel_constants; // [k, l]
  // where k = 8 / (pi * h3)
  // and   l = 48 / (pi * h3)

  float rest_density;
  float viscosity;
} params;

#endif // VKPBD_FLUID_FLUID_SIMULATION_PARAMS_GLSL_
