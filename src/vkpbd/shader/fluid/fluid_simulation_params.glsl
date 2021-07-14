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
} params;

#endif // VKPBD_FLUID_FLUID_SIMULATION_PARAMS_GLSL_
