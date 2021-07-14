#ifndef VKPBD_FLUID_SIMULATION_PARAMS_H_
#define VKPBD_FLUID_SIMULATION_PARAMS_H_

namespace vkpbd
{
struct FluidSimulationParams
{
  alignas(16) float dt = 0.f;
  int num_particles = 0;
  float alpha = 0.f; // compliance of the constraints
  float wall_offset = 0.f; // wall x direction distance is added with this value

  float radius = 0.f;
  int max_num_neighbors = 0;
};
}

#endif // VKPBD_FLUID_SIMULATION_PARAMS_H_
