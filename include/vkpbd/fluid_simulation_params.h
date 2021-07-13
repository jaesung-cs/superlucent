#ifndef VKPBD_FLUID_SIMULATION_PARAMS_H_
#define VKPBD_FLUID_SIMULATION_PARAMS_H_

namespace vkpbd
{
struct FluidSimulationParams
{
  alignas(16) float dt;
  int num_particles;
  float alpha; // compliance of the constraints
  float wall_offset; // wall x direction distance is added with this value

  float radius;
};
}

#endif // VKPBD_FLUID_SIMULATION_PARAMS_H_
