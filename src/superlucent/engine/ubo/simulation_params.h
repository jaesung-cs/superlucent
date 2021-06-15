#ifndef SUPERLUCENT_ENGINE_UBO_SIMULATION_PARAMS_H_
#define SUPERLUCENT_ENGINE_UBO_SIMULATION_PARAMS_H_

namespace supl
{
namespace engine
{
struct SimulationParamsUbo
{
  alignas(16) float dt;
  int num_particles;
  float alpha; // compliance of the constraints
  float wall_offset; // wall x direction distance is added with this value

  float radius;
};
}
}

#endif // SUPERLUCENT_ENGINE_UBO_SIMULATION_PARAMS_H_
