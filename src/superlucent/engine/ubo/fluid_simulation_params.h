#ifndef SUPERLUCENT_ENGINE_UBO_FLUID_SIMULATION_PARAMS_H_
#define SUPERLUCENT_ENGINE_UBO_FLUID_SIMULATION_PARAMS_H_

namespace supl
{
namespace engine
{
struct FluidSimulationParamsUbo
{
  alignas(16) float dt;
  int num_particles;
  float epsilon; // relaxation parameter
  float wall_offset; // wall x direction distance is added with this value

  float h; // in density estimation/gradient calculation kernel computation
  float radius; // particle radius for particle rendering
  float rest_density;
  float c; // vorticity confinement constant, typically 0.01 according to the paper

  float k; // used in smoothing kernel, typically 0.1
  int n; // used in smoothing kernel, typically 4
  int max_num_neighbors;
};
}
}

#endif // SUPERLUCENT_ENGINE_UBO_FLUID_SIMULATION_PARAMS_H_
