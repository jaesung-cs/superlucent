#ifndef SUPERLUCENT_ENGINE_PARTICLE_SIMULATION_H_
#define SUPERLUCENT_ENGINE_PARTICLE_SIMULATION_H_

#include <vector>

#include <superlucent/engine/data/particle.h>
#include <superlucent/engine/ubo/particle_simulation_params.h>

namespace supl
{
namespace engine
{
class ParticleSimulation
{
public:
  ParticleSimulation();

  ~ParticleSimulation();

  void UpdateSimulationParams(double dt, double animation_time);
  void Forward();

  const auto& Particles() const { return particles_; }

private:
  std::vector<Particle> particles_;
  ParticleSimulationParamsUbo params_{};
};
}
}

#endif // SUPERLUCENT_ENGINE_PARTICLE_SIMULATION_H_
