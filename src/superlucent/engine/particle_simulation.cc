#include <superlucent/engine/particle_simulation.h>

namespace supl
{
namespace engine
{
ParticleSimulation::ParticleSimulation()
{
}

ParticleSimulation::~ParticleSimulation()
{
}

void ParticleSimulation::UpdateSimulationParams(double dt, double animation_time)
{
  params_.dt = dt;

  // TODO: set params_.wall_offset
}

void ParticleSimulation::Forward()
{
}
}
}
