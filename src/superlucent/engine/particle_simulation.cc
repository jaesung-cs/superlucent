#include <superlucent/engine/particle_simulation.h>

#include <superlucent/utils/rng.h>

namespace supl
{
namespace engine
{
ParticleSimulation::ParticleSimulation()
{
  // Prticles
  params_.radius = 0.03f;

  constexpr int cell_count = 40;
  const float radius = params_.radius;
  constexpr float density = 1000.f; // water
  const float mass = radius * radius * radius * density;
  constexpr glm::vec2 wall_distance = glm::vec2(3.f, 1.5f);
  const glm::vec3 particle_offset = glm::vec3(-wall_distance + glm::vec2(radius * 1.1f), radius * 1.1f);
  const glm::vec3 particle_stride = glm::vec3(radius * 2.2f);

  utils::Rng rng;
  constexpr float noise_range = 1e-2f;
  const auto noise = [&rng, noise_range]() { return rng.Uniform(-noise_range, noise_range); };

  glm::vec3 gravity = glm::vec3(0.f, 0.f, -9.8f);
  for (int i = 0; i < cell_count; i++)
  {
    for (int j = 0; j < cell_count; j++)
    {
      for (int k = 0; k < cell_count; k++)
      {
        glm::vec4 position{
          particle_offset.x + particle_stride.x * i + noise(),
          particle_offset.y + particle_stride.y * j + noise(),
          particle_offset.z + particle_stride.z * k + noise(),
          0.f
        };
        glm::vec4 velocity{ 0.f };
        glm::vec4 properties{ mass, 0.f, 0.f, 0.f };
        glm::vec4 external_force{
          gravity.x * mass,
          gravity.y * mass,
          gravity.z * mass,
          0.f
        };
        glm::vec4 color{ 0.5f, 0.5f, 0.5f, 0.f };

        // Struct initialization
        particles_.push_back({ position, position, velocity, properties, external_force, color });
      }
    }
  }
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
