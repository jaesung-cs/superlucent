#ifndef SUPERLUCENT_ENGINE_DATA_PARTICLE_H_
#define SUPERLUCENT_ENGINE_DATA_PARTICLE_H_

#include <glm/glm.hpp>

namespace supl
{
namespace engine
{
struct Particle
{
  alignas(16) glm::vec4 prev_position{ 0.f };
  alignas(16) glm::vec4 position{ 0.f };
  alignas(16) glm::vec4 velocity{ 0.f };
  alignas(16) glm::vec4 properties{ 0.f }; // radius, mass
  alignas(16) glm::vec4 external_force{ 0.f };
  alignas(16) glm::vec4 color{ 0.f };
};
}
}

#endif // SUPERLUCENT_ENGINE_DATA_PARTICLE_H_
