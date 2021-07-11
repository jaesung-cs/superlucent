#ifndef VKPBD_PARTICLE_H_
#define VKPBD_PARTICLE_H_

#include <glm/glm.hpp>

namespace vkpbd
{
struct Particle
{
  alignas(16) glm::vec4 position{ 0.f };
  alignas(16) glm::vec4 velocity{ 0.f };
  alignas(16) glm::vec4 properties{ 0.f }; // mass
  alignas(16) glm::vec4 external_force{ 0.f };
  alignas(16) glm::vec4 color{ 0.f };
};
}

#endif // VKPBD_PARTICLE_H_
