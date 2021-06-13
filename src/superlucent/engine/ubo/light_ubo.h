#ifndef SUPERLUCENT_ENGINE_UBO_LIGHT_UBO_H_
#define SUPERLUCENT_ENGINE_UBO_LIGHT_UBO_H_

#include <glm/glm.hpp>

namespace supl
{
namespace engine
{
struct LightUbo
{
  struct Light
  {
    alignas(16) glm::vec3 position;
    alignas(16) glm::vec3 ambient;
    alignas(16) glm::vec3 diffuse;
    alignas(16) glm::vec3 specular;
  };

  static constexpr int max_num_lights = 8;
  Light directional_lights[max_num_lights];
  Light point_lights[max_num_lights];
};
}
}

#endif // SUPERLUCENT_ENGINE_UBO_LIGHT_UBO_H_
