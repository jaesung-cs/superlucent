#ifndef SUPERLUCENT_ENGINE_UBO_CAMERA_UBO_H_
#define SUPERLUCENT_ENGINE_UBO_CAMERA_UBO_H_

#include <glm/glm.hpp>

namespace supl
{
namespace engine
{
struct CameraUbo
{
  alignas(16) glm::mat4 projection;
  alignas(16) glm::mat4 view;
  alignas(16) glm::vec3 eye;
};
}
}

#endif // SUPERLUCENT_ENGINE_UBO_CAMERA_UBO_H_
