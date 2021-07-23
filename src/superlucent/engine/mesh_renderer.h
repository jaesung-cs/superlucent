#ifndef SUPERLUCENT_ENGINE_MESH_RENDERER_H_
#define SUPERLUCENT_ENGINE_MESH_RENDERER_H_

#include <vulkan/vulkan.hpp>

#include <superlucent/engine/ubo/camera_ubo.h>
#include <superlucent/engine/ubo/light_ubo.h>

namespace supl
{
namespace engine
{
class Engine;
class Uniform;

class MeshRenderer
{
public:
  MeshRenderer() = delete;

  explicit MeshRenderer(Engine* engine, uint32_t width, uint32_t height);

  ~MeshRenderer();

  void Resize(uint32_t width, uint32_t height);

  void UpdateLights(const LightUbo& lights, int imageIndex);
  void UpdateCamera(const CameraUbo& camera, int imageIndex);

  void Begin(vk::CommandBuffer& commandBuffer, int imageIndex);
  void End(vk::CommandBuffer& commandBuffer);

private:
  Engine* const engine_;
  uint32_t width_ = 0;
  uint32_t height_ = 0;
};
}
}

#endif // SUPERLUCENT_ENGINE_MESH_RENDERER_H_
