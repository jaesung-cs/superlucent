#ifndef SUPERLUCENT_ENGINE_MESH_RENDERER_H_
#define SUPERLUCENT_ENGINE_MESH_RENDERER_H_

#include <vulkan/vulkan.hpp>

#include <superlucent/engine/ubo/camera_ubo.h>
#include <superlucent/engine/ubo/light_ubo.h>

namespace supl
{
namespace engine
{
class MeshRendererCreateInfo;

class MeshRenderer
{
  friend MeshRenderer createMeshRenderer(const MeshRendererCreateInfo& createInfo);

public:
  MeshRenderer();
  ~MeshRenderer();

  void resize(uint32_t width, uint32_t height);

  void updateLights(const LightUbo& lights, int imageIndex);
  void updateCamera(const CameraUbo& camera, int imageIndex);

  void begin(vk::CommandBuffer& commandBuffer, int imageIndex);
  void end(vk::CommandBuffer& commandBuffer);

  void destroy();

private:
  uint32_t width_ = 0;
  uint32_t height_ = 0;

  vk::Device device_;

  // Pipeline
  vk::RenderPass renderPass_;
  vk::PipelineLayout pipelineLayout_;
  vk::Pipeline meshPipeline_;

  // Descriptor set
  // Binding 0: camera ubo
  // Binding 1: light ubo
  vk::DescriptorSetLayout descriptorSetLayout_;
  std::vector<vk::DescriptorSet> descriptorSets_;
};

class MeshRendererCreateInfo
{
public:
  MeshRendererCreateInfo() = default;

public:
  vk::Device device;
  vk::DescriptorPool descriptorPool;

  uint32_t width;
  uint32_t height;
  uint32_t imageCount;
  vk::Format format;
  vk::ImageLayout finalLayout;
};

MeshRenderer createMeshRenderer(const MeshRendererCreateInfo& createInfo);
}
}

#endif // SUPERLUCENT_ENGINE_MESH_RENDERER_H_
