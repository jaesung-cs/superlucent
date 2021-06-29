#ifndef SUPERLUCENT_ENGINE_PARTICLE_RENDERER_H_
#define SUPERLUCENT_ENGINE_PARTICLE_RENDERER_H_

#include <vulkan/vulkan.hpp>

#include <superlucent/engine/ubo/light_ubo.h>
#include <superlucent/engine/ubo/camera_ubo.h>

namespace supl
{
namespace engine
{
class Engine;
class Uniform;

class ParticleRenderer
{
public:
  ParticleRenderer() = delete;

  explicit ParticleRenderer(Engine* engine, uint32_t width, uint32_t height);

  ~ParticleRenderer();

  void UpdateLights(const LightUbo& lights, int image_index);
  void UpdateCamera(const CameraUbo& camera, int image_index);

  void Begin(vk::CommandBuffer& command_buffer, int image_index);
  void RecordParticleRenderCommands(vk::CommandBuffer& command_buffer, vk::Buffer particle_buffer, uint32_t num_particles, float radius);
  void RecordFloorRenderCommands(vk::CommandBuffer& command_buffer);
  void End(vk::CommandBuffer& command_buffer);

private:
  void CreateSampler();
  void DestroySampler();

  void CreateFramebuffer();
  void DestroyFramebuffer();

  void CreateGraphicsPipelines();
  void DestroyGraphicsPipelines();

  void PrepareResources();
  void DestroyResources();

  vk::Pipeline CreateGraphicsPipeline(vk::GraphicsPipelineCreateInfo& create_info);

  Engine* const engine_;
  uint32_t width_ = 0;
  uint32_t height_ = 0;

  // Sampler
  const uint32_t mipmap_level_ = 3u;
  vk::Sampler sampler_;

  // Resources
  struct VertexBuffer
  {
    vk::Buffer buffer;
    vk::DeviceSize index_offset;
    uint32_t num_indices;
  };
  VertexBuffer floor_buffer_;
  VertexBuffer cells_buffer_;

  struct Texture
  {
    vk::Image image;
    vk::ImageView image_view;
  };
  Texture floor_texture_;

  // Pipeline
  vk::RenderPass render_pass_;
  std::vector<vk::Framebuffer> swapchain_framebuffers_;

  vk::DescriptorSetLayout descriptor_set_layout_;
  vk::PipelineLayout pipeline_layout_;
  vk::PipelineCache pipeline_cache_;
  vk::Pipeline floor_pipeline_;
  vk::Pipeline cell_sphere_pipeline_;

  // Descriptor set
  std::vector<vk::DescriptorSet> descriptor_sets_;

  // Uniform buffer
  std::vector<Uniform> camera_ubos_;
  std::vector<Uniform> light_ubos_;
};
}
}

#endif // SUPERLUCENT_ENGINE_PARTICLE_RENDERER_H_
