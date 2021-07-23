#include <superlucent/engine/mesh_renderer.h>

#include <superlucent/engine/engine.h>

namespace supl
{
namespace engine
{
MeshRenderer::MeshRenderer(Engine* engine, uint32_t width, uint32_t height)
  : engine_{ engine }
  , width_{ width }
  , height_{ height }
{
}

MeshRenderer::~MeshRenderer()
{
}

void MeshRenderer::Resize(uint32_t width, uint32_t height)
{
  width_ = width;
  height_ = height;

  // TODO: Recreate swapchains and framebuffers
}

void MeshRenderer::UpdateLights(const LightUbo& lights, int imageIndex)
{
  // TODO
}

void MeshRenderer::UpdateCamera(const CameraUbo& camera, int imageIndex)
{
  // TODO
}

void MeshRenderer::Begin(vk::CommandBuffer& commandBuffer, int imageIndex)
{
  // TODO: Begin render pass
}

void MeshRenderer::End(vk::CommandBuffer& commandBuffer)
{
  commandBuffer.endRenderPass();
}
}
}
