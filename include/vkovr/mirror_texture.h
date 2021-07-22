#ifndef VKOVR_MIRROR_TEXTURE_H_
#define VKOVR_MIRROR_TEXTURE_H_

#include <vulkan/vulkan.hpp>

#include <OVR_CAPI.h>

namespace vkovr
{
class Device;

class MirrorTexture
{
  friend class Device;

public:
  MirrorTexture();
  ~MirrorTexture();

  auto mirrorTexture() const { return mirrorTexture_; }

private:
  ovrMirrorTexture mirrorTexture_ = nullptr;
  vk::Extent2D extent_;
  vk::Image image_;
};

class MirrorTextureCreateInfo
{
public:
  MirrorTextureCreateInfo() = default;

public:
  vk::Extent2D extent;
  vk::CommandBuffer commandBuffer; // For changing image layout
};
}

#endif // VKOVR_MIRROR_TEXTURE_H_
