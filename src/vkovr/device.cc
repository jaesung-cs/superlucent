#include <vkovr/device.h>

#include <vkovr/mirror_texture.h>

#include <OVR_CAPI_Vk.h>

namespace vkovr
{
Device::Device() = default;

Device::~Device() = default;

MirrorTexture Device::createMirrorTexture(const MirrorTextureCreateInfo& createInfo)
{
  ovrMirrorTexture mirrorTexture;
  ovrMirrorTextureDesc mirrorDesc = {};
  mirrorDesc.Format = OVR_FORMAT_B8G8R8A8_UNORM_SRGB;
  mirrorDesc.Width = createInfo.extent.width;
  mirrorDesc.Height = createInfo.extent.height;
  auto result = ovr_CreateMirrorTextureWithOptionsVk(session_.session(), device_, &mirrorDesc, &mirrorTexture);

  VkImage image;
  result = ovr_GetMirrorTextureBufferVk(session_.session(), mirrorTexture, &image);

  // Switch the mirror buffer from UNDEFINED -> TRANSFER_SRC_OPTIMAL
  vk::ImageMemoryBarrier imageBarrier;
  imageBarrier
    .setOldLayout(vk::ImageLayout::eUndefined)
    .setNewLayout(vk::ImageLayout::eTransferSrcOptimal)
    .setSrcAccessMask({})
    .setDstAccessMask(vk::AccessFlagBits::eTransferRead)
    .setImage(image)
    .setSubresourceRange({ vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1 });

  createInfo.commandBuffer.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eTransfer, {},
    {}, {}, imageBarrier);

  MirrorTexture texture;
  texture.mirrorTexture_ = mirrorTexture;
  texture.extent_ = createInfo.extent;
  texture.image_ = image;
  return texture;
}

vk::Extent2D Device::getFovTextureSize(EyeType eyeType)
{
  ovrEyeType ovrEye;
  switch (eyeType)
  {
  case EyeType::LeftEye:
    ovrEye = ovrEyeType::ovrEye_Left;
    break;
  case EyeType::RightEye:
    ovrEye = ovrEyeType::ovrEye_Right;
    break;
  }

  // Use default fov
  const auto& ovrProperties = session_.getOculusProperties();
  const auto extent = ovr_GetFovTextureSize(session_.session(), ovrEye, ovrProperties.DefaultEyeFov[ovrEye], 1.f);
  return { extent.w, extent.h };
}

void Device::destroy()
{
}

void Device::destroyMirrorTexture(MirrorTexture mirrorTexture)
{
  ovr_DestroyMirrorTexture(session_.session(), mirrorTexture.mirrorTexture());
}
}
