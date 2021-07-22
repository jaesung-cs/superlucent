#include <vkovr/device.h>

#include <vkovr/mirror_texture.h>
#include <vkovr/swapchain.h>

#include <OVR_CAPI_Vk.h>

#include <glm/gtc/quaternion.hpp>

namespace vkovr
{
namespace
{
auto toOvrEyeType(EyeType eyeType)
{
  switch (eyeType)
  {
  case EyeType::LeftEye:
    return ovrEyeType::ovrEye_Left;
  case EyeType::RightEye:
    return ovrEyeType::ovrEye_Right;
  }
  
  throw std::runtime_error("Unknown eye type");
}

auto toMat4(const ovrPosef& pose)
{
  glm::quat q{
    pose.Orientation.w,
    pose.Orientation.x,
    pose.Orientation.y,
    pose.Orientation.z,
  };

  glm::vec3 p{
    pose.Position.x,
    pose.Position.y,
    pose.Position.z,
  };

  glm::mat4 mat{ q };
  mat[3] = { p, 1.f };

  return mat;
}

auto toMat4(const ovrMatrix4f& m)
{
  glm::mat4 mat;
  for (int r = 0; r < 4; r++)
  {
    for (int c = 0; c < 4; c++)
      mat[c][r] = m.M[r][c];
  }

  return mat;
}

auto toOvrMat4(const glm::mat4& m)
{
  ovrMatrix4f mat;
  for (int r = 0; r < 4; r++)
  {
    for (int c = 0; c < 4; c++)
      mat.M[r][c] = m[c][r];
  }

  return mat;
}

auto toOvrPose(const glm::mat4& m)
{
  glm::quat q{ glm::mat3{m} };
  glm::vec3 p = m[3];

  ovrPosef pose;
  pose.Orientation.w = q.w;
  pose.Orientation.x = q.x;
  pose.Orientation.y = q.y;
  pose.Orientation.z = q.z;
  pose.Position.x = p.x;
  pose.Position.y = p.y;
  pose.Position.z = p.z;
  return pose;
}
}

Device::Device() = default;

Device::~Device() = default;

void Device::synchronizeQueue(vk::Queue queue)
{
  // Let the compositor know which queue to synchronize with
  ovr_SetSynchronizationQueueVk(session_, queue);
}

ovrSessionStatus Device::getStatus()
{
  ovrSessionStatus sessionStatus;
  ovr_GetSessionStatus(session_, &sessionStatus);
  return sessionStatus;
}

std::vector<glm::mat4> Device::getEyePoses()
{
  ovrEyeRenderDesc eyeRenderProperties[] = {
    getEyeRenderProperties(EyeType::LeftEye),
    getEyeRenderProperties(EyeType::RightEye),
  };
  ovrPosef hmdToEyePoses[] = {
    eyeRenderProperties[0].HmdToEyePose,
    eyeRenderProperties[1].HmdToEyePose,
  };
  ovrPosef eyeRenderPoses[2];

  ovr_GetEyePoses(session_, 0, ovrTrue, hmdToEyePoses, eyeRenderPoses, &sensorSampleTime_);

  return {
    toMat4(eyeRenderPoses[0]),
    toMat4(eyeRenderPoses[1]),
  };
}

glm::mat4 Device::getEyeProjection(EyeType eyeType, float near, float far)
{
  const auto eye = toOvrEyeType(eyeType);

  // Use default fov
  const auto& ovrProperties = session_.getOculusProperties();
  auto projection = toMat4(ovrMatrix4f_Projection(ovrProperties.DefaultEyeFov[eye], near, far, ovrProjectionModifier::ovrProjection_None));

  // To Vulkan coordinates
  auto yFlip = glm::mat4{ 1.f };
  yFlip[1][1] = -1.f;
  projection = yFlip * projection;

  // Update timewarp projection
  posTimewarpProjectionDescription_ = ovrTimewarpProjectionDesc_FromProjection(toOvrMat4(projection), ovrProjectionModifier::ovrProjection_None);

  return projection;
}

ovrResult Device::submitFrame(const FrameSubmitInfo& submitInfo)
{
  // Submit rendered eyes as an EyeFovDepth layer
  ovrLayerEyeFovDepth ld = {};
  ld.Header.Type = ovrLayerType_EyeFovDepth;
  ld.Header.Flags = 0;
  ld.ProjectionDesc = posTimewarpProjectionDescription_;
  ld.SensorSampleTime = sensorSampleTime_;

  const auto& oculusProperties = session_.getOculusProperties();
  for (int i = 0; i < 2; i++)
  {
    const auto& extent = submitInfo.swapchains[i].extent();
    ovrRecti viewport;
    viewport.Pos = { 0, 0 };
    viewport.Size.w = extent.width;
    viewport.Size.h = extent.height;

    ld.ColorTexture[i] = submitInfo.swapchains[i].colorSwapchain();
    ld.DepthTexture[i] = submitInfo.swapchains[i].depthSwapchain();
    ld.Viewport[i] = viewport;
    ld.Fov[i] = oculusProperties.DefaultEyeFov[i];
    ld.RenderPose[i] = toOvrPose(submitInfo.eyePoses[i]);
  }

  // TODO: return if device is lost
  ovrLayerHeader* layers = &ld.Header;
  return ovr_SubmitFrame(session_, 0, nullptr, &layers, 1);
}

vk::Extent2D Device::getFovTextureSize(EyeType eyeType)
{
  const auto ovrEye = toOvrEyeType(eyeType);

  // Use default fov
  const auto& ovrProperties = session_.getOculusProperties();
  const auto extent = ovr_GetFovTextureSize(session_, ovrEye, ovrProperties.DefaultEyeFov[ovrEye], 1.f);
  return vk::Extent2D(extent.w, extent.h);
}

ovrEyeRenderDesc Device::getEyeRenderProperties(EyeType eyeType)
{
  ovrEyeType eye = toOvrEyeType(eyeType);

  // Use default fov
  const auto& ovrProperties = session_.getOculusProperties();
  return ovr_GetRenderDesc(session_, eye, ovrProperties.DefaultEyeFov[eye]);
}

MirrorTexture Device::createMirrorTexture(const MirrorTextureCreateInfo& createInfo)
{
  ovrMirrorTexture mirrorTexture;
  ovrMirrorTextureDesc mirrorDesc = {};
  mirrorDesc.Format = OVR_FORMAT_B8G8R8A8_UNORM_SRGB;
  mirrorDesc.Width = createInfo.extent.width;
  mirrorDesc.Height = createInfo.extent.height;
  auto result = ovr_CreateMirrorTextureWithOptionsVk(session_, device_, &mirrorDesc, &mirrorTexture);

  VkImage image;
  result = ovr_GetMirrorTextureBufferVk(session_, mirrorTexture, &image);

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

Swapchain Device::createSwapchain(const SwapchainCreateInfo& createInfo)
{
  Swapchain swapchain;
  swapchain.session_ = session_;
  swapchain.extent_ = createInfo.extent;

  // Depth
  ovrTextureSwapChainDesc depthDesc = {};
  depthDesc.Type = ovrTexture_2D;
  depthDesc.ArraySize = 1;
  depthDesc.Format = OVR_FORMAT_D32_FLOAT;
  depthDesc.Width = createInfo.extent.width;
  depthDesc.Height = createInfo.extent.height;
  depthDesc.MipLevels = 1;
  depthDesc.SampleCount = 1;
  depthDesc.MiscFlags = ovrTextureMisc_DX_Typeless;
  depthDesc.BindFlags = ovrTextureBind_DX_DepthStencil;
  depthDesc.StaticImage = ovrFalse;
  auto result = ovr_CreateTextureSwapChainVk(session_, device_, &depthDesc, &swapchain.depthSwapchain_);
  if (!OVR_SUCCESS(result))
    throw std::runtime_error("Failed to create OVR depth swapchain, calling ovr_CreateTextureSwapChainVk()");

  // Color
  ovrTextureSwapChainDesc colorDesc = {};
  colorDesc.Type = ovrTexture_2D;
  colorDesc.ArraySize = 1;
  colorDesc.Format = OVR_FORMAT_B8G8R8A8_UNORM_SRGB;
  colorDesc.Width = createInfo.extent.width;
  colorDesc.Height = createInfo.extent.height;
  colorDesc.MipLevels = 1;
  colorDesc.SampleCount = 1;
  colorDesc.MiscFlags = ovrTextureMisc_DX_Typeless;
  colorDesc.BindFlags = ovrTextureBind_DX_RenderTarget;
  colorDesc.StaticImage = ovrFalse;
  result = ovr_CreateTextureSwapChainVk(session_, device_, &colorDesc, &swapchain.colorSwapchain_);
  if (!OVR_SUCCESS(result))
    throw std::runtime_error("Failed to create OVR texture swapchain, calling ovr_CreateTextureSwapChainVk()");

  int textureCount = 0;
  result = ovr_GetTextureSwapChainLength(session_, swapchain.colorSwapchain_, &textureCount);
  if (!OVR_SUCCESS(result))
    throw std::runtime_error("Failed to get texture swapchain length, calling ovr_GetTextureSwapChainLength()");

  int depthCount = 0;
  result = ovr_GetTextureSwapChainLength(session_, swapchain.depthSwapchain_, &depthCount);
  if (!OVR_SUCCESS(result))
    throw std::runtime_error("Failed to get depth swapchain length, calling ovr_GetTextureSwapChainLength()");

  for (int i = 0; i < textureCount; ++i)
  {
    VkImage colorImage;
    result = ovr_GetTextureSwapChainBufferVk(session_, swapchain.colorSwapchain_, i, &colorImage);
    if (!OVR_SUCCESS(result))
      throw std::runtime_error("Failed to get texture swapchain buffer, calling ovr_GetTextureSwapChainBufferVk()");

    VkImage depthImage;
    result = ovr_GetTextureSwapChainBufferVk(session_, swapchain.depthSwapchain_, i, &depthImage);
    if (!OVR_SUCCESS(result))
      throw std::runtime_error("Failed to get depth swapchain buffer, calling ovr_GetTextureSwapChainBufferVk()");

    swapchain.colorImages_.push_back(colorImage);
    swapchain.depthImages_.push_back(depthImage);
  }

  return swapchain;
}

void Device::destroy()
{
}

void Device::destroyMirrorTexture(MirrorTexture mirrorTexture)
{
  ovr_DestroyMirrorTexture(session_, mirrorTexture.mirrorTexture());
}

void Device::destroySwapchain(Swapchain swapchain)
{
  ovr_DestroyTextureSwapChain(session_, swapchain.colorSwapchain_);
  ovr_DestroyTextureSwapChain(session_, swapchain.depthSwapchain_);
}
}
