#ifndef VKOVR_DEVICE_H_
#define VKOVR_DEVICE_H_

#include <vkovr/session.h>
#include <vkovr/swapchain.h>

#include <glm/glm.hpp>

namespace vkovr
{
class MirrorTexture;
class MirrorTextureCreateInfo;

enum class EyeType
{
  LeftEye,
  RightEye,
};

struct FrameSubmitInfo
{
  std::vector<Swapchain> swapchains;
  std::vector<glm::mat4> eyePoses;
};

class Device
{
  friend class Session;

public:
  Device();
  ~Device();

  void synchronizeQueue(vk::Queue queue);

  vk::Extent2D getFovTextureSize(EyeType eyeType);
  ovrEyeRenderDesc getEyeRenderProperties(EyeType eyeType);
  ovrSessionStatus getStatus();
  std::vector<glm::mat4> getEyePoses();
  glm::mat4 getEyeProjection(EyeType eyeType, float near, float far);

  ovrResult submitFrame(const FrameSubmitInfo& submitInfo);

  MirrorTexture createMirrorTexture(const MirrorTextureCreateInfo& createInfo);
  Swapchain createSwapchain(const SwapchainCreateInfo& createInfo);

  void destroy();
  void destroyMirrorTexture(MirrorTexture mirrorTexture);
  void destroySwapchain(Swapchain swapchain);

private:
  Session session_;
  vk::Device device_;

  // Store latest status
  double sensorSampleTime_ = 0.;
  ovrTimewarpProjectionDesc posTimewarpProjectionDescription_{};
};

class DeviceCreateInfo
{
public:
  DeviceCreateInfo() = default;

public:
  vk::Device device;
};
}

#endif // VKOVR_DEVICE_H_
