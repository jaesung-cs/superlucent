#ifndef VKOVR_DEVICE_H_
#define VKOVR_DEVICE_H_

#include <vkovr/session.h>

namespace vkovr
{
class MirrorTexture;
class MirrorTextureCreateInfo;

enum class EyeType
{
  LeftEye,
  RightEye,
};

class Device
{
  friend class Session;

public:
  Device();
  ~Device();

  MirrorTexture createMirrorTexture(const MirrorTextureCreateInfo& createInfo);

  vk::Extent2D getFovTextureSize(EyeType eyeType);

  void destroy();
  void destroyMirrorTexture(MirrorTexture mirrorTexture);

private:
  Session session_;
  vk::Device device_;
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
