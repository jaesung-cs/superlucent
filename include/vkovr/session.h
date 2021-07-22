#ifndef VKOVR_SESSION_H_
#define VKOVR_SESSION_H_

#include <vector>
#include <string>
#include <memory>

#include <vulkan/vulkan.hpp>

#include <OVR_CAPI_Vk.h>

namespace vkovr
{
class Device;
class DeviceCreateInfo;

class SessionCreateInfo;

class Session
{
  friend Session createSession(const SessionCreateInfo& createInfo);

public:
  Session();
  ~Session();

  auto session() const { return session_; }
  const auto& getOculusProperties() const { return hmdDesc_; }

  std::vector<std::string> getInstanceExtensions();
  vk::PhysicalDevice getPhysicalDevice(vk::Instance instance);
  std::vector<std::string> getDeviceExtensions();

  Device createDevice(const DeviceCreateInfo& createInfo);

  void destroy();

private:
  ovrSession session_ = nullptr;
  ovrGraphicsLuid luid_{};
  ovrHmdDesc hmdDesc_{};
};

class SessionCreateInfo
{
public:
  SessionCreateInfo() = default;

public:
};

Session createSession(const SessionCreateInfo& createInfo);
}

#endif // VKOVR_SESSION_H_
