#ifndef VKOVR_VKOVR_HPP_
#define VKOVR_VKOVR_HPP_

#include <vector>
#include <string>
#include <memory>

#include <vulkan/vulkan.hpp>

struct GLFWwindow;

namespace vkovr
{
class OculusVrCreateInfo;

class OculusVr
{
  friend OculusVr createOclulusVr(const OculusVrCreateInfo& createInfo);

public:
  OculusVr();
  ~OculusVr();

  bool beginSession();

  std::vector<std::string> getInstanceExtensions();
  vk::PhysicalDevice getPhysicalDevice(vk::Instance instance);
  std::vector<std::string> getDeviceExtensions();

  void destroy();

private:
  class Impl;

  // shared_ptr for enabling having copies
  std::shared_ptr<Impl> impl_;
};

class OculusVrCreateInfo
{
public:
  OculusVrCreateInfo() = default;

public:
  GLFWwindow* window = nullptr;
};

OculusVr createOclulusVr(const OculusVrCreateInfo& createInfo);
}

#endif // VKOVR_VKOVR_HPP_
