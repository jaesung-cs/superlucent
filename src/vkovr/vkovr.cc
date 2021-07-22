#include <vkovr/vkovr.hpp>

#include <stdexcept>
#include <iostream>

#include <windows.h>

#include <OVR_CAPI_Vk.h>

#include <vulkan/vulkan.hpp>

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <GLFW/glfw3native.h>

namespace vkovr
{
namespace
{
}

// OculusVr Impl
class OculusVr::Impl
{
  friend OculusVr createOclulusVr(const OculusVrCreateInfo& createInfo);

public:
  Impl()
  {
  }

  ~Impl()
  {
  }

  std::vector<std::string> getInstanceExtensions()
  {
    char extensionNames[4096];
    uint32_t extensionNamesSize = sizeof(extensionNames);

    const auto result = ovr_GetInstanceExtensionsVk(luid_, extensionNames, &extensionNamesSize);
    if (!OVR_SUCCESS(result))
      return {};

    std::vector<std::string> extensions = { "" };
    for (int i = 0; extensionNames[i] != 0; i++)
    {
      if (extensionNames[i] == ' ')
        extensions.emplace_back();
      else
        extensions.back().push_back(extensionNames[i]);
    }

    return extensions;
  }

  vk::PhysicalDevice getPhysicalDevice(vk::Instance instance)
  {
    VkPhysicalDevice physicalDevice;
    const auto result = ovr_GetSessionPhysicalDeviceVk(session_, luid_, instance, &physicalDevice);
    if (!OVR_SUCCESS(result))
      throw std::runtime_error("Failed to get physical device, calling ovr_GetSessionPhysicalDeviceVk()");
    return physicalDevice;
  }

  std::vector<std::string> getDeviceExtensions()
  {
    char extensionNames[4096];
    uint32_t extensionNamesSize = sizeof(extensionNames);

    auto ret = ovr_GetDeviceExtensionsVk(luid_, extensionNames, &extensionNamesSize);
    if (!OVR_SUCCESS(ret))
      throw std::runtime_error("Failed to get device extensions, calling ovr_GetDeviceExtensionsVk()");

    std::vector<std::string> extensions = { "" };
    for (int i = 0; extensionNames[i] != 0; i++)
    {
      if (extensionNames[i] == ' ')
        extensions.emplace_back();
      else
        extensions.back().push_back(extensionNames[i]);
    }

    return extensions;
  }

  bool beginSession()
  {
    const auto result = ovr_Create(&session_, &luid_);
    if (!OVR_SUCCESS(result))
    {
      std::cerr << "Failed to create ovr" << std::endl;
      return false;
    }

    hmdDesc_ = ovr_GetHmdDesc(session_);

    return true;
  }

  void destroy()
  {
    ovr_Shutdown();
  }

private:
  ovrSession session_ = nullptr;
  ovrGraphicsLuid luid_{};
  ovrHmdDesc hmdDesc_{};
};

OculusVr::OculusVr()
  : impl_{ std::make_unique<Impl>() }
{
}

OculusVr::~OculusVr() = default;

OculusVr createOclulusVr(const OculusVrCreateInfo& createInfo)
{
  // Initializes LibOVR, and the Rift
  ovrInitParams initParams = { ovrInit_RequestVersion | ovrInit_FocusAware, OVR_MINOR_VERSION, NULL, 0, 0 };
  ovrResult result = ovr_Initialize(&initParams);
  if (!OVR_SUCCESS(result))
    throw std::runtime_error("Failed to initialize libOVR");

  // Populate ovr info
  OculusVr oculusVr;
  return oculusVr;
}

bool OculusVr::beginSession()
{
  return impl_->beginSession();
}

std::vector<std::string> OculusVr::getInstanceExtensions()
{
  return impl_->getInstanceExtensions();
}

vk::PhysicalDevice OculusVr::getPhysicalDevice(vk::Instance instance)
{
  return impl_->getPhysicalDevice(instance);
}

std::vector<std::string> OculusVr::getDeviceExtensions()
{
  return impl_->getDeviceExtensions();
}

void OculusVr::destroy()
{
  impl_->destroy();
}
}
