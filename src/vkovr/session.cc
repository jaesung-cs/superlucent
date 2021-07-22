#include <vkovr/session.h>

#include <iostream>

#include <OVR_CAPI.h>
#include <OVR_CAPI_Vk.h>

#include <vkovr/device.h>

namespace vkovr
{
Session::Session()
{
}

Session::~Session() = default;

Session createSession(const SessionCreateInfo& createInfo)
{
  // Initializes LibOVR, and the Rift
  ovrInitParams initParams = { ovrInit_RequestVersion | ovrInit_FocusAware, OVR_MINOR_VERSION, NULL, 0, 0 };
  auto result = ovr_Initialize(&initParams);
  if (!OVR_SUCCESS(result))
    throw std::runtime_error("Failed to initialize libOVR");

  // Populate session info
  Session session;
  result = ovr_Create(&session.session_, &session.luid_);
  if (!OVR_SUCCESS(result))
    throw std::runtime_error("Failed to create ovr session");
  session.hmdDesc_ = ovr_GetHmdDesc(session.session_);

  // FloorLevel will give tracking poses where the floor height is 0
  ovr_SetTrackingOriginType(session.session_, ovrTrackingOrigin_FloorLevel);

  return session;
}

std::vector<std::string> Session::getInstanceExtensions()
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

vk::PhysicalDevice Session::getPhysicalDevice(vk::Instance instance)
{
  VkPhysicalDevice physicalDevice;
  const auto result = ovr_GetSessionPhysicalDeviceVk(session_, luid_, instance, &physicalDevice);
  if (!OVR_SUCCESS(result))
    throw std::runtime_error("Failed to get physical device, calling ovr_GetSessionPhysicalDeviceVk()");
  return physicalDevice;
}

std::vector<std::string> Session::getDeviceExtensions()
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

Device Session::createDevice(const DeviceCreateInfo& createInfo)
{
  Device device;
  device.session_ = *this;
  device.device_ = createInfo.device;
  return device;
}

void Session::destroy()
{
  ovr_Shutdown();
}
}
