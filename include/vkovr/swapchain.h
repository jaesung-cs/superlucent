#ifndef VKOVR_SWAPCHAIN_H_
#define VKOVR_SWAPCHAIN_H_

#include <vkovr/session.h>

#include <OVR_CAPI_Vk.h>

namespace vkovr
{
class Device;

class Swapchain
{
  friend class Device;

public:
  Swapchain();
  ~Swapchain();

  auto colorSwapchain() const { return colorSwapchain_; }
  auto depthSwapchain() const { return depthSwapchain_; };
  const auto& extent() const { return extent_; }

  void commit();

private:
  Session session_;
  vk::Extent2D extent_;
  ovrTextureSwapChain colorSwapchain_;
  ovrTextureSwapChain depthSwapchain_;
  std::vector<vk::Image> colorImages_;
  std::vector<vk::Image> depthImages_;
};

class SwapchainCreateInfo
{
public:
  SwapchainCreateInfo() = default;

public:
  vk::Extent2D extent;
};
}

#endif // VKOVR_SWAPCHAIN_H_
