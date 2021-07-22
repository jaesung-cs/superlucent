#include <vkovr/swapchain.h>

namespace vkovr
{
Swapchain::Swapchain() = default;

Swapchain::~Swapchain() = default;

void Swapchain::commit()
{
  ovr_CommitTextureSwapChain(session_, colorSwapchain_);
  ovr_CommitTextureSwapChain(session_, depthSwapchain_);
}
}
