#ifndef SUPERLUCENT_VKSYNC_SYNC_H_
#define SUPERLUCENT_VKSYNC_SYNC_H_

#include <vulkan/vulkan.hpp>

namespace supl
{
namespace vksync
{
class Sync
{
public:
  Sync() = delete;

  explicit Sync(vk::Device device);

  auto Device() const { return device_; }

private:
  vk::Device device_;
};
}
}

#endif // SUPERLUCENT_VKSYNC_SYNC_H_
