#ifndef SUPERLUCENT_VKSYNC_BUFFER_H_
#define SUPERLUCENT_VKSYNC_BUFFER_H_

#include <vulkan/vulkan.hpp>

namespace supl
{
namespace vksync
{
template <typename T>
class Buffer
{
public:
  Buffer() = delete;

  Buffer(Sync* sync, int size)
    : sync_(sync)
    , size_(size)
  {
  }

  auto Size() const { return size_; }

  operator vk::Buffer() const { return buffer_; }
  auto Offset() const { return offset_; }
  auto ByteSize() const { return byte_size_; }

private:
  Sync* const sync_ = nullptr;

  int size_ = 0;

  vk::Buffer buffer_;
  vk::DeviceSize offset_ = 0;
  vk::DeviceSize byte_size_ = 0;
};
}
}

#endif // SUPERLUCENT_VKSYNC_BUFFER_H_
