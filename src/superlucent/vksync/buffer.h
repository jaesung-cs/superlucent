#ifndef SUPERLUCENT_VKSYNC_BUFFER_H_
#define SUPERLUCENT_VKSYNC_BUFFER_H_

#include <vulkan/vulkan.hpp>

namespace supl
{
namespace vksync
{
class DeviceBuffer
{
public:
  DeviceBuffer() = delete;

  DeviceBuffer(vk::Buffer buffer, vk::DeviceSize size)
    : buffer_{ buffer }
    , size_{ size }
  {
  }

  DeviceBuffer(const DeviceBuffer& rhs) = delete;
  DeviceBuffer& operator = (const DeviceBuffer& rhs) = delete;

  DeviceBuffer(DeviceBuffer&& rhs) noexcept
  {
    Move(std::move(rhs));
  }

  DeviceBuffer& operator = (DeviceBuffer&& rhs) noexcept
  {
    Move(std::move(rhs));
    return *this;
  }

public:
  operator vk::Buffer() const { return buffer_; }
  auto ByteSize() const { return size_; }

private:
  void Move(DeviceBuffer&& rhs)
  {
    buffer_ = rhs.buffer_;
    size_ = rhs.size_;

    rhs.buffer_ = nullptr;
    rhs.size_ = 0;
  }

  vk::Buffer buffer_;
  vk::DeviceSize size_ = 0;
};
}
}

#endif // SUPERLUCENT_VKSYNC_BUFFER_H_
