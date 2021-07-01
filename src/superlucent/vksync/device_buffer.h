#ifndef SUPERLUCENT_VKSYNC_DEVICE_BUFFER_H_
#define SUPERLUCENT_VKSYNC_DEVICE_BUFFER_H_

#include <vulkan/vulkan.hpp>

namespace supl
{
namespace vksync
{
template <typename T>
class DeviceBuffer
{
public:
  DeviceBuffer() = delete;

  DeviceBuffer(vk::Buffer buffer, int size)
    : buffer_{ buffer }
    , size_{ size }
    , byte_size_{ size * sizeof(T) }
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
  auto Size() const { return size_; }

  operator vk::Buffer() const { return buffer_; }
  auto ByteSize() const { return byte_size_; }

private:
  void Move(DeviceBuffer&& rhs)
  {
    size_ = rhs.size_;
    buffer_ = rhs.buffer_;
    byte_size_ = rhs.byte_size_;

    rhs.size_ = 0;
    rhs.buffer_ = nullptr;
    rhs.byte_size_ = 0;
  }

  int size_ = 0;
  vk::Buffer buffer_;
  vk::DeviceSize byte_size_ = 0;
};
}
}

#endif // SUPERLUCENT_VKSYNC_DEVICE_BUFFER_H_
