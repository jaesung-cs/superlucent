#ifndef SUPERLUCENT_VKSYNC_MAPPED_BUFFER_H_
#define SUPERLUCENT_VKSYNC_MAPPED_BUFFER_H_

#include <vulkan/vulkan.hpp>

namespace supl
{
namespace vksync
{
class MappedBuffer
{
public:
  MappedBuffer() = delete;

  MappedBuffer(vk::Buffer buffer, vk::DeviceSize size, vk::DeviceMemory memory)
    : buffer_{ buffer }
    , size_{ size }
    , memory_{ memory }
  {
  }

  ~MappedBuffer()
  {
  }

  MappedBuffer(const MappedBuffer& rhs) = delete;
  MappedBuffer& operator = (const MappedBuffer& rhs) = delete;

  MappedBuffer(MappedBuffer&& rhs) noexcept
  {
    Move(std::move(rhs));
  }

  MappedBuffer& operator = (MappedBuffer&& rhs) noexcept
  {
    Move(std::move(rhs));
    return *this;
  }

  operator vk::Buffer() const { return buffer_; }
  auto Size() const { return size_; }

private:
  void Move(MappedBuffer&& rhs) noexcept
  {
    buffer_ = rhs.buffer_;
    size_ = rhs.size_;
    memory_ = rhs.memory_;
    map_ = rhs.map_;

    rhs.buffer_ = nullptr;
    rhs.size_ = 0;
    rhs.memory_ = nullptr;
    rhs.map_ = nullptr;
  }

  vk::Buffer buffer_;
  vk::DeviceSize size_ = 0;
  vk::DeviceMemory memory_;
  uint8_t* map_ = nullptr;
};
}
}

#endif // SUPERLUCENT_VKSYNC_MAPPED_BUFFER_H_
