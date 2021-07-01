#ifndef SUPERLUCENT_VKSYNC_MAPPED_BUFFER_H_
#define SUPERLUCENT_VKSYNC_MAPPED_BUFFER_H_

#include <vulkan/vulkan.hpp>

namespace supl
{
namespace vksync
{
template <typename T>
class MappedBuffer
{
public:
  MappedBuffer() = delete;

  MappedBuffer(vk::Buffer buffer, vk::DeviceSize size, vk::DeviceMemory memory)
    : buffer_{ buffer }
    , size_{ size }
    , memory_{ memory }
    , byte_size_{ size * sizeof(T) }
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

  auto Size() const { return size_; }

  operator vk::Buffer() const { return buffer_; }
  auto ByteSize() const { return byte_size_; }

private:
  void Move(MappedBuffer&& rhs) noexcept
  {
    size_ = rhs.size_;
    buffer_ = rhs.buffer_;
    byte_size_ = rhs.byte_size_;
    memory_ = rhs.memory_;
    map_ = rhs.map_;

    rhs.size_ = 0;
    rhs.buffer_ = nullptr;
    rhs.byte_size_ = 0;
    rhs.memory_ = nullptr;
    rhs.map_ = nullptr;
  }

  int size_ = 0;
  vk::Buffer buffer_;
  vk::DeviceSize byte_size_ = 0;
  vk::DeviceMemory memory_;
  uint8_t* map_ = nullptr;
};
}
}

#endif // SUPERLUCENT_VKSYNC_MAPPED_BUFFER_H_
