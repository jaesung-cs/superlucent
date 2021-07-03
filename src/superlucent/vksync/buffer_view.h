#ifndef SUPERLUCENT_VKSYNC_BUFFER_VIEW_H_
#define SUPERLUCENT_VKSYNC_BUFFER_VIEW_H_

#include <superlucent/vksync/buffer.h>

namespace supl
{
namespace vksync
{
template <typename T>
class BufferView
{
public:
  BufferView() = delete;

  BufferView(int size = 1)
    : size_(size)
  {
  }

  ~BufferView() = default;

  auto Size() const { return size_; }

  operator vk::Buffer() const { return buffer_; }
  auto Offset() const { return offset_; }
  auto ByteSize() const { return byte_size_; }

private:
  int size_;
  vk::Buffer buffer_;
  vk::DeviceSize offset_ = 0;
  vk::DeviceSize byte_size_ = 0;
};
}
}

#endif // SUPERLUCENT_VKSYNC_BUFFER_VIEW_H_
