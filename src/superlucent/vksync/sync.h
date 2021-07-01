#ifndef SUPERLUCENT_VKSYNC_SYNC_H_
#define SUPERLUCENT_VKSYNC_SYNC_H_

#include <vulkan/vulkan.hpp>

#include <superlucent/vksync/device_buffer.h>
#include <superlucent/vksync/mapped_buffer.h>

namespace supl
{
namespace vksync
{
class Sync
{
public:
  Sync() = delete;

  explicit Sync(vk::PhysicalDevice physical_device, vk::Device device);

  ~Sync();

  auto Device() const { return device_; }

  template <typename T>
  DeviceBuffer<T> CreateDeviceBuffer(int size, vk::BufferUsageFlags usage)
  {
    vk::BufferCreateInfo buffer_create_info;
    buffer_create_info
      .setUsage(usage)
      .setSize(size * sizeof(T));
    vk::Buffer buffer = device_.createBuffer(buffer_create_info);

    DeviceBuffer<T> device_buffer{ buffer, size };
    return device_buffer;
  }

  template <typename T>
  MappedBuffer<T> CreateMappedBuffer(int size, vk::BufferUsageFlags usage)
  {
    vk::BufferCreateInfo buffer_create_info;
    buffer_create_info
      .setUsage(usage)
      .setSize(size * sizeof(T));
    vk::Buffer buffer = device_.createBuffer(buffer_create_info);

    vk::MemoryAllocateInfo allocate_info;
    allocate_info
      .setAllocationSize(size * sizeof(T))
      .setMemoryTypeIndex(host_index_);

    MappedBuffer<T> mapped_buffer{ buffer, size, memory };
    return mapped_buffer;
  }

private:
  vk::DeviceSize SsboAlign(vk::DeviceSize offset) const;
  vk::DeviceSize UboAlign(vk::DeviceSize offset) const;

  const vk::PhysicalDevice physical_device_;
  const vk::Device device_;

  vk::DeviceSize ubo_alignment_;
  vk::DeviceSize ssbo_alignment_;

  vk::DeviceMemory device_memory_;
  vk::DeviceMemory host_memory_;

  vk::DeviceSize storage_offset_ = 0;
  vk::DeviceSize uniform_offset_ = 0;
};
}
}

#endif // SUPERLUCENT_VKSYNC_SYNC_H_
