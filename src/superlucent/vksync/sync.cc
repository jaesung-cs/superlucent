#include <superlucent/vksync/sync.h>

#include <superlucent/vksync/host_buffer.h>

namespace supl
{
namespace vksync
{
Sync::Sync(vk::PhysicalDevice physical_device, vk::Device device)
  : physical_device_(physical_device), device_(device)
{
  uint32_t host_index = 0;
  uint32_t device_index = 0;

  // Find memory type index
  uint64_t device_available_size = 0;
  uint64_t host_available_size = 0;
  const auto memory_properties = physical_device_.getMemoryProperties();
  for (uint32_t i = 0; i < memory_properties.memoryTypeCount; i++)
  {
    const auto properties = memory_properties.memoryTypes[i].propertyFlags;
    const auto heap_index = memory_properties.memoryTypes[i].heapIndex;
    const auto heap = memory_properties.memoryHeaps[heap_index];

    if ((properties & vk::MemoryPropertyFlagBits::eDeviceLocal) == vk::MemoryPropertyFlagBits::eDeviceLocal)
    {
      if (heap.size > device_available_size)
      {
        device_index = i;
        device_available_size = heap.size;
      }
    }

    if ((properties & (vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent))
      == (vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent))
    {
      if (heap.size > host_available_size)
      {
        host_index = i;
        host_available_size = heap.size;
      }
    }
  }

  ubo_alignment_ = physical_device_.getProperties().limits.minUniformBufferOffsetAlignment;
  ssbo_alignment_ = physical_device_.getProperties().limits.minStorageBufferOffsetAlignment;

  constexpr vk::DeviceSize memory_chunk_size = 128 * 1024 * 1024; // 128MB

  vk::MemoryAllocateInfo memory_allocate_info;
}

Sync::~Sync()
{
}

vk::DeviceSize Sync::SsboAlign(vk::DeviceSize offset) const
{
  return (offset + ssbo_alignment_ - 1) & ~(offset - 1);
}

vk::DeviceSize Sync::UboAlign(vk::DeviceSize offset) const
{
  return (offset + ubo_alignment_ - 1) & ~(offset - 1);
}
}
}
