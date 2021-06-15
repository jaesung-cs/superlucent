#include <superlucent/engine/uniform_buffer.h>

#include <superlucent/engine/engine.h>

namespace supl
{
namespace engine
{
UniformBuffer::UniformBuffer(Engine* engine, vk::DeviceSize size)
  : engine_(engine)
{
  const auto device = engine->Device();

  vk::BufferCreateInfo buffer_create_info;
  buffer_create_info
    .setSize(size)
    .setUsage(vk::BufferUsageFlagBits::eUniformBuffer);
  buffer_ = device.createBuffer(buffer_create_info);

  // Allocate and bind to memory for uniform buffer
  vk::MemoryAllocateInfo memory_allocate_info;
  memory_allocate_info
    .setAllocationSize(device.getBufferMemoryRequirements(buffer_).size)
    .setMemoryTypeIndex(engine_->HostMemoryIndex());
  memory_ = device.allocateMemory(memory_allocate_info);

  device.bindBufferMemory(buffer_, memory_, 0);
  map_ = static_cast<uint8_t*>(device.mapMemory(memory_, 0, size));
}

UniformBuffer::~UniformBuffer()
{
  const auto device = engine_->Device();

  device.freeMemory(memory_);
  device.destroyBuffer(buffer_);
}

Uniform UniformBuffer::Allocate(vk::DeviceSize size)
{
  Uniform uniform;
  uniform.offset = allocation_offset_;
  uniform.size = size;

  allocation_offset_ = engine_->Align(allocation_offset_ + size, engine_->UboAlignment());

  return uniform;
}

std::vector<Uniform> UniformBuffer::Allocate(vk::DeviceSize size, int count)
{
  std::vector<Uniform> uniforms;
  for (int i = 0; i < count; i++)
    uniforms.emplace_back(Allocate(size));
  return uniforms;
}
}
}
