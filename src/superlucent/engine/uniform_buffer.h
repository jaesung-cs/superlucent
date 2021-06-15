#ifndef SUPERLUCENT_ENGINE_UNIFORM_BUFFER_H_
#define SUPERLUCENT_ENGINE_UNIFORM_BUFFER_H_

#include <vulkan/vulkan.hpp>

#include <superlucent/engine/engine.h>

namespace supl
{
namespace engine
{
class UniformBuffer;

class Uniform
{
public:
  Uniform() = delete;

  Uniform(UniformBuffer* uniform_buffer)
    : uniform_buffer_(uniform_buffer)
  {
  }

  template <typename T>
  Uniform& operator = (const T& rhs);

  // Public members
  vk::DeviceSize offset;
  vk::DeviceSize size;

private:
  UniformBuffer* const uniform_buffer_;
};

class UniformBuffer
{
public:
  UniformBuffer() = delete;

  explicit UniformBuffer(Engine* engine, vk::DeviceSize size);

  ~UniformBuffer();

  auto Buffer() const { return buffer_; }
  uint8_t* Map() const { return map_; }

  template <typename T>
  Uniform Allocate()
  {
    Uniform uniform{ this };
    uniform.offset = allocation_offset_;
    uniform.size = sizeof(T);

    allocation_offset_ = engine_->Align(uniform.offset + uniform.size, engine_->UboAlignment());

    return uniform;
  }

  template <typename T>
  std::vector<Uniform> Allocate(int count)
  {
    std::vector<Uniform> uniforms;
    for (int i = 0; i < count; i++)
      uniforms.emplace_back(Allocate<T>());
    return uniforms;
  }

private:
  Engine* const engine_;

  vk::DeviceMemory memory_;
  vk::Buffer buffer_;
  uint8_t* map_ = nullptr;

  vk::DeviceSize allocation_offset_ = 0;
};

template <typename T>
Uniform& Uniform::operator = (const T& rhs)
{
  std::memcpy(uniform_buffer_->Map() + offset, &rhs, sizeof(T));
  return *this;
}
}
}

#endif // SUPERLUCENT_ENGINE_UNIFORM_BUFFER_H_
