#ifndef SUPERLUCENT_ENGINE_UNIFORM_BUFFER_H_
#define SUPERLUCENT_ENGINE_UNIFORM_BUFFER_H_

#include <vulkan/vulkan.hpp>

namespace supl
{
namespace engine
{
class Engine;

class UniformBuffer
{
public:
  UniformBuffer() = delete;

  explicit UniformBuffer(Engine* engine, vk::DeviceSize size);

  ~UniformBuffer();

  auto Buffer() const { return buffer_; }
  uint8_t* Map() const { return map_; }

private:
  Engine* const engine_;

  vk::DeviceMemory memory_;
  vk::Buffer buffer_;
  uint8_t* map_ = nullptr;
};
}
}

#endif // SUPERLUCENT_ENGINE_UNIFORM_BUFFER_H_
