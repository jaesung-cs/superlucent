#ifndef SUPERLUCENT_ENGINE_H_
#define SUPERLUCENT_ENGINE_H_

#include <vulkan/vulkan.hpp>

struct GLFWwindow;

namespace supl
{
class Engine
{
public:
  Engine() = delete;
  Engine(GLFWwindow* window, int max_width, int max_height);
  ~Engine();

private:
  void CreateInstance(GLFWwindow* window);
  void DestroyInstance();

  void CreateDevice();
  void DestroyDevice();

  void PreallocateMemory();
  void FreeMemory();

private:
  const int max_width_;
  const int max_height_;

  int width_ = 0;
  int height_ = 0;

  // Instance
  vk::Instance instance_;
  vk::DebugUtilsMessengerEXT messenger_;
  vk::SurfaceKHR surface_;

  // Device
  vk::PhysicalDevice physical_device_;
  vk::Device device_;
  uint32_t queue_index_ = 0;
  vk::Queue queue_;
  vk::Queue present_queue_;

  // Memory
  vk::DeviceMemory device_memory_;
  vk::DeviceSize device_offset_ = 0;

  vk::DeviceMemory host_memory_;
  vk::DeviceSize host_offset_ = 0;

  struct StagingBuffer
  {
    static constexpr int size = 32 * 1024 * 1024; // 32MB

    vk::Buffer buffer;
    vk::DeviceMemory memory;
    void* map;
  };
  StagingBuffer staging_buffer_;

  struct UniformBuffer
  {
    static constexpr int size = 32 * 1024 * 1024; // 32MB

    vk::Buffer buffer;
    vk::DeviceMemory memory;
    void* map;
  };
  UniformBuffer uniform_buffer_;
};
}

#endif // SUPERLUCENT_ENGINE_H_
