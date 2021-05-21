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
  Engine(GLFWwindow* window, uint32_t max_width, uint32_t max_height);
  ~Engine();

private:
  void CreateInstance(GLFWwindow* window);
  void DestroyInstance();

  void CreateDevice();
  void DestroyDevice();

  void PreallocateMemory();
  void FreeMemory();

  void CreateSwapchain();
  void DestroySwapchain();

  void CreateRendertarget();
  void DestroyRendertarget();

  struct Memory
  {
    vk::DeviceMemory memory;
    vk::DeviceSize offset;
    vk::DeviceSize size;
  };

  Memory AcquireDeviceMemory(vk::Buffer buffer);
  Memory AcquireDeviceMemory(vk::Image image);
  Memory AcquireDeviceMemory(vk::MemoryRequirements memory_requirements);
  Memory AcquireHostMemory(vk::Buffer buffer);
  Memory AcquireHostMemory(vk::Image image);
  Memory AcquireHostMemory(vk::MemoryRequirements memory_requirements);

private:
  const uint32_t max_width_;
  const uint32_t max_height_;

  uint32_t width_ = 0;
  uint32_t height_ = 0;

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
    void* map = nullptr;
  };
  StagingBuffer staging_buffer_;

  struct UniformBuffer
  {
    static constexpr int size = 32 * 1024 * 1024; // 32MB

    vk::Buffer buffer;
    vk::DeviceMemory memory;
    void* map = nullptr;
  };
  UniformBuffer uniform_buffer_;

  // Swapchain
  vk::SwapchainKHR swapchain_;
  uint32_t swapchain_image_count_ = 0;
  vk::Format swapchain_image_format_;
  std::vector<vk::Image> swapchain_images_;
  std::vector<vk::ImageView> swapchain_image_views_;

  // Rendertarget
  struct Rendertarget
  {
    Memory color_memory;
    vk::Image color_image;
    vk::ImageView color_image_view;
    Memory depth_memory;
    vk::Image depth_image;
    vk::ImageView depth_image_view;
  };
  Rendertarget rendertarget_;
};
}

#endif // SUPERLUCENT_ENGINE_H_
