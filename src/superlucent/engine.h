#ifndef SUPERLUCENT_ENGINE_H_
#define SUPERLUCENT_ENGINE_H_

#include <vulkan/vulkan.hpp>

struct GLFWwindow;

namespace supl
{
namespace scene
{
class Camera;
}

class Engine
{
public:
  Engine() = delete;
  Engine(GLFWwindow* window, uint32_t max_width, uint32_t max_height);
  ~Engine();

  void Resize(uint32_t width, uint32_t height);
  void UpdateCamera(std::shared_ptr<scene::Camera> camera);
  void Draw();

private:
  void CreateInstance(GLFWwindow* window);
  void DestroyInstance();

  void CreateDevice();
  void DestroyDevice();

  void CreateSwapchain();
  void DestroySwapchain();

  void PreallocateMemory();
  void FreeMemory();

  void AllocateCommandBuffers();
  void FreeCommandBuffers();

  void CreateRendertarget();
  void DestroyRendertarget();

  void CreateFramebuffer();
  void DestroyFramebuffer();

  void CreatePipelines();
  void DestroyPipelines();

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

  vk::ShaderModule CreateShaderModule(const std::string& filepath);

  void CreateGraphicsPipeline();
  void DestroyGraphicsPipeline();

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

  // Command pools
  vk::CommandPool command_pool_;
  vk::CommandPool transient_command_pool_;

  // Descriptor pool
  vk::DescriptorPool descriptor_pool_;

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

  // Pipeline
  vk::RenderPass render_pass_;
  std::vector<vk::Framebuffer> swapchain_framebuffers_;
  vk::PipelineCache pipeline_cache_;

  vk::DescriptorSetLayout graphics_descriptor_set_layout_;
  vk::PipelineLayout graphics_pipeline_layout_;
  vk::Pipeline graphics_pipeline_;

  // Command buffers
  vk::CommandBuffer transient_command_buffer_;
  std::vector<vk::CommandBuffer> draw_command_buffers_;
};
}

#endif // SUPERLUCENT_ENGINE_H_
