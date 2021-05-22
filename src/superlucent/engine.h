#ifndef SUPERLUCENT_ENGINE_H_
#define SUPERLUCENT_ENGINE_H_

#include <vulkan/vulkan.hpp>

#include <glm/glm.hpp>

struct GLFWwindow;

namespace supl
{
namespace scene
{
class Camera;
}

class Engine
{
private:
  // Binding 0
  struct CameraUbo
  {
    alignas(16) glm::mat4 projection;
    alignas(16) glm::mat4 view;
    alignas(16) glm::vec3 eye;
  };

  // Binding 1
  struct ModelUbo
  {
    alignas(16) glm::mat4 model;
    alignas(16) glm::mat3x4 model_inverse_transpose;
  };

public:
  Engine() = delete;
  Engine(GLFWwindow* window, uint32_t max_width, uint32_t max_height);
  ~Engine();

  void Resize(uint32_t width, uint32_t height);
  void UpdateCamera(std::shared_ptr<scene::Camera> camera);
  void Draw();

private:
  void RecordDrawCommands(vk::CommandBuffer& command_buffer, uint32_t image_index);

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

  void PrepareResources();
  void DestroyResources();

  void CreateSynchronizationObjects();
  void DestroySynchronizationObjects();

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
    uint8_t* map = nullptr;
  };
  StagingBuffer staging_buffer_;

  struct UniformBuffer
  {
    static constexpr int size = 32 * 1024 * 1024; // 32MB

    vk::Buffer buffer;
    vk::DeviceMemory memory;
    uint8_t* map = nullptr;
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

  // Resources
  struct VertexBuffer
  {
    vk::Buffer buffer;
    vk::DeviceSize index_offset;
    uint32_t num_indices;
  };
  VertexBuffer triangle_buffer_;

  // Descriptor set
  std::vector<vk::DescriptorSet> graphics_descriptor_sets_;

  vk::DeviceSize ubo_alignment_;
  CameraUbo camera_;
  ModelUbo triangle_model_;

  struct Uniform
  {
    vk::DeviceSize offset;
    vk::DeviceSize size;
  };
  std::vector<Uniform> camera_ubos_;
  std::vector<Uniform> triangle_model_ubos_;

  // Transfer
  vk::Fence transfer_fence_;

  // Present synchronization
  std::vector<vk::Semaphore> image_available_semaphores_;
  std::vector<vk::Semaphore> render_finished_semaphores_;
  uint32_t current_frame_ = 0;
  std::vector<vk::Fence> in_flight_fences_;
  std::vector<vk::Fence> images_in_flight_;
};
}

#endif // SUPERLUCENT_ENGINE_H_
