#ifndef SUPERLUCENT_ENGINE_ENGINE_H_
#define SUPERLUCENT_ENGINE_ENGINE_H_

#include <vulkan/vulkan.hpp>

#include <glm/glm.hpp>

#include <vkpbd/vkpbd.hpp>

#include <superlucent/engine/ubo/light_ubo.h>
#include <superlucent/engine/ubo/camera_ubo.h>

struct GLFWwindow;

namespace supl
{
namespace scene
{
class Light;
class Camera;
}

namespace engine
{
class ParticleSimulation;
class ParticleRenderer;
class UniformBuffer;

class Engine
{
private:
  using UniformBufferType = typename UniformBuffer;

public:
  static constexpr auto Align(vk::DeviceSize offset, vk::DeviceSize alignment)
  {
    return (offset + alignment - 1) & ~(alignment - 1);
  }

public:
  Engine() = delete;
  Engine(GLFWwindow* window, uint32_t max_width, uint32_t max_height);
  ~Engine();

  void Resize(uint32_t width, uint32_t height);
  void UpdateLights(const std::vector<std::shared_ptr<scene::Light>>& lights);
  void UpdateCamera(std::shared_ptr<scene::Camera> camera);
  void Draw(double time);

  // Vulkan getters and utils
  auto Device() const { return device_; }
  auto Queue() const { return queue_; }
  auto DescriptorPool() const { return descriptor_pool_; }
  auto HostMemoryIndex() const { return host_index_; }

  const auto UniformBuffer() const { return uniform_buffer_; }

  auto Rendertarget() const { return rendertarget_; }
  auto SwapchainImageCount() const { return swapchain_image_count_; }
  auto SwapchainImageFormat() const { return swapchain_image_format_; }
  const auto& SwapchainImageViews() const { return swapchain_image_views_; }
  auto SsboAlignment() const { return ssbo_alignment_; }
  auto UboAlignment() const { return ubo_alignment_; }

  vk::ShaderModule CreateShaderModule(const std::string& filepath);

  void ImageLayoutTransition(vk::CommandBuffer& command_buffer, vk::Image image, vk::ImageLayout old_layout, vk::ImageLayout new_layout, uint32_t mipmap_levels);
  void GenerateMipmap(vk::CommandBuffer& command_buffer, vk::Image image, uint32_t width, uint32_t height, uint32_t mipmap_levels);

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

  template <typename T>
  void ToDeviceMemory(const std::vector<T>& data, vk::Buffer buffer, vk::DeviceSize offset = 0)
  {
    const auto byte_size = data.size() * sizeof(T);
    std::memcpy(staging_buffer_.map, data.data(), byte_size);

    // Transfer commands
    vk::CommandBufferBeginInfo command_buffer_begin_info;
    command_buffer_begin_info
      .setFlags(vk::CommandBufferUsageFlagBits::eOneTimeSubmit);
    transient_command_buffer_.begin(command_buffer_begin_info);

    vk::BufferCopy copy_region;
    copy_region
      .setSrcOffset(0)
      .setDstOffset(offset)
      .setSize(byte_size);
    transient_command_buffer_.copyBuffer(staging_buffer_.buffer, buffer, copy_region);

    transient_command_buffer_.end();

    vk::SubmitInfo submit_info;
    submit_info
      .setCommandBuffers(transient_command_buffer_);
    queue_.submit(submit_info, transfer_fence_);

    // TODO: Don't wait for transfer finish!
    const auto wait_result = device_.waitForFences(transfer_fence_, true, UINT64_MAX);
    device_.resetFences(transfer_fence_);
    transient_command_buffer_.reset();
  }

  void ToDeviceMemory(const std::vector<uint8_t>& data, vk::Image image, uint32_t width, uint32_t height, uint32_t mipmap_levels);

  vk::CommandBuffer CreateOneTimeCommandBuffer();

private:
  void RecordDrawCommands(vk::CommandBuffer& command_buffer, uint32_t image_index, double dt);

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

  void CreateParticleSimulator();
  void DestroyParticleSimulator();

  void CreateSynchronizationObjects();
  void DestroySynchronizationObjects();

  void RecreateSwapchain();

private:
  const uint32_t max_width_;
  const uint32_t max_height_;

  uint32_t width_ = 0;
  uint32_t height_ = 0;

  double previous_time_ = 0.;
  double animation_time_ = 0.;

  // Instance
  vk::Instance instance_;
  vk::DebugUtilsMessengerEXT messenger_;
  vk::SurfaceKHR surface_;

  // Device
  vk::PhysicalDevice physical_device_;
  vk::Device device_;
  uint32_t queue_index_ = 0;
  uint32_t host_index_ = 0;
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

  std::shared_ptr<UniformBufferType> uniform_buffer_;

  // Swapchain
  vk::SwapchainKHR swapchain_;
  uint32_t swapchain_image_count_ = 0;
  vk::Format swapchain_image_format_;
  std::vector<vk::Image> swapchain_images_;
  std::vector<vk::ImageView> swapchain_image_views_;

  // Rendertarget
  struct RendertargetImages
  {
    Memory color_memory;
    vk::Image color_image;
    vk::ImageView color_image_view;
    Memory depth_memory;
    vk::Image depth_image;
    vk::ImageView depth_image_view;
  };
  RendertargetImages rendertarget_;

  // Renderer
  std::unique_ptr<ParticleRenderer> particle_renderer_;

  // vkpbd
  static constexpr auto commandCount = 3; // Triple buffer
  vkpbd::ParticleSimulator particleSimulator_;
  vk::Buffer particleBuffer_;
  vk::DeviceSize particleBufferSize_ = 0;
  vk::Buffer particleInternalBuffer_;
  vk::DeviceSize particleInternalBufferSize_ = 0;
  vk::Buffer particleUniformBuffer_;
  vk::DeviceSize particleUniformBufferSize_ = 0;
  vk::DeviceMemory particleUniformMemory_;
  uint8_t* particleUniformBufferMap_ = nullptr;

  // Command buffers
  vk::CommandBuffer transient_command_buffer_;
  std::vector<vk::CommandBuffer> draw_command_buffers_;

  vk::DeviceSize ubo_alignment_;
  vk::DeviceSize ssbo_alignment_;

  // Uniforms
  LightUbo lights_;
  CameraUbo camera_;

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
}

#endif // SUPERLUCENT_ENGINE_ENGINE_H_
