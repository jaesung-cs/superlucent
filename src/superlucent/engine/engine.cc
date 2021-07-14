#include <superlucent/engine/engine.h>

#include <iostream>
#include <fstream>

#include <GLFW/glfw3.h>

#include <glm/gtc/matrix_transform.hpp>

#include <vkpbd/particle.h>

#include <superlucent/engine/particle_renderer.h>
#include <superlucent/engine/uniform_buffer.h>
#include <superlucent/scene/light.h>
#include <superlucent/scene/camera.h>
#include <superlucent/utils/rng.h>

namespace supl
{
namespace engine
{
namespace
{
VKAPI_ATTR VkBool32 VKAPI_CALL debug_callback(
  VkDebugUtilsMessageSeverityFlagBitsEXT message_severity,
  VkDebugUtilsMessageTypeFlagsEXT message_type,
  const VkDebugUtilsMessengerCallbackDataEXT* callback_data,
  void* pUserData)
{
  if (message_severity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT)
    std::cerr << callback_data->pMessage << std::endl << std::endl;

  return VK_FALSE;
}
}

Engine::Engine(GLFWwindow* window, uint32_t max_width, uint32_t max_height)
  : max_width_{ max_width }
  , max_height_{ max_height }
{
  // Current width and height
  int width, height;
  glfwGetWindowSize(window, &width, &height);
  width_ = static_cast<uint32_t>(width);
  height_ = static_cast<uint32_t>(height);

  // Prepare vulkan resources
  CreateInstance(window);
  CreateDevice();

  CreateSwapchain();
  PreallocateMemory();
  AllocateCommandBuffers();
  CreateRendertarget();

  // Create particle renderer and simulator
  particle_renderer_ = std::make_unique<ParticleRenderer>(this, width_, height_);

  // Create vkpbd
  CreateSimulator();

  CreateSynchronizationObjects();
}

Engine::~Engine()
{
  device_.waitIdle();

  DestroySynchronizationObjects();
  DestroySimulator();

  particle_renderer_ = nullptr;

  DestroyRendertarget();
  FreeCommandBuffers();
  FreeMemory();
  DestroySwapchain();
  DestroyDevice();
  DestroyInstance();
}

void Engine::Resize(uint32_t width, uint32_t height)
{
  width_ = width;
  height_ = height;

  RecreateSwapchain();
}

void Engine::RecreateSwapchain()
{
  device_.waitIdle();

  // Recreate swapchain image views
  DestroySwapchain();
  CreateSwapchain();

  // Recreate rendertarget images views
  DestroyRendertarget();
  CreateRendertarget();

  // Resize renderer
  particle_renderer_->Resize(width_, height_);
}

void Engine::UpdateLights(const std::vector<std::shared_ptr<scene::Light>>& lights)
{
  int num_directional_lights = 0;
  int num_point_lights = 0;

  for (auto light : lights)
  {
    LightUbo::Light light_data;
    light_data.position = light->Position();
    light_data.ambient = light->Ambient();
    light_data.diffuse = light->Diffuse();
    light_data.specular = light->Specular();

    if (light->IsDirectionalLight())
    {
      light_data.position = glm::normalize(light_data.position);
      lights_.directional_lights[num_directional_lights++] = light_data;
    }
    else if (light->IsPointLight())
      lights_.point_lights[num_point_lights++] = light_data;
  }
}

void Engine::UpdateCamera(std::shared_ptr<scene::Camera> camera)
{
  camera_.view = camera->ViewMatrix();

  camera_.projection = camera->ProjectionMatrix();
  camera_.projection[1][1] *= -1.f;

  camera_.eye = camera->Eye();
}

void Engine::Draw(double time)
{
  auto dt = time - previous_time_;
  previous_time_ = time;

  animation_time_ += dt;

  auto wait_result = device_.waitForFences(in_flight_fences_[current_frame_], true, UINT64_MAX);

  const auto acquire_next_image_result = device_.acquireNextImageKHR(swapchain_, UINT64_MAX, image_available_semaphores_[current_frame_]);
  if (acquire_next_image_result.result == vk::Result::eErrorOutOfDateKHR)
  {
    RecreateSwapchain();
    return;
  }
  else if (acquire_next_image_result.result != vk::Result::eSuccess && acquire_next_image_result.result != vk::Result::eSuboptimalKHR)
    throw std::runtime_error("Failed to acquire next swapchain image");

  const auto image_index = acquire_next_image_result.value;

  if (images_in_flight_[image_index])
    wait_result = device_.waitForFences(images_in_flight_[image_index], true, UINT64_MAX);
  images_in_flight_[image_index] = in_flight_fences_[current_frame_];

  device_.resetFences(in_flight_fences_[current_frame_]);

  // Build command buffer
  auto& draw_command_buffer = draw_command_buffers_[image_index];
  draw_command_buffer.reset();

  draw_command_buffer.begin({ vk::CommandBufferUsageFlagBits::eOneTimeSubmit });

  if (dt > 0.)
  {
    fluidSimulator_.cmdBindSrcParticleBuffer(particleBuffer_, particleBufferSize_ * image_index);
    fluidSimulator_.cmdBindDstParticleBuffer(particleBuffer_, particleBufferSize_ * ((image_index + 1) % 3));
    fluidSimulator_.cmdBindInternalBuffer(particleInternalBuffer_, 0);
    fluidSimulator_.cmdBindUniformBuffer(particleUniformBuffer_, 0, particleUniformBufferMap_);
    fluidSimulator_.cmdStep(draw_command_buffer, image_index, animation_time_, dt);

    vk::BufferMemoryBarrier barrier;
    barrier
      .setBuffer(particleBuffer_)
      .setSrcAccessMask(vk::AccessFlagBits::eShaderWrite)
      .setDstAccessMask(vk::AccessFlagBits::eVertexAttributeRead)
      .setSrcQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
      .setDstQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
      .setOffset(particleBufferSize_ * ((image_index + 1) % 3))
      .setSize(particleBufferSize_);

    draw_command_buffer.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eVertexInput, {},
      {}, barrier, {});
  }
  else
  {
    vk::BufferCopy region;
    region
      .setSrcOffset(particleBufferSize_ * image_index)
      .setDstOffset(particleBufferSize_ * ((image_index + 1) % 3))
      .setSize(particleBufferSize_);
    draw_command_buffer.copyBuffer(particleBuffer_, particleBuffer_, region);

    vk::BufferMemoryBarrier barrier;
    barrier
      .setBuffer(particleBuffer_)
      .setSrcAccessMask(vk::AccessFlagBits::eTransferWrite)
      .setDstAccessMask(vk::AccessFlagBits::eVertexAttributeRead)
      .setSrcQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
      .setDstQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
      .setOffset(particleBufferSize_ * ((image_index + 1) % 3))
      .setSize(particleBufferSize_);

    draw_command_buffer.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eVertexInput, {},
      {}, barrier, {});
  }

  RecordDrawCommands(draw_command_buffer, image_index, dt);

  draw_command_buffer.end();

  // Update uniforms
  particle_renderer_->UpdateLights(lights_, image_index);
  particle_renderer_->UpdateCamera(camera_, image_index);

  // Submit
  std::vector<vk::Semaphore> wait_semaphores{
    image_available_semaphores_[current_frame_],
  };

  std::vector<vk::PipelineStageFlags> stages{
    vk::PipelineStageFlagBits::eColorAttachmentOutput,
  };
  vk::SubmitInfo submit_info;
  submit_info
    .setWaitSemaphores(wait_semaphores)
    .setWaitDstStageMask(stages)
    .setCommandBuffers(draw_command_buffer)
    .setSignalSemaphores(render_finished_semaphores_[current_frame_]);
  queue_.submit(submit_info, in_flight_fences_[current_frame_]);

  // Present
  std::vector<uint32_t> image_indices{ image_index };
  vk::PresentInfoKHR present_info;
  present_info
    .setWaitSemaphores(render_finished_semaphores_[current_frame_])
    .setSwapchains(swapchain_)
    .setImageIndices(image_indices);
  const auto present_result = present_queue_.presentKHR(present_info);

  if (present_result == vk::Result::eErrorOutOfDateKHR || present_result == vk::Result::eSuboptimalKHR)
    RecreateSwapchain();
  else if (present_result != vk::Result::eSuccess)
    throw std::runtime_error("Failed to present swapchain image");

  current_frame_ = (current_frame_ + 1) % 2;
}

void Engine::RecordDrawCommands(vk::CommandBuffer& command_buffer, uint32_t image_index, double dt)
{
  constexpr auto radius = 0.03f;

  particle_renderer_->Begin(command_buffer, image_index);
  particle_renderer_->RecordParticleRenderCommands(command_buffer, particleBuffer_, particleBufferSize_ * ((image_index + 1) % 3), fluidSimulator_.getParticleCount(), radius);
  particle_renderer_->RecordFloorRenderCommands(command_buffer);
  particle_renderer_->End(command_buffer);
}

Engine::Memory Engine::AcquireDeviceMemory(vk::Buffer buffer)
{
  return AcquireDeviceMemory(device_.getBufferMemoryRequirements(buffer));
}

Engine::Memory Engine::AcquireDeviceMemory(vk::Image image)
{
  return AcquireDeviceMemory(device_.getImageMemoryRequirements(image));
}

Engine::Memory Engine::AcquireDeviceMemory(vk::MemoryRequirements memory_requirements)
{
  Memory memory;
  memory.memory = device_memory_;
  memory.offset = Align(device_offset_, memory_requirements.alignment);
  memory.size = memory_requirements.size;
  device_offset_ = memory.offset + memory.size;
  return memory;
}

Engine::Memory Engine::AcquireHostMemory(vk::Buffer buffer)
{
  return AcquireHostMemory(device_.getBufferMemoryRequirements(buffer));
}

Engine::Memory Engine::AcquireHostMemory(vk::Image image)
{
  return AcquireHostMemory(device_.getImageMemoryRequirements(image));
}

Engine::Memory Engine::AcquireHostMemory(vk::MemoryRequirements memory_requirements)
{
  Memory memory;
  memory.memory = host_memory_;
  memory.offset = Align(device_offset_, memory_requirements.alignment);
  memory.size = memory_requirements.size;
  device_offset_ = memory.offset + memory.size;
  return memory;
}

void Engine::ToDeviceMemory(const std::vector<uint8_t>& data, vk::Image image, uint32_t width, uint32_t height, uint32_t mipmap_levels)
{
  const auto byte_size = data.size() * sizeof(uint8_t);
  std::memcpy(staging_buffer_.map, data.data(), byte_size);

  // Transfer commands
  vk::CommandBufferBeginInfo command_buffer_begin_info;
  command_buffer_begin_info
    .setFlags(vk::CommandBufferUsageFlagBits::eOneTimeSubmit);
  transient_command_buffer_.begin(command_buffer_begin_info);

  vk::BufferCopy copy_region;
  ImageLayoutTransition(transient_command_buffer_, image,
    vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal, mipmap_levels);

  vk::ImageSubresourceLayers image_subresource_layer;
  image_subresource_layer
    .setAspectMask(vk::ImageAspectFlagBits::eColor)
    .setMipLevel(0)
    .setBaseArrayLayer(0)
    .setLayerCount(1);

  vk::BufferImageCopy image_copy_region;
  image_copy_region
    .setBufferOffset(0)
    .setBufferRowLength(0)
    .setBufferImageHeight(0)
    .setImageSubresource(image_subresource_layer)
    .setImageOffset(vk::Offset3D{ 0, 0, 0 })
    .setImageExtent(vk::Extent3D{ width, height, 1 });

  transient_command_buffer_.copyBufferToImage(staging_buffer_.buffer, image, vk::ImageLayout::eTransferDstOptimal, image_copy_region);

  GenerateMipmap(transient_command_buffer_, image, width, height, mipmap_levels);

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

vk::CommandBuffer Engine::CreateOneTimeCommandBuffer()
{
  vk::CommandBufferAllocateInfo command_buffer_allocate_info;
  command_buffer_allocate_info
    .setCommandPool(transient_command_pool_)
    .setLevel(vk::CommandBufferLevel::ePrimary)
    .setCommandBufferCount(1);
  auto transient_command_buffer = device_.allocateCommandBuffers(command_buffer_allocate_info)[0];

  // Transfer commands
  vk::CommandBufferBeginInfo command_buffer_begin_info;
  command_buffer_begin_info
    .setFlags(vk::CommandBufferUsageFlagBits::eOneTimeSubmit);
  transient_command_buffer.begin(command_buffer_begin_info);

  return transient_command_buffer;
}

void Engine::CreateInstance(GLFWwindow* window)
{
  const auto instance_extensions = vk::enumerateInstanceExtensionProperties();
  std::cout << "Instance extensions:" << std::endl;
  for (int i = 0; i < instance_extensions.size(); i++)
    std::cout << "  " << instance_extensions[i].extensionName << std::endl;

  const auto instance_layers = vk::enumerateInstanceLayerProperties();
  std::cout << "Instance layers:" << std::endl;
  for (int i = 0; i < instance_layers.size(); i++)
    std::cout << "  " << instance_layers[i].layerName << std::endl;

  // App
  vk::ApplicationInfo app_info;
  app_info
    .setPApplicationName("Superlucent")
    .setApplicationVersion(VK_MAKE_VERSION(1, 0, 0))
    .setPEngineName("Superlucent Engine")
    .setEngineVersion(VK_MAKE_VERSION(1, 0, 0))
    .setApiVersion(VK_API_VERSION_1_2);

  // Layers
  std::vector<const char*> layers = {
    "VK_LAYER_KHRONOS_validation",
  };

  // Extensions
  std::vector<const char*> extensions = {
    VK_EXT_DEBUG_UTILS_EXTENSION_NAME,
  };

  uint32_t num_glfw_extensions = 0;
  const char** glfw_extensions;
  glfw_extensions = glfwGetRequiredInstanceExtensions(&num_glfw_extensions);
  for (uint32_t i = 0; i < num_glfw_extensions; i++)
    extensions.push_back(glfw_extensions[i]);

  // Create instance
  vk::InstanceCreateInfo instance_create_info;
  instance_create_info
    .setPApplicationInfo(&app_info)
    .setPEnabledLayerNames(layers)
    .setPEnabledExtensionNames(extensions);

  vk::DebugUtilsMessengerCreateInfoEXT messenger_create_info;
  messenger_create_info
    .setMessageSeverity(vk::DebugUtilsMessageSeverityFlagBitsEXT::eError | vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning | vk::DebugUtilsMessageSeverityFlagBitsEXT::eVerbose)
    .setMessageType(vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral | vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation | vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance)
    .setPfnUserCallback(debug_callback);

  vk::StructureChain<vk::InstanceCreateInfo, vk::DebugUtilsMessengerCreateInfoEXT> chain{
    instance_create_info, messenger_create_info
  };
  instance_ = vk::createInstance(chain.get<vk::InstanceCreateInfo>());

  // Create messneger
  vk::DynamicLoader dl;
  PFN_vkGetInstanceProcAddr vkGetInstanceProcAddr = dl.getProcAddress<PFN_vkGetInstanceProcAddr>("vkGetInstanceProcAddr");
  vk::DispatchLoaderDynamic dld{ instance_, vkGetInstanceProcAddr };
  messenger_ = instance_.createDebugUtilsMessengerEXT(chain.get<vk::DebugUtilsMessengerCreateInfoEXT>(), nullptr, dld);

  // Create surface
  VkSurfaceKHR surface_handle;
  glfwCreateWindowSurface(instance_, window, nullptr, &surface_handle);
  surface_ = surface_handle;
}

void Engine::DestroyInstance()
{
  instance_.destroySurfaceKHR(surface_);

  vk::DynamicLoader dl;
  PFN_vkGetInstanceProcAddr vkGetInstanceProcAddr = dl.getProcAddress<PFN_vkGetInstanceProcAddr>("vkGetInstanceProcAddr");
  vk::DispatchLoaderDynamic dld{ instance_, vkGetInstanceProcAddr };
  instance_.destroyDebugUtilsMessengerEXT(messenger_, nullptr, dld);

  instance_.destroy();
}

void Engine::CreateDevice()
{
  // Choose the first GPU
  physical_device_ = instance_.enumeratePhysicalDevices()[0];

  // Available extensions
  const auto device_extensions = physical_device_.enumerateDeviceExtensionProperties();
  std::cout << "Device extensions:" << std::endl;
  for (int i = 0; i < device_extensions.size(); i++)
    std::cout << "  " << device_extensions[i].extensionName << std::endl;

  // Device properties
  ubo_alignment_ = physical_device_.getProperties().limits.minUniformBufferOffsetAlignment;
  ssbo_alignment_ = physical_device_.getProperties().limits.minStorageBufferOffsetAlignment;

  // Find general queue capable of graphics, compute and present
  constexpr auto queue_flag = vk::QueueFlagBits::eGraphics | vk::QueueFlagBits::eCompute;
  const auto queue_family_properties = physical_device_.getQueueFamilyProperties();
  for (int i = 0; i < queue_family_properties.size(); i++)
  {
    if ((queue_family_properties[i].queueFlags & queue_flag) == queue_flag &&
      physical_device_.getSurfaceSupportKHR(i, surface_) &&
      queue_family_properties[i].queueCount >= 2)
    {
      queue_index_ = i;
      break;
    }
  }

  std::vector<float> queue_priorities = {
    1.f, 1.f
  };
  vk::DeviceQueueCreateInfo queue_create_info{ {},
    queue_index_, queue_priorities
  };

  // Device extensions
  std::vector<const char*> extensions = {
    VK_KHR_SWAPCHAIN_EXTENSION_NAME
  };

  // Device features
  auto features = physical_device_.getFeatures();
  features
    .setTessellationShader(true)
    .setGeometryShader(true)
    .setSamplerAnisotropy(true);

  // Create device
  vk::DeviceCreateInfo device_create_info;
  device_create_info
    .setQueueCreateInfos(queue_create_info)
    .setPEnabledExtensionNames(extensions)
    .setPEnabledFeatures(&features);
  device_ = physical_device_.createDevice(device_create_info);

  queue_ = device_.getQueue(queue_index_, 0);
  present_queue_ = device_.getQueue(queue_index_, 1);
}

void Engine::DestroyDevice()
{
  device_.destroy();
}

void Engine::CreateSwapchain()
{
  const auto capabilities = physical_device_.getSurfaceCapabilitiesKHR(surface_);

  // Triple buffering
  auto image_count = capabilities.minImageCount + 1;
  if (capabilities.maxImageCount > 0 && image_count > capabilities.maxImageCount)
    image_count = capabilities.maxImageCount;

  if (image_count != 3)
    throw std::runtime_error("Triple buffering is not supported");

  // Present mode: use mailbox if available. Limit fps in draw call
  vk::PresentModeKHR present_mode = vk::PresentModeKHR::eFifo;
  const auto present_modes = physical_device_.getSurfacePresentModesKHR(surface_);
  for (auto available_mode : present_modes)
  {
    if (available_mode == vk::PresentModeKHR::eMailbox)
      present_mode = vk::PresentModeKHR::eMailbox;
  }

  // Format
  const auto available_formats = physical_device_.getSurfaceFormatsKHR(surface_);
  auto format = available_formats[0];
  for (const auto& available_format : available_formats)
  {
    if (available_format.format == vk::Format::eB8G8R8A8Srgb &&
      available_format.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear)
      format = available_format;
  }

  // Extent
  vk::Extent2D extent;
  if (capabilities.currentExtent.width != UINT32_MAX)
    extent = capabilities.currentExtent;
  else
  {
    VkExtent2D actual_extent = { width_, height_ };

    actual_extent.width = std::max(capabilities.minImageExtent.width, std::min(capabilities.maxImageExtent.width, actual_extent.width));
    actual_extent.height = std::max(capabilities.minImageExtent.height, std::min(capabilities.maxImageExtent.height, actual_extent.height));

    extent = actual_extent;
  }

  // Create swapchain
  vk::SwapchainCreateInfoKHR swapchain_create_info;
  swapchain_create_info
    .setSurface(surface_)
    .setMinImageCount(image_count)
    .setImageFormat(format.format)
    .setImageColorSpace(format.colorSpace)
    .setImageExtent(extent)
    .setImageArrayLayers(1)
    .setImageUsage(vk::ImageUsageFlagBits::eColorAttachment)
    .setImageSharingMode(vk::SharingMode::eExclusive)
    .setPreTransform(capabilities.currentTransform)
    .setCompositeAlpha(vk::CompositeAlphaFlagBitsKHR::eOpaque)
    .setPresentMode(present_mode)
    .setClipped(true);
  swapchain_ = device_.createSwapchainKHR(swapchain_create_info);

  swapchain_image_format_ = format.format;
  swapchain_image_count_ = image_count;

  swapchain_images_ = device_.getSwapchainImagesKHR(swapchain_);

  // Create image view for swapchain
  swapchain_image_views_.resize(swapchain_images_.size());
  for (int i = 0; i < swapchain_images_.size(); i++)
  {
    vk::ImageSubresourceRange subresource_range;
    subresource_range
      .setAspectMask(vk::ImageAspectFlagBits::eColor)
      .setBaseArrayLayer(0)
      .setLayerCount(1)
      .setBaseMipLevel(0)
      .setLevelCount(1);

    vk::ImageViewCreateInfo image_view_create_info;
    image_view_create_info
      .setImage(swapchain_images_[i])
      .setViewType(vk::ImageViewType::e2D)
      .setFormat(swapchain_image_format_)
      .setSubresourceRange(subresource_range);

    swapchain_image_views_[i] = device_.createImageView(image_view_create_info);
  }
}

void Engine::DestroySwapchain()
{
  for (auto& image_view : swapchain_image_views_)
    device_.destroyImageView(image_view);
  swapchain_image_views_.clear();

  device_.destroySwapchainKHR(swapchain_);
}

void Engine::PreallocateMemory()
{
  uint32_t device_index = 0;

  // Find memroy type index
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
        host_index_ = i;
        host_available_size = heap.size;
      }
    }
  }

  constexpr uint64_t chunk_size = 256 * 1024 * 1024; // 256MB

  vk::MemoryAllocateInfo memory_allocate_info;
  memory_allocate_info
    .setAllocationSize(chunk_size)
    .setMemoryTypeIndex(device_index);
  device_memory_ = device_.allocateMemory(memory_allocate_info);

  memory_allocate_info
    .setAllocationSize(chunk_size)
    .setMemoryTypeIndex(host_index_);
  host_memory_ = device_.allocateMemory(memory_allocate_info);

  // Persistently mapped staging buffer
  vk::BufferCreateInfo buffer_create_info;
  buffer_create_info
    .setSize(staging_buffer_.size)
    .setUsage(vk::BufferUsageFlagBits::eTransferSrc);
  staging_buffer_.buffer = device_.createBuffer(buffer_create_info);

  memory_allocate_info
    .setAllocationSize(device_.getBufferMemoryRequirements(staging_buffer_.buffer).size)
    .setMemoryTypeIndex(host_index_);
  staging_buffer_.memory = device_.allocateMemory(memory_allocate_info);

  device_.bindBufferMemory(staging_buffer_.buffer, staging_buffer_.memory, 0);
  staging_buffer_.map = static_cast<uint8_t*>(device_.mapMemory(staging_buffer_.memory, 0, staging_buffer_.size));

  // Persistently mapped uniform buffer
  constexpr uint64_t uniform_buffer_size = 32 * 1024 * 1024; // 32MB
  uniform_buffer_ = std::make_shared<UniformBufferType>(this, uniform_buffer_size);

  // Preallocate framebuffer memory
  vk::ImageCreateInfo image_create_info;
  image_create_info
    .setImageType(vk::ImageType::e2D)
    .setFormat(swapchain_image_format_)
    .setExtent(vk::Extent3D{ max_width_, max_height_, 1 })
    .setMipLevels(1)
    .setArrayLayers(1)
    .setSamples(vk::SampleCountFlagBits::e4)
    .setTiling(vk::ImageTiling::eOptimal)
    .setUsage(vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eTransientAttachment)
    .setSharingMode(vk::SharingMode::eExclusive)
    .setInitialLayout(vk::ImageLayout::eUndefined);
  auto temp_color_image = device_.createImage(image_create_info);
  rendertarget_.color_memory = AcquireDeviceMemory(temp_color_image);
  device_.destroyImage(temp_color_image);

  image_create_info
    .setImageType(vk::ImageType::e2D)
    .setFormat(vk::Format::eD24UnormS8Uint)
    .setExtent(vk::Extent3D{ max_width_, max_height_, 1 })
    .setMipLevels(1)
    .setArrayLayers(1)
    .setSamples(vk::SampleCountFlagBits::e4)
    .setTiling(vk::ImageTiling::eOptimal)
    .setUsage(vk::ImageUsageFlagBits::eDepthStencilAttachment | vk::ImageUsageFlagBits::eTransientAttachment)
    .setSharingMode(vk::SharingMode::eExclusive)
    .setInitialLayout(vk::ImageLayout::eUndefined);
  auto temp_depth_image = device_.createImage(image_create_info);
  rendertarget_.depth_memory = AcquireDeviceMemory(temp_depth_image);
  device_.destroyImage(temp_depth_image);

  // Preallocate descriptor pool
  constexpr uint32_t max_num_descriptors = 1024;
  constexpr uint32_t max_sets = 1024;
  std::vector<vk::DescriptorPoolSize> pool_sizes{
    { vk::DescriptorType::eUniformBuffer, max_num_descriptors },
    { vk::DescriptorType::eUniformBufferDynamic, max_num_descriptors },
    { vk::DescriptorType::eCombinedImageSampler, 16 },
    { vk::DescriptorType::eStorageBuffer, max_num_descriptors },
  };
  vk::DescriptorPoolCreateInfo descriptor_pool_create_info;
  descriptor_pool_create_info
    .setMaxSets(max_sets)
    .setPoolSizes(pool_sizes);
  descriptor_pool_ = device_.createDescriptorPool(descriptor_pool_create_info);

  // Preallocate command pools
  vk::CommandPoolCreateInfo command_pool_create_info;
  command_pool_create_info
    .setFlags(vk::CommandPoolCreateFlagBits::eResetCommandBuffer)
    .setQueueFamilyIndex(queue_index_);
  command_pool_ = device_.createCommandPool(command_pool_create_info);

  command_pool_create_info
    .setFlags(vk::CommandPoolCreateFlagBits::eResetCommandBuffer | vk::CommandPoolCreateFlagBits::eTransient)
    .setQueueFamilyIndex(queue_index_);
  transient_command_pool_ = device_.createCommandPool(command_pool_create_info);

  // Transfer fence
  transfer_fence_ = device_.createFence({});
}

void Engine::FreeMemory()
{
  device_.destroyFence(transfer_fence_);

  device_.destroyCommandPool(transient_command_pool_);
  device_.destroyCommandPool(command_pool_);

  device_.destroyDescriptorPool(descriptor_pool_);

  device_.unmapMemory(staging_buffer_.memory);
  device_.freeMemory(staging_buffer_.memory);
  device_.destroyBuffer(staging_buffer_.buffer);

  uniform_buffer_ = nullptr;

  device_.freeMemory(device_memory_);
  device_.freeMemory(host_memory_);
}

void Engine::AllocateCommandBuffers()
{
  vk::CommandBufferAllocateInfo command_buffer_allocate_info;
  command_buffer_allocate_info
    .setCommandPool(command_pool_)
    .setLevel(vk::CommandBufferLevel::ePrimary)
    .setCommandBufferCount(swapchain_image_count_);
  draw_command_buffers_ = device_.allocateCommandBuffers(command_buffer_allocate_info);

  command_buffer_allocate_info
    .setCommandPool(transient_command_pool_)
    .setLevel(vk::CommandBufferLevel::ePrimary)
    .setCommandBufferCount(1);
  transient_command_buffer_ = device_.allocateCommandBuffers(command_buffer_allocate_info)[0];
}

void Engine::FreeCommandBuffers()
{
  device_.freeCommandBuffers(command_pool_, draw_command_buffers_);
  draw_command_buffers_.clear();

  device_.freeCommandBuffers(transient_command_pool_, transient_command_buffer_);
}

void Engine::CreateRendertarget()
{
  // Color image
  vk::ImageCreateInfo image_create_info;
  image_create_info
    .setImageType(vk::ImageType::e2D)
    .setFormat(swapchain_image_format_)
    .setExtent(vk::Extent3D{ width_, height_, 1u })
    .setMipLevels(1)
    .setArrayLayers(1)
    .setSamples(vk::SampleCountFlagBits::e4)
    .setTiling(vk::ImageTiling::eOptimal)
    .setUsage(vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eTransientAttachment)
    .setSharingMode(vk::SharingMode::eExclusive)
    .setInitialLayout(vk::ImageLayout::eUndefined);
  rendertarget_.color_image = device_.createImage(image_create_info);
  device_.bindImageMemory(rendertarget_.color_image, rendertarget_.color_memory.memory, rendertarget_.color_memory.offset);

  vk::ImageSubresourceRange subresource_range;
  subresource_range
    .setAspectMask(vk::ImageAspectFlagBits::eColor)
    .setBaseArrayLayer(0)
    .setLayerCount(1)
    .setBaseMipLevel(0)
    .setLevelCount(1);

  vk::ImageViewCreateInfo image_view_create_info;
  image_view_create_info
    .setImage(rendertarget_.color_image)
    .setViewType(vk::ImageViewType::e2D)
    .setFormat(swapchain_image_format_)
    .setSubresourceRange(subresource_range);

  rendertarget_.color_image_view = device_.createImageView(image_view_create_info);

  // Depth image
  image_create_info
    .setImageType(vk::ImageType::e2D)
    .setFormat(vk::Format::eD24UnormS8Uint)
    .setExtent(vk::Extent3D{ width_, height_, 1u })
    .setMipLevels(1)
    .setArrayLayers(1)
    .setSamples(vk::SampleCountFlagBits::e4)
    .setTiling(vk::ImageTiling::eOptimal)
    .setUsage(vk::ImageUsageFlagBits::eDepthStencilAttachment | vk::ImageUsageFlagBits::eTransientAttachment)
    .setSharingMode(vk::SharingMode::eExclusive)
    .setInitialLayout(vk::ImageLayout::eUndefined);
  rendertarget_.depth_image = device_.createImage(image_create_info);
  device_.bindImageMemory(rendertarget_.depth_image, rendertarget_.depth_memory.memory, rendertarget_.depth_memory.offset);

  subresource_range
    .setAspectMask(vk::ImageAspectFlagBits::eDepth | vk::ImageAspectFlagBits::eStencil)
    .setBaseArrayLayer(0)
    .setLayerCount(1)
    .setBaseMipLevel(0)
    .setLevelCount(1);

  image_view_create_info
    .setImage(rendertarget_.depth_image)
    .setViewType(vk::ImageViewType::e2D)
    .setFormat(vk::Format::eD24UnormS8Uint)
    .setSubresourceRange(subresource_range);

  rendertarget_.depth_image_view = device_.createImageView(image_view_create_info);
}

void Engine::DestroyRendertarget()
{
  device_.destroyImageView(rendertarget_.color_image_view);
  device_.destroyImage(rendertarget_.color_image);

  device_.destroyImageView(rendertarget_.depth_image_view);
  device_.destroyImage(rendertarget_.depth_image);
}

void Engine::CreateSimulator()
{
  constexpr auto particleDimension = 40;
  constexpr auto particleCount = particleDimension * particleDimension * particleDimension;
  constexpr auto radius = 0.03f;
  constexpr float density = 1000.f; // water
  constexpr float pi = 3.141592f;
  const float mass = 4.f / 3.f * pi * radius * radius * radius * density;
  const float invMass = 1.f / mass;
  constexpr glm::vec2 wallDistance = glm::vec2(3.f, 1.5f);
  const glm::vec3 particleOffset = glm::vec3(-wallDistance + glm::vec2(radius * 1.1f), radius * 1.1f);
  const glm::vec3 particleStride = glm::vec3(radius); // Compressed at initial state

  utils::Rng rng;
  constexpr float noiseRange = 1e-2f;
  const auto noise = [&rng, noiseRange]() { return rng.Uniform(-noiseRange, noiseRange); };

  std::vector<vkpbd::Particle> particles;
  glm::vec3 gravity = glm::vec3(0.f, 0.f, -9.8f);
  for (int i = 0; i < particleDimension; i++)
  {
    for (int j = 0; j < particleDimension; j++)
    {
      for (int k = 0; k < particleDimension; k++)
      {
        glm::vec4 position{
          particleOffset.x + particleStride.x * i + noise(),
          particleOffset.y + particleStride.y * j + noise(),
          particleOffset.z + particleStride.z * k + noise(),
          0.f
        };
        glm::vec4 velocity{ 0.f };
        glm::vec4 properties{ invMass, mass, 0.f, 0.f };
        glm::vec4 externalForce{
          gravity.x * mass,
          gravity.y * mass,
          gravity.z * mass,
          0.f
        };
        glm::vec4 color{ 0.5f, 0.5f, 0.5f, 0.f };

        // Struct initialization
        particles.push_back({ position, velocity, properties, externalForce, color });
      }
    }
  }

  vkpbd::FluidSimulatorCreateInfo fluidSimulatorCreateInfo;
  fluidSimulatorCreateInfo.device = device_;
  fluidSimulatorCreateInfo.physicalDevice = physical_device_;
  fluidSimulatorCreateInfo.descriptorPool = descriptor_pool_;
  fluidSimulatorCreateInfo.particleCount = particleCount;
  fluidSimulatorCreateInfo.maxNeighborCount = 30;
  fluidSimulatorCreateInfo.commandCount = commandCount;
  fluidSimulator_ = vkpbd::createFluidSimulator(fluidSimulatorCreateInfo);

  // Create buffers
  const auto particleBufferRequirements = fluidSimulator_.getParticleBufferRequirements();
  const auto internalBufferRequirements = fluidSimulator_.getInternalBufferRequirements();
  const auto uniformBufferRequirements = fluidSimulator_.getUniformBufferRequirements();

  vk::BufferCreateInfo bufferCreateInfo;
  bufferCreateInfo
    .setUsage(particleBufferRequirements.usage | vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eVertexBuffer)
    .setSize(particleBufferRequirements.size * commandCount);
  particleBuffer_ = device_.createBuffer(bufferCreateInfo);
  particleBufferSize_ = particleBufferRequirements.size;

  bufferCreateInfo
    .setUsage(internalBufferRequirements.usage)
    .setSize(internalBufferRequirements.size);
  particleInternalBuffer_ = device_.createBuffer(bufferCreateInfo);
  particleInternalBufferSize_ = internalBufferRequirements.size;

  bufferCreateInfo
    .setUsage(uniformBufferRequirements.usage)
    .setSize(uniformBufferRequirements.size);
  particleUniformBuffer_ = device_.createBuffer(bufferCreateInfo);
  particleUniformBufferSize_ = uniformBufferRequirements.size;

  // Bind to memory
  const auto particleMemory = AcquireDeviceMemory(particleBuffer_);
  device_.bindBufferMemory(particleBuffer_, particleMemory.memory, particleMemory.offset);

  const auto particleInternalMemory = AcquireDeviceMemory(particleInternalBuffer_);
  device_.bindBufferMemory(particleInternalBuffer_, particleInternalMemory.memory, particleInternalMemory.offset);

  // Persistently mapped uniform buffer
  vk::MemoryAllocateInfo memoryAllocateInfo;
  memoryAllocateInfo
    .setAllocationSize(device_.getBufferMemoryRequirements(particleUniformBuffer_).size)
    .setMemoryTypeIndex(host_index_);
  particleUniformMemory_ = device_.allocateMemory(memoryAllocateInfo);
  particleUniformBufferMap_ = reinterpret_cast<uint8_t*>(device_.mapMemory(particleUniformMemory_, 0, uniformBufferRequirements.size));
  device_.bindBufferMemory(particleUniformBuffer_, particleUniformMemory_, 0);

  // To staging buffer and copy
  ToDeviceMemory(particles, particleBuffer_, 0);
}

void Engine::DestroySimulator()
{
  device_.destroyBuffer(particleBuffer_);
  device_.destroyBuffer(particleInternalBuffer_);
  device_.destroyBuffer(particleUniformBuffer_);
  device_.unmapMemory(particleUniformMemory_);
  device_.freeMemory(particleUniformMemory_);

  fluidSimulator_.destroy();
}

void Engine::CreateSynchronizationObjects()
{
  for (int i = 0; i < 2; i++)
  {
    image_available_semaphores_.emplace_back(device_.createSemaphore({}));
    render_finished_semaphores_.emplace_back(device_.createSemaphore({}));
    in_flight_fences_.emplace_back(device_.createFence({ vk::FenceCreateFlagBits::eSignaled }));
  }

  // Pointer to fence
  images_in_flight_.resize(swapchain_image_count_);
}

void Engine::DestroySynchronizationObjects()
{
  for (auto& semaphore : image_available_semaphores_)
    device_.destroySemaphore(semaphore);
  image_available_semaphores_.clear();

  for (auto& semaphore : render_finished_semaphores_)
    device_.destroySemaphore(semaphore);
  render_finished_semaphores_.clear();

  for (auto& fence : in_flight_fences_)
    device_.destroyFence(fence);
  in_flight_fences_.clear();
}

vk::ShaderModule Engine::CreateShaderModule(const std::string& filepath)
{
  std::ifstream file(filepath, std::ios::ate | std::ios::binary);
  if (!file.is_open())
    throw std::runtime_error("Failed to open file: " + filepath);

  size_t file_size = (size_t)file.tellg();
  std::vector<char> buffer(file_size);
  file.seekg(0);
  file.read(buffer.data(), file_size);
  file.close();

  std::vector<uint32_t> code;
  auto* int_ptr = reinterpret_cast<uint32_t*>(buffer.data());
  for (int i = 0; i < file_size / 4; i++)
    code.push_back(int_ptr[i]);

  vk::ShaderModuleCreateInfo shader_module_create_info;
  shader_module_create_info
    .setCode(code);
  return device_.createShaderModule(shader_module_create_info);
}

void Engine::ImageLayoutTransition(vk::CommandBuffer& command_buffer, vk::Image image, vk::ImageLayout old_layout, vk::ImageLayout new_layout, uint32_t mipmap_levels)
{
  vk::PipelineStageFlags src_stage_mask = {};
  vk::AccessFlags src_access_mask = {};
  vk::PipelineStageFlags dst_stage_mask = {};
  vk::AccessFlags dst_access_mask = {};

  if (old_layout == vk::ImageLayout::eUndefined && new_layout == vk::ImageLayout::eTransferDstOptimal)
  {
    src_stage_mask = vk::PipelineStageFlagBits::eTopOfPipe;
    dst_stage_mask = vk::PipelineStageFlagBits::eTransfer;
    dst_access_mask = vk::AccessFlagBits::eTransferWrite;
  }
  else if (old_layout == vk::ImageLayout::eTransferDstOptimal && new_layout == vk::ImageLayout::eShaderReadOnlyOptimal)
  {
    src_stage_mask = vk::PipelineStageFlagBits::eTransfer;
    src_access_mask = vk::AccessFlagBits::eTransferWrite;
    dst_stage_mask = vk::PipelineStageFlagBits::eFragmentShader;
    dst_access_mask = vk::AccessFlagBits::eShaderRead;
  }

  vk::ImageSubresourceRange subresource_range;
  subresource_range
    .setAspectMask(vk::ImageAspectFlagBits::eColor)
    .setBaseMipLevel(0)
    .setLevelCount(mipmap_levels)
    .setBaseArrayLayer(0)
    .setLayerCount(1);

  vk::ImageMemoryBarrier image_memory_barrier;
  image_memory_barrier
    .setSrcAccessMask(src_access_mask)
    .setDstAccessMask(dst_access_mask)
    .setOldLayout(old_layout)
    .setNewLayout(new_layout)
    .setSrcQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
    .setDstQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
    .setImage(image)
    .setSubresourceRange(subresource_range);

  command_buffer.pipelineBarrier(src_stage_mask, dst_stage_mask, {},
    {},
    {},
    image_memory_barrier);
}

void Engine::GenerateMipmap(vk::CommandBuffer& command_buffer, vk::Image image, uint32_t width, uint32_t height, uint32_t mipmap_levels)
{
  for (uint32_t i = 0; i < mipmap_levels - 1; i++)
  {
    // Layout transition from transfer dst to transfer src
    vk::ImageSubresourceRange subresource_range;
    subresource_range
      .setAspectMask(vk::ImageAspectFlagBits::eColor)
      .setBaseMipLevel(i)
      .setLevelCount(1)
      .setBaseArrayLayer(0)
      .setLayerCount(1);

    vk::ImageMemoryBarrier image_memory_barrier;
    image_memory_barrier
      .setSrcAccessMask(vk::AccessFlagBits::eTransferWrite)
      .setDstAccessMask(vk::AccessFlagBits::eTransferRead)
      .setOldLayout(vk::ImageLayout::eTransferDstOptimal)
      .setNewLayout(vk::ImageLayout::eTransferSrcOptimal)
      .setSrcQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
      .setDstQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
      .setImage(image)
      .setSubresourceRange(subresource_range);

    command_buffer.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eTransfer, {},
      {},
      {},
      image_memory_barrier);

    // Image blit
    vk::ImageSubresourceLayers src_subresource;
    src_subresource
      .setAspectMask(vk::ImageAspectFlagBits::eColor)
      .setMipLevel(i)
      .setBaseArrayLayer(0)
      .setLayerCount(1);

    vk::Offset3D src_offset{ 0, 0, 0 };
    vk::Offset3D src_extent{ static_cast<int32_t>(width), static_cast<int32_t>(height), 1 };

    vk::ImageSubresourceLayers dst_subresource;
    dst_subresource
      .setAspectMask(vk::ImageAspectFlagBits::eColor)
      .setMipLevel(i + 1)
      .setBaseArrayLayer(0)
      .setLayerCount(1);

    vk::Offset3D dst_offset{ 0, 0, 0 };
    vk::Offset3D dst_extent{ static_cast<int32_t>(width / 2), static_cast<int32_t>(height / 2), 1 };

    vk::ImageBlit image_blit;
    image_blit
      .setSrcSubresource(src_subresource)
      .setSrcOffsets({ src_offset, src_extent })
      .setDstSubresource(dst_subresource)
      .setDstOffsets({ dst_offset, dst_extent });

    command_buffer.blitImage(
      image, vk::ImageLayout::eTransferSrcOptimal,
      image, vk::ImageLayout::eTransferDstOptimal,
      image_blit, vk::Filter::eLinear);

    // Layout transition from transfer dst to shader read optimal
    image_memory_barrier
      .setSrcAccessMask(vk::AccessFlagBits::eTransferRead)
      .setDstAccessMask(vk::AccessFlagBits::eShaderRead)
      .setOldLayout(vk::ImageLayout::eTransferSrcOptimal)
      .setNewLayout(vk::ImageLayout::eShaderReadOnlyOptimal)
      .setSrcQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
      .setDstQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
      .setImage(image)
      .setSubresourceRange(subresource_range);

    command_buffer.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eFragmentShader, {},
      {},
      {},
      image_memory_barrier);

    // Half image size
    width /= 2;
    height /= 2;
  }

  // Layout transition of last mipmap level
  vk::ImageSubresourceRange subresource_range;
  subresource_range
    .setAspectMask(vk::ImageAspectFlagBits::eColor)
    .setBaseMipLevel(mipmap_levels - 1)
    .setLevelCount(1)
    .setBaseArrayLayer(0)
    .setLayerCount(1);

  vk::ImageMemoryBarrier image_memory_barrier;
  image_memory_barrier
    .setSrcAccessMask(vk::AccessFlagBits::eTransferWrite)
    .setDstAccessMask(vk::AccessFlagBits::eShaderRead)
    .setOldLayout(vk::ImageLayout::eTransferDstOptimal)
    .setNewLayout(vk::ImageLayout::eShaderReadOnlyOptimal)
    .setSrcQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
    .setDstQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
    .setImage(image)
    .setSubresourceRange(subresource_range);

  command_buffer.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eFragmentShader, {},
    {},
    {},
    image_memory_barrier);
}
}
}
