#include <superlucent/engine.h>

#include <iostream>
#include <fstream>

#include <GLFW/glfw3.h>

#include <superlucent/scene/camera.h>

namespace supl
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

constexpr auto align(vk::DeviceSize offset, vk::DeviceSize alignment)
{
  return (offset + alignment - 1) & ~(alignment - 1);
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
  CreateFramebuffer();
  CreatePipelines();
  PrepareResources();
  CreateSynchronizationObjects();

  // Initialize uniform values
  triangle_model_.model = glm::mat4(1.f);
  triangle_model_.model[3][2] = 1.f;
  triangle_model_.model_inverse_transpose = glm::mat3(1.f);
}

Engine::~Engine()
{
  device_.waitIdle();

  DestroySynchronizationObjects();
  DestroyResources();
  DestroyPipelines();
  DestroyFramebuffer();
  DestroyRendertarget();
  FreeCommandBuffers();
  FreeMemory();
  DestroySwapchain();
  DestroyDevice();
  DestroyInstance();
}

void Engine::Resize(uint32_t width, uint32_t height)
{
  // TODO
}

void Engine::UpdateCamera(std::shared_ptr<scene::Camera> camera)
{
  camera_.view = camera->ViewMatrix();

  camera_.projection = camera->ProjectionMatrix();
  camera_.projection[1][1] *= -1.f;

  camera_.eye = camera->Eye();
}

void Engine::Draw()
{
  auto wait_result = device_.waitForFences(in_flight_fences_[current_frame_], true, UINT64_MAX);

  const auto acquire_next_image_result = device_.acquireNextImageKHR(swapchain_, UINT64_MAX, image_available_semaphores_[current_frame_]);
  if (acquire_next_image_result.result == vk::Result::eErrorOutOfDateKHR)
  {
    // TODO: recreate swapchain
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
  RecordDrawCommands(draw_command_buffer, image_index);
  draw_command_buffer.end();

  // Update uniforms
  std::memcpy(uniform_buffer_.map + camera_ubos_[image_index].offset, &camera_, sizeof(CameraUbo));
  std::memcpy(uniform_buffer_.map + triangle_model_ubos_[image_index].offset, &triangle_model_, sizeof(ModelUbo));

  // Submit
  std::vector<vk::PipelineStageFlags> stages{
    vk::PipelineStageFlagBits::eColorAttachmentOutput
  };
  queue_.submit({
    { image_available_semaphores_[current_frame_], stages, draw_command_buffer, render_finished_semaphores_[current_frame_] },
    }, in_flight_fences_[current_frame_]);

  // Present
  std::vector<uint32_t> image_indices{ image_index };
  const auto present_result = present_queue_.presentKHR(
    { render_finished_semaphores_[current_frame_], swapchain_, image_indices });

  if (present_result == vk::Result::eErrorOutOfDateKHR || present_result == vk::Result::eSuboptimalKHR)
  {
    // TODO: Recreate swapchain
  }
  else if (present_result != vk::Result::eSuccess)
    throw std::runtime_error("Failed to present swapchain image");

  current_frame_ = (current_frame_ + 1) % 2;
}

void Engine::RecordDrawCommands(vk::CommandBuffer& command_buffer, uint32_t image_index)
{
  vk::Viewport viewport{ 0.f, 0.f, static_cast<float>(width_), static_cast<float>(height_), 0.f, 1.f };
  command_buffer.setViewport(0, viewport);

  command_buffer.setScissor(0, vk::Rect2D{ {0u, 0u}, {width_, height_} });

  std::vector<vk::ClearValue> clear_values{
    vk::ClearColorValue{ std::array<float, 4>{0.8f, 0.8f, 0.8f, 1.f} },
    vk::ClearDepthStencilValue{ 1.f, 0u }
  };
  command_buffer.beginRenderPass({ render_pass_, swapchain_framebuffers_[image_index],
    vk::Rect2D{ {0u, 0u}, {width_, height_} }, clear_values
    }, vk::SubpassContents::eInline
  );

  // Draw triangle model
  command_buffer.bindPipeline(vk::PipelineBindPoint::eGraphics, color_pipeline_);

  command_buffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, graphics_pipeline_layout_, 0u,
    graphics_descriptor_sets_[image_index], {});

  command_buffer.bindVertexBuffers(0u, { triangle_buffer_.buffer }, { 0ull });

  command_buffer.bindIndexBuffer(triangle_buffer_.buffer, triangle_buffer_.index_offset, vk::IndexType::eUint32);

  command_buffer.drawIndexed(triangle_buffer_.num_indices, 1u, 0u, 0u, 0u);

  // Draw floor model
  command_buffer.bindPipeline(vk::PipelineBindPoint::eGraphics, floor_pipeline_);

  command_buffer.bindVertexBuffers(0u, { floor_buffer_.buffer }, { 0ull });

  command_buffer.bindIndexBuffer(floor_buffer_.buffer, floor_buffer_.index_offset, vk::IndexType::eUint32);

  command_buffer.drawIndexed(floor_buffer_.num_indices, 1u, 0u, 0u, 0u);

  command_buffer.endRenderPass();
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
  memory.offset = align(device_offset_, memory_requirements.alignment);
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
  memory.offset = align(device_offset_, memory_requirements.alignment);
  memory.size = memory_requirements.size;
  device_offset_ = memory.offset + memory.size;
  return memory;
}

void Engine::CreateInstance(GLFWwindow* window)
{
  // App
  vk::ApplicationInfo app_info{
    "Superlucent", VK_MAKE_VERSION(1, 0, 0),
    "Superlucent Engine", VK_MAKE_VERSION(1, 0, 0), VK_API_VERSION_1_2
  };

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
  vk::StructureChain<vk::InstanceCreateInfo, vk::DebugUtilsMessengerCreateInfoEXT> chain{
    { {}, &app_info, layers, extensions },
    { {},
    vk::DebugUtilsMessageSeverityFlagBitsEXT::eError | vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning | vk::DebugUtilsMessageSeverityFlagBitsEXT::eVerbose,
    vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral | vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation | vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance,
    debug_callback
    }
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

  // Device properties
  ubo_alignment_ = physical_device_.getProperties().limits.minUniformBufferOffsetAlignment;

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
    .setGeometryShader(true);

  // Create device
  vk::DeviceCreateInfo device_create_info{ {},
    queue_create_info, {}, extensions, &features
  };
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
  swapchain_ = device_.createSwapchainKHR({ {},
    surface_, image_count,
    format.format, format.colorSpace,
    extent, 1, vk::ImageUsageFlagBits::eColorAttachment,
    vk::SharingMode::eExclusive, {}, capabilities.currentTransform,
    vk::CompositeAlphaFlagBitsKHR::eOpaque,
    present_mode, true, nullptr });
  swapchain_image_format_ = format.format;
  swapchain_image_count_ = image_count;

  swapchain_images_ = device_.getSwapchainImagesKHR(swapchain_);

  // Create image view for swapchain
  swapchain_image_views_.resize(swapchain_images_.size());
  for (int i = 0; i < swapchain_images_.size(); i++)
  {
    swapchain_image_views_[i] = device_.createImageView({
      {}, swapchain_images_[i], vk::ImageViewType::e2D, swapchain_image_format_, {},
      {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1} });
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
  uint32_t host_index = 0;

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
        host_index = i;
        host_available_size = heap.size;
      }
    }
  }

  constexpr uint64_t chunk_size = 256 * 1024 * 1024; // 256MB
  device_memory_ = device_.allocateMemory({ chunk_size, device_index });
  host_memory_ = device_.allocateMemory({ chunk_size, host_index });

  // Persistently mapped staging buffer
  staging_buffer_.buffer = device_.createBuffer({ {},
    staging_buffer_.size, vk::BufferUsageFlagBits::eTransferSrc });
  staging_buffer_.memory = device_.allocateMemory({
    device_.getBufferMemoryRequirements(staging_buffer_.buffer).size,
    host_index });
  device_.bindBufferMemory(staging_buffer_.buffer, staging_buffer_.memory, 0);
  staging_buffer_.map = static_cast<uint8_t*>(device_.mapMemory(staging_buffer_.memory, 0, staging_buffer_.size));

  // Persistently mapped uniform buffer
  uniform_buffer_.buffer = device_.createBuffer({ {},
    uniform_buffer_.size, vk::BufferUsageFlagBits::eUniformBuffer });
  uniform_buffer_.memory = device_.allocateMemory({
    device_.getBufferMemoryRequirements(uniform_buffer_.buffer).size,
    host_index });
  device_.bindBufferMemory(uniform_buffer_.buffer, uniform_buffer_.memory, 0);
  uniform_buffer_.map = static_cast<uint8_t*>(device_.mapMemory(uniform_buffer_.memory, 0, uniform_buffer_.size));

  // Preallocate framebuffer memory
  auto temp_color_image = device_.createImage({ {},
    vk::ImageType::e2D, swapchain_image_format_, {max_width_, max_height_, 1}, 1, 1,
    vk::SampleCountFlagBits::e4, vk::ImageTiling::eOptimal,
    vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eTransientAttachment,
    vk::SharingMode::eExclusive, {},
    vk::ImageLayout::eUndefined
    });
  rendertarget_.color_memory = AcquireDeviceMemory(temp_color_image);
  device_.destroyImage(temp_color_image);

  auto temp_depth_image = device_.createImage({ {},
    vk::ImageType::e2D, vk::Format::eD24UnormS8Uint, {max_width_, max_height_, 1}, 1, 1,
    vk::SampleCountFlagBits::e4, vk::ImageTiling::eOptimal,
    vk::ImageUsageFlagBits::eDepthStencilAttachment | vk::ImageUsageFlagBits::eTransientAttachment,
    vk::SharingMode::eExclusive, {},
    vk::ImageLayout::eUndefined
    });
  rendertarget_.depth_memory = AcquireDeviceMemory(temp_depth_image);
  device_.destroyImage(temp_depth_image);

  // Preallocate descriptor pool
  constexpr uint32_t max_num_descriptors = 1024;
  constexpr uint32_t max_sets = 1024;
  std::vector<vk::DescriptorPoolSize> pool_sizes{
    { vk::DescriptorType::eUniformBuffer, max_num_descriptors },
    { vk::DescriptorType::eUniformBufferDynamic, max_num_descriptors },
  };
  descriptor_pool_ = device_.createDescriptorPool({ vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet, max_sets, pool_sizes });

  // Preallocate command pools
  command_pool_ = device_.createCommandPool({ vk::CommandPoolCreateFlagBits::eResetCommandBuffer, queue_index_ });
  transient_command_pool_ = device_.createCommandPool({ 
    vk::CommandPoolCreateFlagBits::eResetCommandBuffer | vk::CommandPoolCreateFlagBits::eTransient, queue_index_ });

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

  device_.unmapMemory(uniform_buffer_.memory);
  device_.freeMemory(uniform_buffer_.memory);
  device_.destroyBuffer(uniform_buffer_.buffer);

  device_.freeMemory(device_memory_);
  device_.freeMemory(host_memory_);
}

void Engine::AllocateCommandBuffers()
{
  draw_command_buffers_ = device_.allocateCommandBuffers({ command_pool_, vk::CommandBufferLevel::ePrimary, swapchain_image_count_ });
  transient_command_buffer_ = device_.allocateCommandBuffers({ transient_command_pool_, vk::CommandBufferLevel::ePrimary, 1 })[0];
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
  rendertarget_.color_image = device_.createImage({ {},
    vk::ImageType::e2D, swapchain_image_format_, {width_, height_, 1u}, 1, 1,
    vk::SampleCountFlagBits::e4, vk::ImageTiling::eOptimal,
    vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eTransientAttachment,
    vk::SharingMode::eExclusive, {},
    vk::ImageLayout::eUndefined });
  device_.bindImageMemory(rendertarget_.color_image, rendertarget_.color_memory.memory, rendertarget_.color_memory.offset);
  rendertarget_.color_image_view = device_.createImageView({
    {}, rendertarget_.color_image, vk::ImageViewType::e2D, swapchain_image_format_,
    {}, vk::ImageSubresourceRange{vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1} });

  // Depth image
  rendertarget_.depth_image = device_.createImage({ {},
    vk::ImageType::e2D, vk::Format::eD24UnormS8Uint, {width_, height_, 1u}, 1, 1,
    vk::SampleCountFlagBits::e4, vk::ImageTiling::eOptimal,
    vk::ImageUsageFlagBits::eDepthStencilAttachment | vk::ImageUsageFlagBits::eTransientAttachment,
    vk::SharingMode::eExclusive, {},
    vk::ImageLayout::eUndefined
    });
  device_.bindImageMemory(rendertarget_.depth_image, rendertarget_.depth_memory.memory, rendertarget_.depth_memory.offset);
  rendertarget_.depth_image_view = device_.createImageView({
    {}, rendertarget_.depth_image, vk::ImageViewType::e2D, vk::Format::eD24UnormS8Uint,
    {}, vk::ImageSubresourceRange{vk::ImageAspectFlagBits::eDepth | vk::ImageAspectFlagBits::eStencil, 0, 1, 0, 1} });
}

void Engine::DestroyRendertarget()
{
  device_.destroyImageView(rendertarget_.color_image_view);
  device_.destroyImage(rendertarget_.color_image);

  device_.destroyImageView(rendertarget_.depth_image_view);
  device_.destroyImage(rendertarget_.depth_image);
}

void Engine::CreateFramebuffer()
{
  // Render pass
  std::vector<vk::AttachmentDescription> attachment_descriptions{
    // Color
    { {}, swapchain_image_format_, vk::SampleCountFlagBits::e4,
    vk::AttachmentLoadOp::eClear, vk::AttachmentStoreOp::eStore,
    vk::AttachmentLoadOp::eDontCare, vk::AttachmentStoreOp::eDontCare,
    vk::ImageLayout::eUndefined, vk::ImageLayout::eColorAttachmentOptimal },
    // Depth
    { {}, vk::Format::eD24UnormS8Uint, vk::SampleCountFlagBits::e4,
    vk::AttachmentLoadOp::eClear, vk::AttachmentStoreOp::eDontCare,
    vk::AttachmentLoadOp::eDontCare, vk::AttachmentStoreOp::eDontCare,
    vk::ImageLayout::eUndefined, vk::ImageLayout::eDepthStencilAttachmentOptimal },
    // Resolve
    { {}, swapchain_image_format_, vk::SampleCountFlagBits::e1,
    vk::AttachmentLoadOp::eDontCare, vk::AttachmentStoreOp::eStore,
    vk::AttachmentLoadOp::eDontCare, vk::AttachmentStoreOp::eDontCare,
    vk::ImageLayout::eUndefined, vk::ImageLayout::ePresentSrcKHR },
  };

  vk::AttachmentReference color_attachment_reference{ 0, vk::ImageLayout::eColorAttachmentOptimal };
  vk::AttachmentReference depth_attachment_reference{ 1, vk::ImageLayout::eDepthStencilAttachmentOptimal };
  vk::AttachmentReference resolve_attachment_reference{ 2, vk::ImageLayout::eColorAttachmentOptimal };

  std::vector<vk::SubpassDescription> subpasses{
    { {}, vk::PipelineBindPoint::eGraphics,
    {}, color_attachment_reference, resolve_attachment_reference, &depth_attachment_reference },
  };

  std::vector<vk::SubpassDependency> dependencies{
    { VK_SUBPASS_EXTERNAL, 0,
    vk::PipelineStageFlagBits::eColorAttachmentOutput | vk::PipelineStageFlagBits::eEarlyFragmentTests,
    vk::PipelineStageFlagBits::eColorAttachmentOutput | vk::PipelineStageFlagBits::eEarlyFragmentTests,
    {}, vk::AccessFlagBits::eColorAttachmentWrite | vk::AccessFlagBits::eDepthStencilAttachmentWrite },
  };

  render_pass_ = device_.createRenderPass({ {}, attachment_descriptions, subpasses, dependencies });

  // Framebuffer
  swapchain_framebuffers_.resize(swapchain_image_count_);
  for (uint32_t i = 0; i < swapchain_image_count_; i++)
  {
    std::vector<vk::ImageView> attachments{
      rendertarget_.color_image_view,
      rendertarget_.depth_image_view,
      swapchain_image_views_[i],
    };

    swapchain_framebuffers_[i] = device_.createFramebuffer({ {}, render_pass_, attachments, width_, height_, 1 });
  }
}

void Engine::DestroyFramebuffer()
{
  for (auto& framebuffer : swapchain_framebuffers_)
    device_.destroyFramebuffer(framebuffer);
  swapchain_framebuffers_.clear();

  device_.destroyRenderPass(render_pass_);
}

void Engine::CreatePipelines()
{
  // Create pipeline cache
  pipeline_cache_ = device_.createPipelineCache({});

  CreateGraphicsPipelines();
}

void Engine::DestroyPipelines()
{
  DestroyGraphicsPipelines();

  device_.destroyPipelineCache(pipeline_cache_);
}

void Engine::CreateGraphicsPipelines()
{
  // Descriptor set layout
  std::vector<vk::DescriptorSetLayoutBinding> descriptor_set_layout_bindings{
    { 0u, vk::DescriptorType::eUniformBuffer, 1, vk::ShaderStageFlagBits::eVertex },
    { 1u, vk::DescriptorType::eUniformBuffer, 1, vk::ShaderStageFlagBits::eVertex },
  };
  graphics_descriptor_set_layout_ = device_.createDescriptorSetLayout({ {}, descriptor_set_layout_bindings });

  // Pipeline layout
  graphics_pipeline_layout_ = device_.createPipelineLayout({ {}, graphics_descriptor_set_layout_, {} });

  // Shader modules
  const std::string base_dir = "C:\\workspace\\superlucent\\src\\superlucent\\shader";
  vk::ShaderModule vert_module = CreateShaderModule(base_dir + "\\color.vert.spv");
  vk::ShaderModule frag_module = CreateShaderModule(base_dir + "\\color.frag.spv");

  // Shader stages
  std::vector<vk::PipelineShaderStageCreateInfo> shader_stages{
    { {}, vk::ShaderStageFlagBits::eVertex, vert_module, "main" },
    { {}, vk::ShaderStageFlagBits::eFragment, frag_module, "main" },
  };

  // Vertex input
  std::vector<vk::VertexInputBindingDescription> vertex_binding_descriptions{
    { 0, sizeof(float) * 6, vk::VertexInputRate::eVertex },
  };
  std::vector<vk::VertexInputAttributeDescription> vertex_attribute_descriptions{
    { 0, 0, vk::Format::eR32G32B32Sfloat, 0 },
    { 1, 0, vk::Format::eR32G32B32Sfloat, sizeof(float) * 3 },
  };
  vk::PipelineVertexInputStateCreateInfo vertex_input{ {},
    vertex_binding_descriptions, vertex_attribute_descriptions };

  // Input assembly
  vk::PipelineInputAssemblyStateCreateInfo input_assembly{ {},
    vk::PrimitiveTopology::eTriangleList, false
  };

  // Viewport
  vk::Viewport viewport_area{
    0.f, 0.f,
    static_cast<float>(width_), static_cast<float>(height_),
    0.f, 1.f
  };
  vk::Rect2D scissor{ { 0u, 0u }, { width_, height_ } };
  vk::PipelineViewportStateCreateInfo viewport{ {}, viewport_area, scissor };

  // Rasterization
  vk::PipelineRasterizationStateCreateInfo rasterization{ {},
    false, false, vk::PolygonMode::eFill, vk::CullModeFlagBits::eNone, vk::FrontFace::eCounterClockwise,
    false, 0.f, 0.f, 0.f,
    1.f
  };

  // Multisample
  vk::PipelineMultisampleStateCreateInfo multisample{ {}, vk::SampleCountFlagBits::e4 };

  // Depth stencil
  vk::PipelineDepthStencilStateCreateInfo depth_stencil{ {},
    true, true, vk::CompareOp::eLess, false,
    false, {}, {},
    0.f, 1.f
  };

  // Color blend
  vk::PipelineColorBlendAttachmentState color_blend_attachment{
    false,
    vk::BlendFactor::eSrcAlpha, vk::BlendFactor::eOneMinusSrcAlpha, vk::BlendOp::eAdd,
    vk::BlendFactor::eOne, vk::BlendFactor::eZero, vk::BlendOp::eAdd,
    vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA
  };
  vk::PipelineColorBlendStateCreateInfo color_blend{ {},
    false, {}, color_blend_attachment,
    { 0.f, 0.f, 0.f, 0.f }
  };

  // Dynamic states
  std::vector<vk::DynamicState> dynamic_states{
    vk::DynamicState::eViewport,
    vk::DynamicState::eScissor,
  };
  vk::PipelineDynamicStateCreateInfo dynamic_state{ {}, dynamic_states };
 
  vk::GraphicsPipelineCreateInfo pipeline_create_info{ {}, shader_stages,
    &vertex_input, &input_assembly, nullptr,
    &viewport, &rasterization, &multisample, &depth_stencil, &color_blend, &dynamic_state,
    graphics_pipeline_layout_,
    render_pass_, 0,
    nullptr, -1
  };

  color_pipeline_ = CreateGraphicsPipeline(pipeline_create_info);
  device_.destroyShaderModule(vert_module);
  device_.destroyShaderModule(frag_module);

  // Floor graphics pipelineshader_stages
  vert_module = CreateShaderModule(base_dir + "\\floor.vert.spv");
  frag_module = CreateShaderModule(base_dir + "\\floor.frag.spv");

  shader_stages = {
    { {}, vk::ShaderStageFlagBits::eVertex, vert_module, "main" },
    { {}, vk::ShaderStageFlagBits::eFragment, frag_module, "main" },
  };

  vertex_binding_descriptions = {
    { 0u, sizeof(float) * 2, vk::VertexInputRate::eVertex },
  };
  vertex_attribute_descriptions = {
    { 0u, 0u, vk::Format::eR32G32B32Sfloat, 0u },
  };
  vertex_input
    .setVertexAttributeDescriptions(vertex_attribute_descriptions)
    .setVertexBindingDescriptions(vertex_binding_descriptions);

  input_assembly.setTopology(vk::PrimitiveTopology::eTriangleStrip);

  pipeline_create_info.setStages(shader_stages);

  floor_pipeline_ = CreateGraphicsPipeline(pipeline_create_info);
  device_.destroyShaderModule(vert_module);
  device_.destroyShaderModule(frag_module);
}

void Engine::DestroyGraphicsPipelines()
{
  device_.destroyDescriptorSetLayout(graphics_descriptor_set_layout_);
  device_.destroyPipeline(color_pipeline_);
  device_.destroyPipeline(floor_pipeline_);
  device_.destroyPipelineLayout(graphics_pipeline_layout_);
}

void Engine::PrepareResources()
{
  // Triangle vertex buffer
  std::vector<float> triangle_vertex_buffer{
    0.f, 0.f, 0.f, 1.f, 0.f, 0.f,
    1.f, 0.f, 0.f, 0.f, 1.f, 0.f,
    0.f, 1.f, 0.f, 0.f, 0.f, 1.f,
  };
  std::vector<uint32_t> triangle_index_buffer{
    0, 1, 2
  };
  const auto triangle_vertex_buffer_size = triangle_vertex_buffer.size() * sizeof(float);
  const auto triangle_index_buffer_size = triangle_index_buffer.size() * sizeof(uint32_t);
  const auto triangle_buffer_size = triangle_vertex_buffer_size + triangle_index_buffer_size;

  triangle_buffer_.buffer = device_.createBuffer({ {},
    triangle_buffer_size, vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eIndexBuffer
    });
  triangle_buffer_.index_offset = triangle_vertex_buffer_size;
  triangle_buffer_.num_indices = static_cast<uint32_t>(triangle_index_buffer.size());

  // Floor buffer
  constexpr float floor_range = 20.f;
  std::vector<float> floor_vertex_buffer{
    -floor_range, -floor_range,
    floor_range, -floor_range,
    -floor_range, floor_range,
    floor_range, floor_range,
  };
  std::vector<uint32_t> floor_index_buffer{
    0, 1, 2, 3
  };

  const auto floor_vertex_buffer_size = floor_vertex_buffer.size() * sizeof(float);
  const auto floor_index_buffer_size = floor_index_buffer.size() * sizeof(uint32_t);
  const auto floor_buffer_size = floor_vertex_buffer_size + floor_index_buffer_size;

  floor_buffer_.buffer = device_.createBuffer({ {},
    floor_buffer_size, vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eIndexBuffer
    });
  floor_buffer_.index_offset = floor_vertex_buffer_size;
  floor_buffer_.num_indices = static_cast<uint32_t>(floor_index_buffer.size());

  // Memory binding
  const auto triangle_memory = AcquireDeviceMemory(triangle_buffer_.buffer);
  device_.bindBufferMemory(triangle_buffer_.buffer, triangle_memory.memory, triangle_memory.offset);

  const auto floor_memory = AcquireDeviceMemory(floor_buffer_.buffer);
  device_.bindBufferMemory(floor_buffer_.buffer, floor_memory.memory, floor_memory.offset);

  // Transfer
  vk::DeviceSize staging_offset = 0ull;
  std::memcpy(staging_buffer_.map + staging_offset, triangle_vertex_buffer.data(), triangle_vertex_buffer_size);
  staging_offset += triangle_vertex_buffer_size;
  std::memcpy(staging_buffer_.map + staging_offset, triangle_index_buffer.data(), triangle_index_buffer_size);
  staging_offset += triangle_index_buffer_size;

  std::memcpy(staging_buffer_.map + staging_offset, floor_vertex_buffer.data(), floor_vertex_buffer_size);
  staging_offset += floor_vertex_buffer_size;
  std::memcpy(staging_buffer_.map + staging_offset, floor_index_buffer.data(), floor_index_buffer_size);
  staging_offset += floor_index_buffer_size;

  transient_command_buffer_.begin({ vk::CommandBufferUsageFlagBits::eOneTimeSubmit });
  transient_command_buffer_.copyBuffer(staging_buffer_.buffer, triangle_buffer_.buffer, {
    { 0ull, 0ull, triangle_buffer_size },
    });
  transient_command_buffer_.copyBuffer(staging_buffer_.buffer, floor_buffer_.buffer, {
    { triangle_buffer_size, 0ull, floor_buffer_size },
    });
  transient_command_buffer_.end();

  queue_.submit({ { {}, {}, transient_command_buffer_, {} } }, transfer_fence_);
  const auto wait_result = device_.waitForFences(transfer_fence_, true, UINT64_MAX);
  transient_command_buffer_.reset();

  // Calculate uniform buffer offset and ranges in uniform buffer
  vk::DeviceSize uniform_offset = 0ull;
  camera_ubos_.resize(swapchain_image_count_);
  for (uint32_t i = 0; i < swapchain_image_count_; i++)
  {
    camera_ubos_[i].offset = uniform_offset;
    camera_ubos_[i].size = sizeof(CameraUbo);
    uniform_offset = align(uniform_offset + sizeof(CameraUbo), ubo_alignment_);
  }

  triangle_model_ubos_.resize(swapchain_image_count_);
  for (uint32_t i = 0; i < swapchain_image_count_; i++)
  {
    triangle_model_ubos_[i].offset = uniform_offset;
    triangle_model_ubos_[i].size = sizeof(ModelUbo);
    uniform_offset = align(uniform_offset + sizeof(ModelUbo), ubo_alignment_);
  }
  
  // Descriptor set
  std::vector<vk::DescriptorSetLayout> set_layouts(swapchain_image_count_, graphics_descriptor_set_layout_);
  graphics_descriptor_sets_ = device_.allocateDescriptorSets({
    descriptor_pool_, set_layouts
    });

  for (int i = 0; i < graphics_descriptor_sets_.size(); i++)
  {
    std::vector<vk::DescriptorBufferInfo> buffer_infos{
      { uniform_buffer_.buffer, camera_ubos_[i].offset, camera_ubos_[i].size },
      { uniform_buffer_.buffer, triangle_model_ubos_[i].offset, triangle_model_ubos_[i].size },
    };
    std::vector<vk::WriteDescriptorSet> descriptor_writes{
      { graphics_descriptor_sets_[i], 0u, 0u, vk::DescriptorType::eUniformBuffer,
      nullptr, buffer_infos[0], nullptr },
      { graphics_descriptor_sets_[i], 1u, 0u, vk::DescriptorType::eUniformBuffer,
      nullptr, buffer_infos[1], nullptr },
    };
    device_.updateDescriptorSets(descriptor_writes, {});
  }
}

void Engine::DestroyResources()
{
  device_.freeDescriptorSets(descriptor_pool_, graphics_descriptor_sets_);
  graphics_descriptor_sets_.clear();

  device_.destroyBuffer(triangle_buffer_.buffer);
  device_.destroyBuffer(floor_buffer_.buffer);
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

  return device_.createShaderModule({ {}, code });
}

vk::Pipeline Engine::CreateGraphicsPipeline(vk::GraphicsPipelineCreateInfo& create_info)
{
  auto result = device_.createGraphicsPipeline(pipeline_cache_, create_info);
  if (result.result != vk::Result::eSuccess)
    throw std::runtime_error("Failed to create graphics pipeline, with error code: " + vk::to_string(result.result));
  return result.value;
}
}
