#include <superlucent/engine.h>

#include <iostream>
#include <fstream>

#include <GLFW/glfw3.h>

#include <glm/gtc/matrix_transform.hpp>

#include <superlucent/scene/light.h>
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
  CreateSampler();
  PrepareResources();
  CreateSynchronizationObjects();

  // Initialize uniform values
  triangle_model_.model = glm::mat4(1.f);
  triangle_model_.model[3][2] = 0.1f;
  triangle_model_.model_inverse_transpose = glm::mat3(1.f);
}

Engine::~Engine()
{
  device_.waitIdle();

  DestroySynchronizationObjects();
  DestroyResources();
  DestroySampler();
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

void Engine::Draw(std::chrono::high_resolution_clock::time_point timestamp)
{
  double dt = 0.;
  if (first_draw_)
    first_draw_ = false;
  else
    dt = std::chrono::duration<double>(timestamp - previous_timestamp_).count();

  previous_timestamp_ = timestamp;

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
  RecordDrawCommands(draw_command_buffer, image_index, dt);
  draw_command_buffer.end();

  // Update uniforms
  std::memcpy(uniform_buffer_.map + camera_ubos_[image_index].offset, &camera_, sizeof(CameraUbo));
  std::memcpy(uniform_buffer_.map + triangle_model_ubos_[image_index].offset, &triangle_model_, sizeof(ModelUbo));
  std::memcpy(uniform_buffer_.map + light_ubos_[image_index].offset, &lights_, sizeof(LightUbo));

  particle_simulation_.simulation_params.dt = dt;
  particle_simulation_.simulation_params.num_particles = particle_simulation_.num_particles;
  particle_simulation_.simulation_params.alpha = 0.1f;
  std::memcpy(uniform_buffer_.map + particle_simulation_.simulation_params_ubos[image_index].offset, &particle_simulation_.simulation_params, sizeof(SimulationParamsUbo));

  // Submit
  std::vector<vk::PipelineStageFlags> stages{
    vk::PipelineStageFlagBits::eColorAttachmentOutput
  };
  vk::SubmitInfo submit_info;
  submit_info
    .setWaitSemaphores(image_available_semaphores_[current_frame_])
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
  {
    // TODO: Recreate swapchain
  }
  else if (present_result != vk::Result::eSuccess)
    throw std::runtime_error("Failed to present swapchain image");

  current_frame_ = (current_frame_ + 1) % 2;
}

void Engine::RecordDrawCommands(vk::CommandBuffer& command_buffer, uint32_t image_index, double dt)
{
  if (dt > 0.)
  {
    // Barrier to make sure previous rendering command
    // TODO: triple buffering as well as for particle buffers
    vk::BufferMemoryBarrier particle_buffer_memory_barrier;
    particle_buffer_memory_barrier
      .setSrcAccessMask(vk::AccessFlagBits::eVertexAttributeRead)
      .setDstAccessMask(vk::AccessFlagBits::eShaderWrite)
      .setSrcQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
      .setDstQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
      .setBuffer(particle_simulation_.particle_buffer)
      .setOffset(0)
      .setSize(particle_simulation_.num_particles * sizeof(float) * 20);
    command_buffer.pipelineBarrier(vk::PipelineStageFlagBits::eVertexInput, vk::PipelineStageFlagBits::eComputeShader, {},
      {}, particle_buffer_memory_barrier, {});

    // Prepare compute shaders
    command_buffer.bindDescriptorSets(vk::PipelineBindPoint::eCompute, particle_simulation_.pipeline_layout, 0u,
      particle_simulation_.descriptor_sets[image_index], {});

    // Forward
    command_buffer.bindPipeline(vk::PipelineBindPoint::eCompute, particle_simulation_.forward_pipeline);
    command_buffer.dispatch((particle_simulation_.num_particles + 255) / 256, 1, 1);

    // Initialize collision detection
    command_buffer.bindPipeline(vk::PipelineBindPoint::eCompute, particle_simulation_.initialize_collision_detection_pipeline);
    command_buffer.dispatch(1, 1, 1);

    particle_buffer_memory_barrier
      .setSrcAccessMask(vk::AccessFlagBits::eShaderWrite)
      .setDstAccessMask(vk::AccessFlagBits::eShaderRead);

    vk::BufferMemoryBarrier collision_buffer_memory_barrier;
    collision_buffer_memory_barrier
      .setSrcAccessMask(vk::AccessFlagBits::eShaderWrite)
      .setDstAccessMask(vk::AccessFlagBits::eShaderRead)
      .setSrcQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
      .setDstQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
      .setBuffer(particle_simulation_.collision_pairs_buffer)
      .setOffset(0)
      .setSize(particle_simulation_.collision_pairs_buffer_size);

    command_buffer.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eComputeShader, {},
      {}, { particle_buffer_memory_barrier, collision_buffer_memory_barrier }, {});

    // Collision detection
    // TODO: dispatch indirect
    command_buffer.bindPipeline(vk::PipelineBindPoint::eCompute, particle_simulation_.collision_detection_pipeline);
    command_buffer.dispatch((particle_simulation_.num_collisions + 255) / 256, 1, 1);

    command_buffer.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eComputeShader, {},
      {}, collision_buffer_memory_barrier, {});

    // Initialize solver
    command_buffer.bindPipeline(vk::PipelineBindPoint::eCompute, particle_simulation_.initialize_solver_pipeline);
    command_buffer.dispatch((particle_simulation_.num_collisions + particle_simulation_.num_particles * 3 + 255) / 256, 1, 1);

    vk::BufferMemoryBarrier solver_buffer_memory_barrier;
    solver_buffer_memory_barrier
      .setSrcAccessMask(vk::AccessFlagBits::eShaderWrite)
      .setDstAccessMask(vk::AccessFlagBits::eShaderRead)
      .setSrcQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
      .setDstQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
      .setBuffer(particle_simulation_.solver_buffer)
      .setOffset(0)
      .setSize(particle_simulation_.solver_buffer_size);

    command_buffer.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eComputeShader, {},
      {}, solver_buffer_memory_barrier, {});

    // Solve
    constexpr int solver_iterations = 10;
    for (int i = 0; i < solver_iterations; i++)
    {
      // Solve delta lambda
      command_buffer.bindPipeline(vk::PipelineBindPoint::eCompute, particle_simulation_.solve_delta_lambda_pipeline);
      command_buffer.dispatch((particle_simulation_.num_collisions + 255) / 256, 1, 1);

      command_buffer.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eComputeShader, {},
        {}, solver_buffer_memory_barrier, {});

      // Solve delta x
      command_buffer.bindPipeline(vk::PipelineBindPoint::eCompute, particle_simulation_.solve_delta_x_pipeline);
      command_buffer.dispatch((particle_simulation_.num_particles + 255) / 256, 1, 1);

      command_buffer.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eComputeShader, {},
        {}, solver_buffer_memory_barrier, {});

      // Solve x and lambda
      command_buffer.bindPipeline(vk::PipelineBindPoint::eCompute, particle_simulation_.solve_x_lambda_pipeline);
      command_buffer.dispatch((particle_simulation_.num_collisions + particle_simulation_.num_particles * 3 + 255) / 256, 1, 1);

      command_buffer.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eComputeShader, {},
        {}, solver_buffer_memory_barrier, {});
    }

    // Velocity update
    command_buffer.bindPipeline(vk::PipelineBindPoint::eCompute, particle_simulation_.velocity_update_pipeline);
    command_buffer.dispatch((particle_simulation_.num_particles + 255) / 256, 1, 1);

    particle_buffer_memory_barrier
      .setSrcAccessMask(vk::AccessFlagBits::eShaderWrite)
      .setDstAccessMask(vk::AccessFlagBits::eVertexAttributeRead);
    command_buffer.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eVertexInput, {},
      {}, particle_buffer_memory_barrier, {});
  }

  // Begin render pass
  command_buffer.setViewport(0, vk::Viewport{ 0.f, 0.f, static_cast<float>(width_), static_cast<float>(height_), 0.f, 1.f });

  command_buffer.setScissor(0, vk::Rect2D{ {0u, 0u}, {width_, height_} });

  std::vector<vk::ClearValue> clear_values{
    vk::ClearColorValue{ std::array<float, 4>{0.8f, 0.8f, 0.8f, 1.f} },
    vk::ClearDepthStencilValue{ 1.f, 0u }
  };
  vk::RenderPassBeginInfo render_pass_begin_info;
  render_pass_begin_info
    .setRenderPass(render_pass_)
    .setFramebuffer(swapchain_framebuffers_[image_index])
    .setRenderArea(vk::Rect2D{ {0u, 0u}, {width_, height_} })
    .setClearValues(clear_values);
  command_buffer.beginRenderPass(render_pass_begin_info, vk::SubpassContents::eInline);

  // Draw triangle model
  command_buffer.bindPipeline(vk::PipelineBindPoint::eGraphics, color_pipeline_);

  command_buffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, graphics_pipeline_layout_, 0u,
    graphics_descriptor_sets_[image_index], {});

  command_buffer.bindVertexBuffers(0u, { triangle_buffer_.buffer }, { 0ull });

  command_buffer.bindIndexBuffer(triangle_buffer_.buffer, triangle_buffer_.index_offset, vk::IndexType::eUint32);

  command_buffer.drawIndexed(triangle_buffer_.num_indices, 1u, 0u, 0u, 0u);

  // Draw cells
  command_buffer.bindPipeline(vk::PipelineBindPoint::eGraphics, cell_sphere_pipeline_);

  command_buffer.bindVertexBuffers(0u,
    { cells_buffer_.vertex.buffer, particle_simulation_.particle_buffer },
    { 0ull, 0ull });

  command_buffer.bindIndexBuffer(cells_buffer_.vertex.buffer, cells_buffer_.vertex.index_offset, vk::IndexType::eUint32);

  command_buffer.drawIndexed(cells_buffer_.vertex.num_indices, particle_simulation_.num_particles, 0u, 0u, 0u);

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

  vk::MemoryAllocateInfo memory_allocate_info;
  memory_allocate_info
    .setAllocationSize(chunk_size)
    .setMemoryTypeIndex(device_index);
  device_memory_ = device_.allocateMemory(memory_allocate_info);

  memory_allocate_info
    .setAllocationSize(chunk_size)
    .setMemoryTypeIndex(host_index);
  host_memory_ = device_.allocateMemory(memory_allocate_info);

  // Persistently mapped staging buffer
  vk::BufferCreateInfo buffer_create_info;
  buffer_create_info
    .setSize(staging_buffer_.size)
    .setUsage(vk::BufferUsageFlagBits::eTransferSrc);
  staging_buffer_.buffer = device_.createBuffer(buffer_create_info);

  memory_allocate_info
    .setAllocationSize(device_.getBufferMemoryRequirements(staging_buffer_.buffer).size)
    .setMemoryTypeIndex(host_index);
  staging_buffer_.memory = device_.allocateMemory(memory_allocate_info);

  device_.bindBufferMemory(staging_buffer_.buffer, staging_buffer_.memory, 0);
  staging_buffer_.map = static_cast<uint8_t*>(device_.mapMemory(staging_buffer_.memory, 0, staging_buffer_.size));

  // Persistently mapped uniform buffer
  buffer_create_info
    .setSize(uniform_buffer_.size)
    .setUsage(vk::BufferUsageFlagBits::eUniformBuffer);
  uniform_buffer_.buffer = device_.createBuffer(buffer_create_info);

  memory_allocate_info
    .setAllocationSize(device_.getBufferMemoryRequirements(uniform_buffer_.buffer).size)
    .setMemoryTypeIndex(host_index);
  uniform_buffer_.memory = device_.allocateMemory(memory_allocate_info);

  device_.bindBufferMemory(uniform_buffer_.buffer, uniform_buffer_.memory, 0);
  uniform_buffer_.map = static_cast<uint8_t*>(device_.mapMemory(uniform_buffer_.memory, 0, uniform_buffer_.size));

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
    { vk::DescriptorType::eStorageBuffer, 2 },
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

  device_.unmapMemory(uniform_buffer_.memory);
  device_.freeMemory(uniform_buffer_.memory);
  device_.destroyBuffer(uniform_buffer_.buffer);

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

void Engine::CreateFramebuffer()
{
  // Attachment descriptions
  vk::AttachmentDescription color_attachment_description;
  color_attachment_description
    .setFormat(swapchain_image_format_)
    .setSamples(vk::SampleCountFlagBits::e4)
    .setLoadOp(vk::AttachmentLoadOp::eClear)
    .setStoreOp(vk::AttachmentStoreOp::eStore)
    .setStencilLoadOp(vk::AttachmentLoadOp::eDontCare)
    .setStencilStoreOp(vk::AttachmentStoreOp::eDontCare)
    .setInitialLayout(vk::ImageLayout::eUndefined)
    .setFinalLayout(vk::ImageLayout::eColorAttachmentOptimal);

  vk::AttachmentDescription depth_attachment_description;
  depth_attachment_description
    .setFormat(vk::Format::eD24UnormS8Uint)
    .setSamples(vk::SampleCountFlagBits::e4)
    .setLoadOp(vk::AttachmentLoadOp::eClear)
    .setStoreOp(vk::AttachmentStoreOp::eDontCare)
    .setStencilLoadOp(vk::AttachmentLoadOp::eDontCare)
    .setStencilStoreOp(vk::AttachmentStoreOp::eDontCare)
    .setInitialLayout(vk::ImageLayout::eUndefined)
    .setFinalLayout(vk::ImageLayout::eDepthStencilAttachmentOptimal);

  vk::AttachmentDescription resolve_attachment_description;
  resolve_attachment_description
    .setFormat(swapchain_image_format_)
    .setSamples(vk::SampleCountFlagBits::e1)
    .setLoadOp(vk::AttachmentLoadOp::eDontCare)
    .setStoreOp(vk::AttachmentStoreOp::eStore)
    .setStencilLoadOp(vk::AttachmentLoadOp::eDontCare)
    .setStencilStoreOp(vk::AttachmentStoreOp::eDontCare)
    .setInitialLayout(vk::ImageLayout::eUndefined)
    .setFinalLayout(vk::ImageLayout::ePresentSrcKHR);

  std::vector<vk::AttachmentDescription> attachment_descriptions{
    color_attachment_description,
    depth_attachment_description,
    resolve_attachment_description,
  };

  // Attachment references
  vk::AttachmentReference color_attachment_reference;
  color_attachment_reference
    .setAttachment(0)
    .setLayout(vk::ImageLayout::eColorAttachmentOptimal);

  vk::AttachmentReference depth_attachment_reference;
  depth_attachment_reference
    .setAttachment(1)
    .setLayout(vk::ImageLayout::eDepthStencilAttachmentOptimal);

  vk::AttachmentReference resolve_attachment_reference;
  resolve_attachment_reference
    .setAttachment(2)
    .setLayout(vk::ImageLayout::eColorAttachmentOptimal);

  // Subpasses
  vk::SubpassDescription subpass;
  subpass
    .setPipelineBindPoint(vk::PipelineBindPoint::eGraphics)
    .setColorAttachments(color_attachment_reference)
    .setResolveAttachments(resolve_attachment_reference)
    .setPDepthStencilAttachment(&depth_attachment_reference);

  // Dependencies
  vk::SubpassDependency dependency;
  dependency
    .setSrcSubpass(VK_SUBPASS_EXTERNAL)
    .setDstSubpass(0)
    .setSrcStageMask(vk::PipelineStageFlagBits::eColorAttachmentOutput | vk::PipelineStageFlagBits::eEarlyFragmentTests)
    .setDstStageMask(vk::PipelineStageFlagBits::eColorAttachmentOutput | vk::PipelineStageFlagBits::eEarlyFragmentTests)
    .setSrcAccessMask({})
    .setDstAccessMask(vk::AccessFlagBits::eColorAttachmentWrite | vk::AccessFlagBits::eDepthStencilAttachmentWrite);

  // Render pass
  vk::RenderPassCreateInfo render_pass_create_info;
  render_pass_create_info
    .setAttachments(attachment_descriptions)
    .setSubpasses(subpass)
    .setDependencies(dependency);
  render_pass_ = device_.createRenderPass(render_pass_create_info);

  // Framebuffer
  swapchain_framebuffers_.resize(swapchain_image_count_);
  for (uint32_t i = 0; i < swapchain_image_count_; i++)
  {
    std::vector<vk::ImageView> attachments{
      rendertarget_.color_image_view,
      rendertarget_.depth_image_view,
      swapchain_image_views_[i],
    };

    vk::FramebufferCreateInfo framebuffer_create_info;
    framebuffer_create_info
      .setRenderPass(render_pass_)
      .setAttachments(attachments)
      .setWidth(width_)
      .setHeight(height_)
      .setLayers(1);
    swapchain_framebuffers_[i] = device_.createFramebuffer(framebuffer_create_info);
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
  CreateComputePipelines();
}

void Engine::DestroyPipelines()
{
  DestroyGraphicsPipelines();
  DestroyComputePipelines();

  device_.destroyPipelineCache(pipeline_cache_);
}

void Engine::CreateGraphicsPipelines()
{
  // Descriptor set layout
  std::vector<vk::DescriptorSetLayoutBinding> descriptor_set_layout_bindings;
  vk::DescriptorSetLayoutBinding descriptor_set_layout_binding;

  descriptor_set_layout_binding
    .setBinding(0)
    .setDescriptorType(vk::DescriptorType::eUniformBuffer)
    .setDescriptorCount(1)
    .setStageFlags(vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment);
  descriptor_set_layout_bindings.push_back(descriptor_set_layout_binding);

  descriptor_set_layout_binding
    .setBinding(1)
    .setDescriptorType(vk::DescriptorType::eUniformBuffer)
    .setDescriptorCount(1)
    .setStageFlags(vk::ShaderStageFlagBits::eVertex);
  descriptor_set_layout_bindings.push_back(descriptor_set_layout_binding);

  descriptor_set_layout_binding
    .setBinding(2)
    .setDescriptorType(vk::DescriptorType::eCombinedImageSampler)
    .setDescriptorCount(1)
    .setStageFlags(vk::ShaderStageFlagBits::eFragment);
  descriptor_set_layout_bindings.push_back(descriptor_set_layout_binding);

  descriptor_set_layout_binding
    .setBinding(3)
    .setDescriptorType(vk::DescriptorType::eUniformBuffer)
    .setDescriptorCount(1)
    .setStageFlags(vk::ShaderStageFlagBits::eFragment);
  descriptor_set_layout_bindings.push_back(descriptor_set_layout_binding);

  vk::DescriptorSetLayoutCreateInfo descriptor_set_layout_create_info;
  descriptor_set_layout_create_info
    .setBindings(descriptor_set_layout_bindings);
  graphics_descriptor_set_layout_ = device_.createDescriptorSetLayout(descriptor_set_layout_create_info);

  // Pipeline layout
  vk::PipelineLayoutCreateInfo pipeline_layout_create_info;
  pipeline_layout_create_info
    .setSetLayouts(graphics_descriptor_set_layout_);
  graphics_pipeline_layout_ = device_.createPipelineLayout(pipeline_layout_create_info);

  // Shader modules
  const std::string base_dir = "C:\\workspace\\superlucent\\src\\superlucent\\shader";
  vk::ShaderModule vert_module = CreateShaderModule(base_dir + "\\color.vert.spv");
  vk::ShaderModule frag_module = CreateShaderModule(base_dir + "\\color.frag.spv");

  // Shader stages
  std::vector<vk::PipelineShaderStageCreateInfo> shader_stages(2);

  shader_stages[0]
    .setStage(vk::ShaderStageFlagBits::eVertex)
    .setModule(vert_module)
    .setPName("main");

  shader_stages[1]
    .setStage(vk::ShaderStageFlagBits::eFragment)
    .setModule(frag_module)
    .setPName("main");

  // Vertex input
  vk::VertexInputBindingDescription vertex_binding_description;
  vertex_binding_description
    .setBinding(0)
    .setStride(sizeof(float) * 6)
    .setInputRate(vk::VertexInputRate::eVertex);

  std::vector<vk::VertexInputAttributeDescription> vertex_attribute_descriptions(2);
  vertex_attribute_descriptions[0]
    .setLocation(0)
    .setBinding(0)
    .setFormat(vk::Format::eR32G32B32Sfloat)
    .setOffset(0);

  vertex_attribute_descriptions[1]
    .setLocation(1)
    .setBinding(0)
    .setFormat(vk::Format::eR32G32B32Sfloat)
    .setOffset(sizeof(float) * 3);

  vk::PipelineVertexInputStateCreateInfo vertex_input;
  vertex_input
    .setVertexBindingDescriptions(vertex_binding_description)
    .setVertexAttributeDescriptions(vertex_attribute_descriptions);

  // Input assembly
  vk::PipelineInputAssemblyStateCreateInfo input_assembly;
  input_assembly
    .setTopology(vk::PrimitiveTopology::eTriangleList)
    .setPrimitiveRestartEnable(false);

  // Viewport
  vk::Viewport viewport_area;
  viewport_area
    .setX(0.f)
    .setY(0.f)
    .setWidth(width_)
    .setHeight(height_)
    .setMinDepth(0.f)
    .setMaxDepth(1.f);

  vk::Rect2D scissor{ { 0u, 0u }, { width_, height_ } };

  vk::PipelineViewportStateCreateInfo viewport;
  viewport
    .setViewports(viewport_area)
    .setScissors(scissor);

  // Rasterization
  vk::PipelineRasterizationStateCreateInfo rasterization;
  rasterization
    .setDepthClampEnable(false)
    .setRasterizerDiscardEnable(false)
    .setPolygonMode(vk::PolygonMode::eFill)
    .setCullMode(vk::CullModeFlagBits::eNone)
    .setFrontFace(vk::FrontFace::eCounterClockwise)
    .setDepthBiasEnable(false)
    .setLineWidth(1.f);

  // Multisample
  vk::PipelineMultisampleStateCreateInfo multisample;
  multisample
    .setRasterizationSamples(vk::SampleCountFlagBits::e4);

  // Depth stencil
  vk::PipelineDepthStencilStateCreateInfo depth_stencil;
  depth_stencil
    .setDepthTestEnable(true)
    .setDepthWriteEnable(true)
    .setDepthCompareOp(vk::CompareOp::eLess)
    .setDepthBoundsTestEnable(false)
    .setStencilTestEnable(false);

  // Color blend
  vk::PipelineColorBlendAttachmentState color_blend_attachment;
  color_blend_attachment
    .setBlendEnable(true)
    .setSrcColorBlendFactor(vk::BlendFactor::eSrcAlpha)
    .setDstColorBlendFactor(vk::BlendFactor::eOneMinusSrcAlpha)
    .setColorBlendOp(vk::BlendOp::eAdd)
    .setSrcAlphaBlendFactor(vk::BlendFactor::eOne)
    .setDstAlphaBlendFactor(vk::BlendFactor::eZero)
    .setColorBlendOp(vk::BlendOp::eAdd)
    .setColorWriteMask(vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA);

  vk::PipelineColorBlendStateCreateInfo color_blend;
  color_blend
    .setLogicOpEnable(false)
    .setAttachments(color_blend_attachment)
    .setBlendConstants({ 0.f, 0.f, 0.f, 0.f });

  // Dynamic states
  std::vector<vk::DynamicState> dynamic_states{
    vk::DynamicState::eViewport,
    vk::DynamicState::eScissor,
  };
  vk::PipelineDynamicStateCreateInfo dynamic_state;
  dynamic_state
    .setDynamicStates(dynamic_states);
 
  vk::GraphicsPipelineCreateInfo pipeline_create_info;
  pipeline_create_info
    .setStages(shader_stages)
    .setPVertexInputState(&vertex_input)
    .setPInputAssemblyState(&input_assembly)
    .setPViewportState(&viewport)
    .setPRasterizationState(&rasterization)
    .setPMultisampleState(&multisample)
    .setPDepthStencilState(&depth_stencil)
    .setPColorBlendState(&color_blend)
    .setPDynamicState(&dynamic_state)
    .setLayout(graphics_pipeline_layout_)
    .setRenderPass(render_pass_)
    .setSubpass(0);
  color_pipeline_ = CreateGraphicsPipeline(pipeline_create_info);
  device_.destroyShaderModule(vert_module);
  device_.destroyShaderModule(frag_module);

  // Floor graphics pipeline
  vert_module = CreateShaderModule(base_dir + "\\floor.vert.spv");
  frag_module = CreateShaderModule(base_dir + "\\floor.frag.spv");

  shader_stages.resize(2);
  shader_stages[0]
    .setStage(vk::ShaderStageFlagBits::eVertex)
    .setModule(vert_module)
    .setPName("main");

  shader_stages[1]
    .setStage(vk::ShaderStageFlagBits::eFragment)
    .setModule(frag_module)
    .setPName("main");

  vertex_binding_description
    .setBinding(0)
    .setStride(sizeof(float) * 2)
    .setInputRate(vk::VertexInputRate::eVertex);

  vertex_attribute_descriptions.resize(1);
  vertex_attribute_descriptions[0]
    .setLocation(0)
    .setBinding(0)
    .setFormat(vk::Format::eR32G32B32Sfloat)
    .setOffset(0);

  vertex_input
    .setVertexBindingDescriptions(vertex_binding_description)
    .setVertexAttributeDescriptions(vertex_attribute_descriptions);

  input_assembly.setTopology(vk::PrimitiveTopology::eTriangleStrip);

  pipeline_create_info.setStages(shader_stages);

  floor_pipeline_ = CreateGraphicsPipeline(pipeline_create_info);
  device_.destroyShaderModule(vert_module);
  device_.destroyShaderModule(frag_module);

  // Cell sphere graphics pipeline
  vert_module = CreateShaderModule(base_dir + "\\cell_sphere.vert.spv");
  frag_module = CreateShaderModule(base_dir + "\\cell_sphere.frag.spv");

  shader_stages.resize(2);

  shader_stages.resize(2);
  shader_stages[0]
    .setStage(vk::ShaderStageFlagBits::eVertex)
    .setModule(vert_module)
    .setPName("main");

  shader_stages[1]
    .setStage(vk::ShaderStageFlagBits::eFragment)
    .setModule(frag_module)
    .setPName("main");

  std::vector<vk::VertexInputBindingDescription> vertex_binding_descriptions(2);
  vertex_binding_descriptions[0]
    .setBinding(0)
    .setStride(sizeof(float) * 6)
    .setInputRate(vk::VertexInputRate::eVertex);

  // Binding 1 shared with particle compute shader
  vertex_binding_descriptions[1]
    .setBinding(1)
    .setStride(sizeof(float) * 20)
    .setInputRate(vk::VertexInputRate::eInstance);

  vertex_attribute_descriptions.resize(4);
  vertex_attribute_descriptions[0]
    .setLocation(0)
    .setBinding(0)
    .setFormat(vk::Format::eR32G32B32Sfloat)
    .setOffset(0);

  vertex_attribute_descriptions[1]
    .setLocation(1)
    .setBinding(0)
    .setFormat(vk::Format::eR32G32B32Sfloat)
    .setOffset(sizeof(float) * 3);

  vertex_attribute_descriptions[2]
    .setLocation(2)
    .setBinding(1)
    .setFormat(vk::Format::eR32G32B32Sfloat)
    .setOffset(sizeof(float) * 4); // position offset

  vertex_attribute_descriptions[3]
    .setLocation(3)
    .setBinding(1)
    .setFormat(vk::Format::eR32Sfloat)
    .setOffset(sizeof(float) * 12); // radius

  vertex_input
    .setVertexBindingDescriptions(vertex_binding_descriptions)
    .setVertexAttributeDescriptions(vertex_attribute_descriptions);

  input_assembly
    .setTopology(vk::PrimitiveTopology::eTriangleStrip)
    .setPrimitiveRestartEnable(true);

  rasterization.setCullMode(vk::CullModeFlagBits::eBack);

  pipeline_create_info.setStages(shader_stages);

  cell_sphere_pipeline_ = CreateGraphicsPipeline(pipeline_create_info);
  device_.destroyShaderModule(vert_module);
  device_.destroyShaderModule(frag_module);
}

void Engine::DestroyGraphicsPipelines()
{
  device_.destroyDescriptorSetLayout(graphics_descriptor_set_layout_);
  device_.destroyPipeline(color_pipeline_);
  device_.destroyPipeline(floor_pipeline_);
  device_.destroyPipeline(cell_sphere_pipeline_);
  device_.destroyPipelineLayout(graphics_pipeline_layout_);
}

void Engine::CreateComputePipelines()
{
  // Descriptor set layout
  std::vector<vk::DescriptorSetLayoutBinding> bindings(4);
  bindings[0]
    .setBinding(0)
    .setStageFlags(vk::ShaderStageFlagBits::eCompute)
    .setDescriptorType(vk::DescriptorType::eStorageBuffer)
    .setDescriptorCount(1);

  bindings[1]
    .setBinding(1)
    .setStageFlags(vk::ShaderStageFlagBits::eCompute)
    .setDescriptorType(vk::DescriptorType::eUniformBuffer)
    .setDescriptorCount(1);

  bindings[2]
    .setBinding(2)
    .setStageFlags(vk::ShaderStageFlagBits::eCompute)
    .setDescriptorType(vk::DescriptorType::eStorageBuffer)
    .setDescriptorCount(1);

  bindings[3]
    .setBinding(3)
    .setStageFlags(vk::ShaderStageFlagBits::eCompute)
    .setDescriptorType(vk::DescriptorType::eStorageBuffer)
    .setDescriptorCount(1);

  vk::DescriptorSetLayoutCreateInfo descriptor_set_layout_create_info;
  descriptor_set_layout_create_info
    .setBindings(bindings);
  particle_simulation_.descriptor_set_layout = device_.createDescriptorSetLayout(descriptor_set_layout_create_info);

  // Pipeline layout
  vk::PipelineLayoutCreateInfo pipeline_layout_create_info;
  pipeline_layout_create_info
    .setSetLayouts(particle_simulation_.descriptor_set_layout);
  particle_simulation_.pipeline_layout = device_.createPipelineLayout(pipeline_layout_create_info);
  
  // Shader modules
  const std::string base_dir = "C:\\workspace\\superlucent\\src\\superlucent\\shader";
  vk::ShaderModule particle_forward_module = CreateShaderModule(base_dir + "\\particle_forward.comp.spv");
  vk::ShaderModule particle_initialize_collision_detection_module = CreateShaderModule(base_dir + "\\particle_initialize_collision_detection.comp.spv");
  vk::ShaderModule particle_collision_detection_module = CreateShaderModule(base_dir + "\\particle_collision_detection.comp.spv");
  vk::ShaderModule particle_initialize_solver_module = CreateShaderModule(base_dir + "\\particle_initialize_solver.comp.spv");
  vk::ShaderModule particle_solve_delta_lambda_module = CreateShaderModule(base_dir + "\\particle_solve_delta_lambda.comp.spv");
  vk::ShaderModule particle_solve_delta_x_module = CreateShaderModule(base_dir + "\\particle_solve_delta_x.comp.spv");
  vk::ShaderModule particle_solve_x_lambda_module = CreateShaderModule(base_dir + "\\particle_solve_x_lambda.comp.spv");
  vk::ShaderModule particle_velocity_update_module = CreateShaderModule(base_dir + "\\particle_velocity_update.comp.spv");

  // Forward
  vk::PipelineShaderStageCreateInfo shader_stage;
  shader_stage
    .setStage(vk::ShaderStageFlagBits::eCompute)
    .setModule(particle_forward_module)
    .setPName("main");

  vk::ComputePipelineCreateInfo compute_pipeline_create_info;
  compute_pipeline_create_info
    .setStage(shader_stage)
    .setLayout(particle_simulation_.pipeline_layout);
  particle_simulation_.forward_pipeline = CreateComputePipeline(compute_pipeline_create_info);

  // Initialize collision detection
  shader_stage.setModule(particle_initialize_collision_detection_module);
  compute_pipeline_create_info.setStage(shader_stage);
  particle_simulation_.initialize_collision_detection_pipeline = CreateComputePipeline(compute_pipeline_create_info);

  // Collision detection
  shader_stage .setModule(particle_collision_detection_module);
  compute_pipeline_create_info.setStage(shader_stage);
  particle_simulation_.collision_detection_pipeline = CreateComputePipeline(compute_pipeline_create_info);

  // Initialize solver
  shader_stage.setModule(particle_initialize_solver_module);
  compute_pipeline_create_info.setStage(shader_stage);
  particle_simulation_.initialize_solver_pipeline = CreateComputePipeline(compute_pipeline_create_info);

  // Solve delta lambda
  shader_stage.setModule(particle_solve_delta_lambda_module);
  compute_pipeline_create_info.setStage(shader_stage);
  particle_simulation_.solve_delta_lambda_pipeline = CreateComputePipeline(compute_pipeline_create_info);

  // Solve delta x
  shader_stage.setModule(particle_solve_delta_x_module);
  compute_pipeline_create_info.setStage(shader_stage);
  particle_simulation_.solve_delta_x_pipeline = CreateComputePipeline(compute_pipeline_create_info);

  // Solve x lambda
  shader_stage.setModule(particle_solve_x_lambda_module);
  compute_pipeline_create_info.setStage(shader_stage);
  particle_simulation_.solve_x_lambda_pipeline = CreateComputePipeline(compute_pipeline_create_info);

  // Velocity update
  shader_stage.setModule(particle_velocity_update_module);
  compute_pipeline_create_info.setStage(shader_stage);
  particle_simulation_.velocity_update_pipeline = CreateComputePipeline(compute_pipeline_create_info);

  device_.destroyShaderModule(particle_forward_module);
  device_.destroyShaderModule(particle_initialize_collision_detection_module);
  device_.destroyShaderModule(particle_collision_detection_module);
  device_.destroyShaderModule(particle_initialize_solver_module);
  device_.destroyShaderModule(particle_solve_delta_lambda_module);
  device_.destroyShaderModule(particle_solve_delta_x_module);
  device_.destroyShaderModule(particle_solve_x_lambda_module);
  device_.destroyShaderModule(particle_velocity_update_module);
}

void Engine::DestroyComputePipelines()
{
  device_.destroyDescriptorSetLayout(particle_simulation_.descriptor_set_layout);
  device_.destroyPipelineLayout(particle_simulation_.pipeline_layout);
  device_.destroyPipeline(particle_simulation_.forward_pipeline);
  device_.destroyPipeline(particle_simulation_.initialize_collision_detection_pipeline);
  device_.destroyPipeline(particle_simulation_.collision_detection_pipeline);
  device_.destroyPipeline(particle_simulation_.initialize_solver_pipeline);
  device_.destroyPipeline(particle_simulation_.solve_delta_lambda_pipeline);
  device_.destroyPipeline(particle_simulation_.solve_delta_x_pipeline);
  device_.destroyPipeline(particle_simulation_.solve_x_lambda_pipeline);
  device_.destroyPipeline(particle_simulation_.velocity_update_pipeline);
}

void Engine::CreateSampler()
{
  vk::SamplerCreateInfo sampler_create_info;
  sampler_create_info
    .setMagFilter(vk::Filter::eLinear)
    .setMinFilter(vk::Filter::eLinear)
    .setMipmapMode(vk::SamplerMipmapMode::eLinear)
    .setAddressModeU(vk::SamplerAddressMode::eRepeat)
    .setAddressModeV(vk::SamplerAddressMode::eRepeat)
    .setAddressModeW(vk::SamplerAddressMode::eRepeat)
    .setMipLodBias(0.f)
    .setAnisotropyEnable(true)
    .setMaxAnisotropy(16.f)
    .setCompareEnable(false)
    .setMinLod(0.f)
    .setMaxLod(mipmap_level_)
    .setBorderColor(vk::BorderColor::eFloatTransparentBlack)
    .setUnnormalizedCoordinates(false);
  sampler_ = device_.createSampler(sampler_create_info);
}

void Engine::DestroySampler()
{
  device_.destroySampler(sampler_);
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

  vk::BufferCreateInfo buffer_create_info;
  buffer_create_info
    .setSize(triangle_buffer_size)
    .setUsage(vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eIndexBuffer);

  triangle_buffer_.buffer = device_.createBuffer(buffer_create_info);
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

  buffer_create_info
    .setSize(floor_buffer_size)
    .setUsage(vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eIndexBuffer);

  floor_buffer_.buffer = device_.createBuffer(buffer_create_info);
  floor_buffer_.index_offset = floor_vertex_buffer_size;
  floor_buffer_.num_indices = static_cast<uint32_t>(floor_index_buffer.size());

  // Floor texture
  constexpr int floor_texture_length = 128;
  std::vector<uint8_t> floor_texture(floor_texture_length * floor_texture_length * 4);
  for (int u = 0; u < floor_texture_length; u++)
  {
    for (int v = 0; v < floor_texture_length; v++)
    {
      uint8_t color = (255 - 64) + 64 * !((u < floor_texture_length / 2) ^ (v < floor_texture_length / 2));
      floor_texture[(v * floor_texture_length + u) * 4 + 0] = color;
      floor_texture[(v * floor_texture_length + u) * 4 + 1] = color;
      floor_texture[(v * floor_texture_length + u) * 4 + 2] = color;
      floor_texture[(v * floor_texture_length + u) * 4 + 3] = 255;
    }
  }
  constexpr int floor_texture_size = floor_texture_length * floor_texture_length * sizeof(uint8_t) * 4;

  vk::ImageCreateInfo image_create_info;
  image_create_info
    .setImageType(vk::ImageType::e2D)
    .setFormat(vk::Format::eR8G8B8A8Srgb)
    .setExtent(vk::Extent3D{ floor_texture_length, floor_texture_length, 1 })
    .setMipLevels(mipmap_level_)
    .setArrayLayers(1)
    .setSamples(vk::SampleCountFlagBits::e1)
    .setTiling(vk::ImageTiling::eOptimal)
    .setUsage(vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eSampled)
    .setSharingMode(vk::SharingMode::eExclusive)
    .setInitialLayout(vk::ImageLayout::eUndefined);
  floor_texture_.image = device_.createImage(image_create_info);

  // Cells buffer
  constexpr int sphere_segments = 16;
  constexpr int cell_count = 8;
  std::vector<float> cells_buffer;
  std::vector<std::vector<uint32_t>> cells_indices;
  std::vector<uint32_t> cells_index_buffer;

  cells_buffer.push_back(0.f);
  cells_buffer.push_back(0.f);
  cells_buffer.push_back(1.f);
  cells_buffer.push_back(0.f);
  cells_buffer.push_back(0.f);
  cells_buffer.push_back(1.f);

  cells_buffer.push_back(0.f);
  cells_buffer.push_back(0.f);
  cells_buffer.push_back(-1.f);
  cells_buffer.push_back(0.f);
  cells_buffer.push_back(0.f);
  cells_buffer.push_back(-1.f);

  uint32_t cell_index = 2;

  cells_indices.resize(sphere_segments);
  constexpr auto pi = glm::pi<float>();
  for (int i = 0; i < sphere_segments; i++)
  {
    cells_indices[i].resize(sphere_segments);

    const auto theta = static_cast<float>(i) / sphere_segments * 2.f * pi;
    const auto cos_theta = std::cos(theta);
    const auto sin_theta = std::sin(theta);
    for (int j = 1; j < sphere_segments; j++)
    {
      cells_indices[i][j] = cell_index++;

      const auto phi = (0.5f - static_cast<float>(j) / sphere_segments) * pi;
      const auto cos_phi = std::cos(phi);
      const auto sin_phi = std::sin(phi);

      cells_buffer.push_back(cos_theta * cos_phi);
      cells_buffer.push_back(sin_theta * cos_phi);
      cells_buffer.push_back(sin_phi);
      cells_buffer.push_back(cos_theta * cos_phi);
      cells_buffer.push_back(sin_theta * cos_phi);
      cells_buffer.push_back(sin_phi);
    }
  }
  const auto cells_vertex_buffer_size = cells_buffer.size() * sizeof(float);

  // Sphere indices
  for (int i = 0; i < sphere_segments; i++)
  {
    cells_index_buffer.push_back(0);
    for (int j = 1; j < sphere_segments; j++)
    {
      cells_index_buffer.push_back(cells_indices[i][j]);
      cells_index_buffer.push_back(cells_indices[(i + 1) % sphere_segments][j]);
    }
    cells_index_buffer.push_back(1);
    cells_index_buffer.push_back(-1);
  }
  const auto cells_index_buffer_size = cells_index_buffer.size() * sizeof(uint32_t);

  // Prticles
  constexpr float radius = 1.f / cell_count;
  constexpr float mass = radius * radius * radius;
  glm::vec3 gravity = glm::vec3(0.f, 0.f, -9.8f);
  std::vector<float> particle_buffer;
  uint32_t num_particles = 0;
  for (int i = 0; i < cell_count; i++)
  {
    for (int j = 0; j < cell_count; j++)
    {
      for (int k = 0; k < cell_count; k++)
      {
        num_particles++;

        // prev_position
        particle_buffer.push_back(radius * (i * 4 + k - cell_count * 2));
        particle_buffer.push_back(radius * (j * 4 + k - cell_count * 2));
        particle_buffer.push_back(radius * k * 4 + 2.f);
        particle_buffer.push_back(0.f);

        // position
        particle_buffer.push_back(radius * (i * 4 + k - cell_count * 2));
        particle_buffer.push_back(radius * (j * 4 + k - cell_count * 2));
        particle_buffer.push_back(radius * k * 4 + 2.f);
        particle_buffer.push_back(0.f);

        // velocity
        particle_buffer.push_back(-1.f);
        particle_buffer.push_back(1.f);
        particle_buffer.push_back(1.f);
        particle_buffer.push_back(0.f);

        // properties
        particle_buffer.push_back(radius);
        particle_buffer.push_back(mass);
        particle_buffer.push_back(0.f);
        particle_buffer.push_back(0.f);

        // external_force
        particle_buffer.push_back(gravity.x * mass);
        particle_buffer.push_back(gravity.y * mass);
        particle_buffer.push_back(gravity.z * mass);
        particle_buffer.push_back(0.f);
      }
    }
  }
  const auto particle_buffer_size = particle_buffer.size() * sizeof(float);

  // Collision and solver size
  const auto num_collisions =
    num_particles + 5 // walls
    + num_particles * 10; // max 10 collisions for each sphere
  const auto collision_pairs_size = sizeof(uint32_t) + num_collisions * (sizeof(int32_t) * 4 + sizeof(float) * 12);
  
  const auto solver_size =
    (num_collisions // lambda
      + num_particles * 3) // x
    * 2 // delta
    * sizeof(float);

  buffer_create_info
    .setSize(cells_vertex_buffer_size + cells_index_buffer_size)
    .setUsage(vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eIndexBuffer);
  cells_buffer_.vertex.buffer = device_.createBuffer(buffer_create_info);
  cells_buffer_.vertex.index_offset = cells_vertex_buffer_size;
  cells_buffer_.vertex.num_indices = cells_index_buffer.size();

  buffer_create_info
    .setSize(particle_buffer_size)
    .setUsage(vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eStorageBuffer);
  particle_simulation_.particle_buffer = device_.createBuffer(buffer_create_info);
  particle_simulation_.num_particles = num_particles;

  buffer_create_info
    .setSize(collision_pairs_size)
    .setUsage(vk::BufferUsageFlagBits::eStorageBuffer);
  particle_simulation_.collision_pairs_buffer = device_.createBuffer(buffer_create_info);
  particle_simulation_.collision_pairs_buffer_size = collision_pairs_size;
  particle_simulation_.num_collisions = num_collisions;

  buffer_create_info
    .setSize(solver_size)
    .setUsage(vk::BufferUsageFlagBits::eStorageBuffer);
  particle_simulation_.solver_buffer = device_.createBuffer(buffer_create_info);
  particle_simulation_.solver_buffer_size = solver_size;

  // Memory binding
  const auto triangle_memory = AcquireDeviceMemory(triangle_buffer_.buffer);
  device_.bindBufferMemory(triangle_buffer_.buffer, triangle_memory.memory, triangle_memory.offset);

  const auto floor_memory = AcquireDeviceMemory(floor_buffer_.buffer);
  device_.bindBufferMemory(floor_buffer_.buffer, floor_memory.memory, floor_memory.offset);

  const auto floor_texture_memory = AcquireDeviceMemory(floor_texture_.image);
  device_.bindImageMemory(floor_texture_.image, floor_texture_memory.memory, floor_texture_memory.offset);

  const auto cells_vertex_memory = AcquireDeviceMemory(cells_buffer_.vertex.buffer);
  device_.bindBufferMemory(cells_buffer_.vertex.buffer, cells_vertex_memory.memory, cells_vertex_memory.offset);

  const auto particle_memory = AcquireDeviceMemory(particle_simulation_.particle_buffer);
  device_.bindBufferMemory(particle_simulation_.particle_buffer, particle_memory.memory, particle_memory.offset);

  const auto collision_pairs_memory = AcquireDeviceMemory(particle_simulation_.collision_pairs_buffer);
  device_.bindBufferMemory(particle_simulation_.collision_pairs_buffer, collision_pairs_memory.memory, collision_pairs_memory.offset);

  const auto solver_memory = AcquireDeviceMemory(particle_simulation_.solver_buffer);
  device_.bindBufferMemory(particle_simulation_.solver_buffer, solver_memory.memory, solver_memory.offset);

  // Create image view for floor texture
  vk::ImageSubresourceRange subresource_range;
  subresource_range
    .setAspectMask(vk::ImageAspectFlagBits::eColor)
    .setBaseMipLevel(0)
    .setLevelCount(mipmap_level_)
    .setBaseArrayLayer(0)
    .setLayerCount(1);

  vk::ImageViewCreateInfo image_view_create_info;
  image_view_create_info
    .setImage(floor_texture_.image)
    .setViewType(vk::ImageViewType::e2D)
    .setFormat(vk::Format::eR8G8B8A8Srgb)
    .setSubresourceRange(subresource_range);

  floor_texture_.image_view = device_.createImageView(image_view_create_info);

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

  std::memcpy(staging_buffer_.map + staging_offset, cells_buffer.data(), cells_vertex_buffer_size);
  staging_offset += cells_vertex_buffer_size;
  std::memcpy(staging_buffer_.map + staging_offset, cells_index_buffer.data(), cells_index_buffer_size);
  staging_offset += cells_index_buffer_size;

  std::memcpy(staging_buffer_.map + staging_offset, particle_buffer.data(), particle_buffer_size);
  staging_offset += particle_buffer_size;

  std::memcpy(staging_buffer_.map + staging_offset, floor_texture.data(), floor_texture_size);
  staging_offset += floor_texture_size;

  // Transfer commands
  vk::CommandBufferBeginInfo command_buffer_begin_info;
  command_buffer_begin_info
    .setFlags(vk::CommandBufferUsageFlagBits::eOneTimeSubmit);
  transient_command_buffer_.begin(command_buffer_begin_info);

  vk::BufferCopy copy_region;
  copy_region
    .setSrcOffset(0)
    .setDstOffset(0)
    .setSize(triangle_buffer_size);
  transient_command_buffer_.copyBuffer(staging_buffer_.buffer, triangle_buffer_.buffer, copy_region);

  copy_region
    .setSrcOffset(triangle_buffer_size)
    .setDstOffset(0)
    .setSize(floor_buffer_size);
  transient_command_buffer_.copyBuffer(staging_buffer_.buffer, floor_buffer_.buffer, copy_region);

  copy_region
    .setSrcOffset(triangle_buffer_size + floor_buffer_size)
    .setDstOffset(0)
    .setSize(cells_vertex_buffer_size + cells_index_buffer_size);
  transient_command_buffer_.copyBuffer(staging_buffer_.buffer, cells_buffer_.vertex.buffer, copy_region);

  copy_region
    .setSrcOffset(triangle_buffer_size + floor_buffer_size + cells_vertex_buffer_size + cells_index_buffer_size)
    .setDstOffset(0)
    .setSize(particle_buffer_size);
  transient_command_buffer_.copyBuffer(staging_buffer_.buffer, particle_simulation_.particle_buffer, copy_region);

  ImageLayoutTransition(transient_command_buffer_, floor_texture_.image,
    vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal);

  vk::ImageSubresourceLayers image_subresource_layer;
  image_subresource_layer
    .setAspectMask(vk::ImageAspectFlagBits::eColor)
    .setMipLevel(0)
    .setBaseArrayLayer(0)
    .setLayerCount(1);

  vk::BufferImageCopy image_copy_region;
  image_copy_region
    .setBufferOffset(triangle_buffer_size + floor_buffer_size + cells_vertex_buffer_size + cells_index_buffer_size + particle_buffer_size)
    .setBufferRowLength(0)
    .setBufferImageHeight(0)
    .setImageSubresource(image_subresource_layer)
    .setImageOffset(vk::Offset3D{ 0, 0, 0 })
    .setImageExtent(vk::Extent3D{ floor_texture_length, floor_texture_length, 1 });

  transient_command_buffer_.copyBufferToImage(staging_buffer_.buffer, floor_texture_.image,
    vk::ImageLayout::eTransferDstOptimal, image_copy_region);

  GenerateMipmap(transient_command_buffer_, floor_texture_.image, floor_texture_length, floor_texture_length, mipmap_level_);

  transient_command_buffer_.end();

  vk::SubmitInfo submit_info;
  submit_info
    .setCommandBuffers(transient_command_buffer_);
  queue_.submit(submit_info, transfer_fence_);

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

  light_ubos_.resize(swapchain_image_count_);
  for (uint32_t i = 0; i < swapchain_image_count_; i++)
  {
    light_ubos_[i].offset = uniform_offset;
    light_ubos_[i].size = sizeof(LightUbo);
    uniform_offset = align(uniform_offset + sizeof(LightUbo), ubo_alignment_);
  }

  // Descriptor set
  std::vector<vk::DescriptorSetLayout> set_layouts(swapchain_image_count_, graphics_descriptor_set_layout_);
  vk::DescriptorSetAllocateInfo descriptor_set_allocate_info;
  descriptor_set_allocate_info
    .setDescriptorPool(descriptor_pool_)
    .setSetLayouts(set_layouts);
  graphics_descriptor_sets_ = device_.allocateDescriptorSets(descriptor_set_allocate_info);

  for (int i = 0; i < graphics_descriptor_sets_.size(); i++)
  {
    std::vector<vk::DescriptorBufferInfo> buffer_infos(3);
    buffer_infos[0]
      .setBuffer(uniform_buffer_.buffer)
      .setOffset(camera_ubos_[i].offset)
      .setRange(camera_ubos_[i].size);

    buffer_infos[1]
      .setBuffer(uniform_buffer_.buffer)
      .setOffset(triangle_model_ubos_[i].offset)
      .setRange(triangle_model_ubos_[i].size);

    buffer_infos[2]
      .setBuffer(uniform_buffer_.buffer)
      .setOffset(light_ubos_[i].offset)
      .setRange(light_ubos_[i].size);

    std::vector<vk::DescriptorImageInfo> image_infos(1);
    image_infos[0]
      .setSampler(sampler_)
      .setImageView(floor_texture_.image_view)
      .setImageLayout(vk::ImageLayout::eShaderReadOnlyOptimal);

    std::vector<vk::WriteDescriptorSet> descriptor_writes(4);
    descriptor_writes[0]
      .setDstSet(graphics_descriptor_sets_[i])
      .setDstBinding(0)
      .setDstArrayElement(0)
      .setDescriptorType(vk::DescriptorType::eUniformBuffer)
      .setBufferInfo(buffer_infos[0]);

    descriptor_writes[1]
      .setDstSet(graphics_descriptor_sets_[i])
      .setDstBinding(1)
      .setDstArrayElement(0)
      .setDescriptorType(vk::DescriptorType::eUniformBuffer)
      .setBufferInfo(buffer_infos[1]);

    descriptor_writes[2]
      .setDstSet(graphics_descriptor_sets_[i])
      .setDstBinding(2)
      .setDstArrayElement(0)
      .setDescriptorType(vk::DescriptorType::eCombinedImageSampler)
      .setImageInfo(image_infos[0]);

    descriptor_writes[3]
      .setDstSet(graphics_descriptor_sets_[i])
      .setDstBinding(3)
      .setDstArrayElement(0)
      .setDescriptorType(vk::DescriptorType::eUniformBuffer)
      .setBufferInfo(buffer_infos[2]);

    device_.updateDescriptorSets(descriptor_writes, {});
  }

  // Particle descriptor set
  particle_simulation_.simulation_params_ubos.resize(swapchain_image_count_);
  for (int i = 0; i < swapchain_image_count_; i++)
  {
    particle_simulation_.simulation_params_ubos[i].offset = uniform_offset;
    particle_simulation_.simulation_params_ubos[i].size = sizeof(SimulationParamsUbo);
    uniform_offset = align(uniform_offset + sizeof(SimulationParamsUbo), ubo_alignment_);
  }

  set_layouts = std::vector<vk::DescriptorSetLayout>(swapchain_image_count_, particle_simulation_.descriptor_set_layout);
  descriptor_set_allocate_info
    .setDescriptorPool(descriptor_pool_)
    .setSetLayouts(set_layouts);
  particle_simulation_.descriptor_sets = device_.allocateDescriptorSets(descriptor_set_allocate_info);

  for (int i = 0; i < swapchain_image_count_; i++)
  {
    std::vector<vk::DescriptorBufferInfo> buffer_infos(4);
    buffer_infos[0]
      .setBuffer(particle_simulation_.particle_buffer)
      .setOffset(0)
      .setRange(particle_simulation_.num_particles * sizeof(float) * 20);

    buffer_infos[1]
      .setBuffer(uniform_buffer_.buffer)
      .setOffset(particle_simulation_.simulation_params_ubos[i].offset)
      .setRange(particle_simulation_.simulation_params_ubos[i].size);

    buffer_infos[2]
      .setBuffer(particle_simulation_.collision_pairs_buffer)
      .setOffset(0)
      .setRange(particle_simulation_.collision_pairs_buffer_size);

    buffer_infos[3]
      .setBuffer(particle_simulation_.solver_buffer)
      .setOffset(0)
      .setRange(particle_simulation_.solver_buffer_size);

    std::vector<vk::WriteDescriptorSet> descriptor_writes(4);
    descriptor_writes[0]
      .setDstSet(particle_simulation_.descriptor_sets[i])
      .setDstBinding(0)
      .setDstArrayElement(0)
      .setDescriptorType(vk::DescriptorType::eStorageBuffer)
      .setBufferInfo(buffer_infos[0]);

    descriptor_writes[1]
      .setDstSet(particle_simulation_.descriptor_sets[i])
      .setDstBinding(1)
      .setDstArrayElement(0)
      .setDescriptorType(vk::DescriptorType::eUniformBuffer)
      .setBufferInfo(buffer_infos[1]);

    descriptor_writes[2]
      .setDstSet(particle_simulation_.descriptor_sets[i])
      .setDstBinding(2)
      .setDstArrayElement(0)
      .setDescriptorType(vk::DescriptorType::eStorageBuffer)
      .setBufferInfo(buffer_infos[2]);

    descriptor_writes[3]
      .setDstSet(particle_simulation_.descriptor_sets[i])
      .setDstBinding(3)
      .setDstArrayElement(0)
      .setDescriptorType(vk::DescriptorType::eStorageBuffer)
      .setBufferInfo(buffer_infos[3]);

    device_.updateDescriptorSets(descriptor_writes, {});
  }
}

void Engine::DestroyResources()
{
  graphics_descriptor_sets_.clear();
  particle_simulation_.descriptor_sets.clear();

  device_.destroyBuffer(triangle_buffer_.buffer);
  device_.destroyBuffer(floor_buffer_.buffer);
  device_.destroyBuffer(cells_buffer_.vertex.buffer);
  device_.destroyImage(floor_texture_.image);
  device_.destroyImageView(floor_texture_.image_view);
  device_.destroyBuffer(particle_simulation_.collision_pairs_buffer);
  device_.destroyBuffer(particle_simulation_.particle_buffer);
  device_.destroyBuffer(particle_simulation_.solver_buffer);
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

vk::Pipeline Engine::CreateGraphicsPipeline(vk::GraphicsPipelineCreateInfo& create_info)
{
  auto result = device_.createGraphicsPipeline(pipeline_cache_, create_info);
  if (result.result != vk::Result::eSuccess)
    throw std::runtime_error("Failed to create graphics pipeline, with error code: " + vk::to_string(result.result));
  return result.value;
}

vk::Pipeline Engine::CreateComputePipeline(vk::ComputePipelineCreateInfo& create_info)
{
  auto result = device_.createComputePipeline(pipeline_cache_, create_info);
  if (result.result != vk::Result::eSuccess)
    throw std::runtime_error("Failed to create compute pipeline, with error code: " + vk::to_string(result.result));
  return result.value;
}

void Engine::ImageLayoutTransition(vk::CommandBuffer& command_buffer, vk::Image image, vk::ImageLayout old_layout, vk::ImageLayout new_layout)
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
    .setLevelCount(mipmap_level_)
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

void Engine::GenerateMipmap(vk::CommandBuffer& command_buffer, vk::Image image, uint32_t width, uint32_t height, int mipmap_levels)
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
