#include <superlucent/engine.h>

#include <iostream>
#include <fstream>

#include <GLFW/glfw3.h>

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
  CreateRendertarget();
  CreateFramebuffer();
  CreatePipelines();
}

Engine::~Engine()
{
  DestroyPipelines();
  DestroyFramebuffer();
  DestroyRendertarget();
  FreeMemory();
  DestroySwapchain();
  DestroyDevice();
  DestroyInstance();
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
      {}, swapchain_images_[0], vk::ImageViewType::e2D, swapchain_image_format_, {},
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
  staging_buffer_.map = device_.mapMemory(staging_buffer_.memory, 0, staging_buffer_.size);

  // Persistently mapped uniform buffer
  uniform_buffer_.buffer = device_.createBuffer({ {},
    uniform_buffer_.size, vk::BufferUsageFlagBits::eTransferSrc });
  uniform_buffer_.memory = device_.allocateMemory({
    device_.getBufferMemoryRequirements(uniform_buffer_.buffer).size,
    host_index });
  device_.bindBufferMemory(uniform_buffer_.buffer, uniform_buffer_.memory, 0);
  uniform_buffer_.map = device_.mapMemory(uniform_buffer_.memory, 0, uniform_buffer_.size);

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
  descriptor_pool_ = device_.createDescriptorPool({ {}, max_sets, pool_sizes });
}

void Engine::FreeMemory()
{
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

  CreateGraphicsPipeline();
}

void Engine::DestroyPipelines()
{
  DestroyGraphicsPipeline();

  device_.destroyPipelineCache(pipeline_cache_);
}

void Engine::CreateGraphicsPipeline()
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
  const std::string base_dir = "c:\\workspace\\superlucent\\src\\superlucent\\shader";
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
  std::vector < vk::VertexInputAttributeDescription> vertex_attribute_descriptions{
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
    false, false, vk::PolygonMode::eFill, vk::CullModeFlagBits::eBack, vk::FrontFace::eCounterClockwise,
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
    true,
    vk::BlendFactor::eSrcAlpha, vk::BlendFactor::eOneMinusSrcAlpha, vk::BlendOp::eAdd,
    vk::BlendFactor::eOne, vk::BlendFactor::eZero, vk::BlendOp::eAdd
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
  auto result = device_.createGraphicsPipeline(pipeline_cache_, pipeline_create_info);
  std::cout << "Grahpics pipeline creation result: " << vk::to_string(result.result) << std::endl;
  graphics_pipeline_ = result.value;

  // Destroy shader module
  device_.destroyShaderModule(vert_module);
  device_.destroyShaderModule(frag_module);
}

void Engine::DestroyGraphicsPipeline()
{
  device_.destroyDescriptorSetLayout(graphics_descriptor_set_layout_);
  device_.destroyPipeline(graphics_pipeline_);
  device_.destroyPipelineLayout(graphics_pipeline_layout_);
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
}
