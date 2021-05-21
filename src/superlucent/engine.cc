#include <superlucent/engine.h>

#include <iostream>

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
}

Engine::Engine(GLFWwindow* window, int max_width, int max_height)
  : max_width_(max_width)
  , max_height_(max_height)
{
  // Current width and height
  glfwGetWindowSize(window, &width_, &height_);

  // Prepare vulkan resources
  CreateInstance(window);
  CreateDevice();
  PreallocateMemory();
}

Engine::~Engine()
{
  FreeMemory();
  DestroyDevice();
  DestroyInstance();
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
}

void Engine::FreeMemory()
{
  device_.unmapMemory(staging_buffer_.memory);
  device_.freeMemory(staging_buffer_.memory);
  device_.destroyBuffer(staging_buffer_.buffer);

  device_.unmapMemory(uniform_buffer_.memory);
  device_.freeMemory(uniform_buffer_.memory);
  device_.destroyBuffer(uniform_buffer_.buffer);

  device_.freeMemory(device_memory_);
  device_.freeMemory(host_memory_);
}
}
