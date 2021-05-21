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
};
}

#endif // SUPERLUCENT_ENGINE_H_
