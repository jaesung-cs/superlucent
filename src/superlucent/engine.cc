#include <superlucent/engine.h>

#include <vulkan/vulkan.hpp>

#include <GLFW/glfw3.h>

namespace supl
{
Engine::Engine(GLFWwindow* window, int max_width, int max_height)
  : max_width_(max_width)
  , max_height_(max_height)
{
  // Current width and height
  glfwGetWindowSize(window, &width_, &height_);

  // TODO: initialize vulkan
}

Engine::~Engine()
{
  // TODO: terminate vulkan
}
}
