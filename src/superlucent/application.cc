#include <superlucent/application.h>

#include <iostream>
#include <chrono>
#include <thread>
#include <stdexcept>

#include <vulkan/vulkan.hpp>

#include <GLFW/glfw3.h>

#include <superlucent/engine.h>

namespace supl
{
namespace
{
void mouse_button_callback(GLFWwindow* window, int button, int action, int mods)
{
  auto module_window = static_cast<Application*>(glfwGetWindowUserPointer(window));
  module_window->MouseButton(button, action, mods);
}

void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
  auto module_window = static_cast<Application*>(glfwGetWindowUserPointer(window));
  module_window->Key(key, scancode, action, mods);
}

void cursor_pos_callback(GLFWwindow* window, double x, double y)
{
  auto module_window = static_cast<Application*>(glfwGetWindowUserPointer(window));
  module_window->CursorPos(x, y);
}

void scroll_callback(GLFWwindow* window, double xoffset, double yoffset)
{
  auto module_window = static_cast<Application*>(glfwGetWindowUserPointer(window));
  module_window->Scroll(yoffset);
}

void resize_callback(GLFWwindow* window, int width, int height)
{
  auto module_window = static_cast<Application*>(glfwGetWindowUserPointer(window));
  module_window->Resize(width, height);
}
}

Application::Application()
{
  if (!glfwInit())
    throw std::runtime_error("Failed to initialize GLFW");

  glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

  width_ = 1600;
  height_ = 900;
  window_ = glfwCreateWindow(width_, height_, "Superlucent", NULL, NULL);

  constexpr int max_width = 1920;
  constexpr int max_height = 1080;
  glfwSetWindowSizeLimits(window_, 100, 100, max_width, max_height);

  glfwSetWindowUserPointer(window_, this);
  glfwSetMouseButtonCallback(window_, mouse_button_callback);
  glfwSetCursorPosCallback(window_, cursor_pos_callback);
  glfwSetKeyCallback(window_, key_callback);
  glfwSetScrollCallback(window_, scroll_callback);
  glfwSetWindowSizeCallback(window_, resize_callback);

  glfwSetWindowPos(window_, 100, 100);

  engine_ = std::make_unique<Engine>(window_, max_width, max_height);
}

Application::~Application()
{
  glfwDestroyWindow(window_);
  glfwTerminate();
}

void Application::Run()
{
  using Clock = std::chrono::high_resolution_clock;
  using Duration = std::chrono::duration<double>;

  uint64_t frame = 0;
  const auto start_time = Clock::now();
  while (!glfwWindowShouldClose(window_))
  {
    glfwPollEvents();

    // TODO: draw

    frame++;

    std::this_thread::sleep_until(start_time + Duration(frame / fps_));

    const auto current_time = Clock::now();
    std::cout << "fps: " << frame / Duration(current_time - start_time).count() << std::endl;
  }
}

void Application::MouseButton(int button, int action, int mods)
{
}

void Application::Key(int key, int scancode, int action, int mods)
{
  if (action == GLFW_PRESS)
  {
    if (key == GLFW_KEY_GRAVE_ACCENT)
    {
      glfwSetWindowShouldClose(window_, true);
      return;
    }
  }
}

void Application::CursorPos(double x, double y)
{
}

void Application::Scroll(double scroll)
{
}

void Application::Resize(int width, int height)
{
}
}
