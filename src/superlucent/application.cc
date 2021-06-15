#include <superlucent/application.h>

#include <iostream>
#include <chrono>
#include <thread>
#include <stdexcept>
#include <queue>

#include <vulkan/vulkan.hpp>

#include <GLFW/glfw3.h>

#include <superlucent/engine/engine.h>
#include <superlucent/scene/light.h>
#include <superlucent/scene/camera.h>
#include <superlucent/scene/camera_control.h>

namespace supl
{
namespace
{
void mouse_button_callback(GLFWwindow* window, int button, int action, int mods)
{
  double x, y;
  glfwGetCursorPos(window, &x, &y);
  auto module_window = static_cast<Application*>(glfwGetWindowUserPointer(window));
  module_window->MouseButton(button, action, mods, x, y);
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

  engine_ = std::make_unique<engine::Engine>(window_, max_width, max_height);

  camera_ = std::make_shared<scene::Camera>();
  camera_->SetScreenSize(width_, height_);

  camera_control_ = std::make_unique<scene::CameraControl>(camera_);

  auto light = std::make_shared<scene::Light>();
  light->SetDirectionalLight();
  light->SetPosition(glm::vec3{ 0.f, 0.f, 1.f });
  light->SetAmbient(glm::vec3{ 0.1f, 0.1f, 0.1f });
  light->SetDiffuse(glm::vec3{ 0.2f, 0.2f, 0.2f });
  light->SetSpecular(glm::vec3{ 0.1f, 0.1f, 0.1f });
  lights_.emplace_back(std::move(light));

  light = std::make_shared<scene::Light>();
  light->SetPointLight();
  light->SetPosition(glm::vec3{ 0.f, -5.f, 3.f });
  light->SetAmbient(glm::vec3{ 0.2f, 0.2f, 0.2f });
  light->SetDiffuse(glm::vec3{ 0.8f, 0.8f, 0.8f });
  light->SetSpecular(glm::vec3{ 1.f, 1.f, 1.f });
  lights_.emplace_back(std::move(light));

  // Initialize events
  for (int i = 0; i < key_pressed_.size(); i++)
    key_pressed_[i] = false;
  for (int i = 0; i < mouse_buttons_.size(); i++)
    mouse_buttons_[i] = false;
}

Application::~Application()
{
  glfwDestroyWindow(window_);
  glfwTerminate();
}

void Application::Run()
{
  using namespace std::chrono_literals;
  using Clock = std::chrono::high_resolution_clock;
  using Duration = std::chrono::duration<double>;
  using Timestamp = Clock::time_point;

  std::deque<Timestamp> deque;

  int64_t recent_seconds = 0;
  const auto start_time = Clock::now();
  Timestamp previous_time = start_time;
  double animation_time = 0.;
  while (!glfwWindowShouldClose(window_))
  {
    glfwPollEvents();

    const auto current_time = Clock::now();

    // Update keyboard
    const auto dt = std::chrono::duration<double>(current_time - previous_time).count();
    UpdateKeyboard(dt);

    // Light position updated to camera eye
    lights_[0]->SetPosition(camera_->Eye() - camera_->Center());

    engine_->UpdateLights(lights_);
    engine_->UpdateCamera(camera_);
    engine_->Draw(animation_time);

    while (!deque.empty() && deque.front() < current_time - 1s)
      deque.pop_front();

    deque.push_back(current_time);

    // Update animation time
    if (is_animated_)
      animation_time += dt;

    const auto seconds = static_cast<int64_t>(Duration(Clock::now() - start_time).count());
    if (seconds > recent_seconds)
    {
      const auto fps = deque.size();
      std::cout << fps << std::endl;

      recent_seconds = seconds;
    }

    previous_time = current_time;
  }
}

void Application::MouseButton(int button, int action, int mods, double x, double y)
{
  int mouse_button_index = -1;
  switch (button)
  {
  case GLFW_MOUSE_BUTTON_LEFT: mouse_button_index = 0; break;
  case GLFW_MOUSE_BUTTON_RIGHT: mouse_button_index = 1; break;
  }

  int mouse_button_state_index = -1;
  switch (action)
  {
  case GLFW_RELEASE: mouse_button_state_index = 0; break;
  case GLFW_PRESS: mouse_button_state_index = 1; break;
  }

  if (mouse_button_index >= 0)
    mouse_buttons_[mouse_button_index] = mouse_button_state_index;
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

    if (key == GLFW_KEY_ENTER)
    {
      is_animated_ = !is_animated_;
    }
  }

  switch (action)
  {
  case GLFW_PRESS:
    key_pressed_[key] = true;
    break;
  case GLFW_RELEASE:
    key_pressed_[key] = false;
    break;
  }

}

void Application::CursorPos(double x, double y)
{
  const auto dx = static_cast<int>(x - mouse_last_x_);
  const auto dy = static_cast<int>(y - mouse_last_y_);

  if (mouse_buttons_[0] && mouse_buttons_[1])
  {
    camera_control_->ZoomByPixels(dx, dy);
    camera_control_->Update();
  }

  else if (mouse_buttons_[0])
  {
    camera_control_->RotateByPixels(dx, dy);
    camera_control_->Update();
  }

  else if (mouse_buttons_[1])
  {
    camera_control_->TranslateByPixels(dx, dy);
    camera_control_->Update();
  }

  mouse_last_x_ = x;
  mouse_last_y_ = y;
}

void Application::Scroll(double scroll)
{
  camera_control_->ZoomByWheel(static_cast<int>(scroll));
  camera_control_->Update();
}

void Application::Resize(int width, int height)
{
  camera_->SetScreenSize(width, height);
  engine_->Resize(static_cast<uint32_t>(width), static_cast<uint32_t>(height));
}

void Application::UpdateKeyboard(double dt)
{
  const auto dtf = static_cast<float>(dt);

  // Move camera
  if (key_pressed_['W'])
    camera_control_->MoveForward(dtf);
  if (key_pressed_['S'])
    camera_control_->MoveForward(-dtf);
  if (key_pressed_['A'])
    camera_control_->MoveRight(-dtf);
  if (key_pressed_['D'])
    camera_control_->MoveRight(dtf);
  if (key_pressed_[' '])
    camera_control_->MoveUp(dtf);

  camera_control_->Update();
}
}
