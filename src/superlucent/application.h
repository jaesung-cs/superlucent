#ifndef SUPERLUCENT_APPLICATION_H_
#define SUPERLUCENT_APPLICATION_H_

#include <memory>
#include <array>
#include <vector>

struct GLFWwindow;

namespace supl
{
class Engine;

namespace scene
{
class Light;
class Camera;
class CameraControl;
}

class Application
{
public:
  Application();
  ~Application();

  void MouseButton(int button, int action, int mods, double x, double y);
  void Key(int key, int scancode, int action, int mods);
  void CursorPos(double x, double y);
  void Scroll(double scroll);
  void Resize(int width, int height);

  void Run();

private:
  void UpdateKeyboard(double dt);

  GLFWwindow* window_;
  int width_ = 0;
  int height_ = 0;

  std::unique_ptr<Engine> engine_;

  // Light
  std::vector<std::shared_ptr<scene::Light>> lights_;

  // Camera
  std::shared_ptr<scene::Camera> camera_;
  std::unique_ptr<scene::CameraControl> camera_control_;

  // Events
  std::array<bool, 2> mouse_buttons_{};
  double mouse_last_x_ = 0.;
  double mouse_last_y_ = 0.;
  std::array<bool, 256> key_pressed_{};

  // Animation
  bool is_animated_ = false;
};
}

#endif // SUPERLUCENT_APPLICATION_H_
