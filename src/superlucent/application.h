#ifndef SUPERLUCENT_APPLICATION_H_
#define SUPERLUCENT_APPLICATION_H_

#include <memory>

struct GLFWwindow;

namespace supl
{
class Engine;

class Application
{
public:
  Application();
  ~Application();

  void MouseButton(int button, int action, int mods);
  void Key(int key, int scancode, int action, int mods);
  void CursorPos(double x, double y);
  void Scroll(double scroll);
  void Resize(int width, int height);

  void Run();

private:
  GLFWwindow* window_;
  int width_ = 0;
  int height_ = 0;

  const double fps_ = 144.;

  std::unique_ptr<Engine> engine_;
};
}

#endif // SUPERLUCENT_APPLICATION_H_
