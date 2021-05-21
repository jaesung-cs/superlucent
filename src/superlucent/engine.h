#ifndef SUPERLUCENT_ENGINE_H_
#define SUPERLUCENT_ENGINE_H_

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
  const int max_width_;
  const int max_height_;

  int width_ = 0;
  int height_ = 0;
};
}

#endif // SUPERLUCENT_ENGINE_H_
