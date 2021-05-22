#ifndef SUPERLUCENT_SCENE_CAMERA_CONTROL_H_
#define SUPERLUCENT_SCENE_CAMERA_CONTROL_H_

#include <memory>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

namespace supl
{
namespace scene
{
class Camera;

class CameraControl
{
public:
  CameraControl() = delete;
  explicit CameraControl(std::shared_ptr<Camera> camera);
  ~CameraControl() = default;

  auto Camera() const { return camera_; }

  void Update();

  void TranslateByPixels(int dx, int dy);
  void RotateByPixels(int dx, int dy);
  void ZoomByPixels(int dx, int dy);
  void ZoomByWheel(int scroll);

  void MoveForward(float dt);
  void MoveRight(float dt);
  void MoveUp(float dt);

private:
  std::shared_ptr<scene::Camera> camera_;

  // pos = center + radius * (cos theta cos phi, sin theta cos phi, sin phi)
  glm::vec3 center_{ 0.f, 0.f, 0.f };
  const glm::vec3 up_{ 0.f, 0.f, 1.f };
  float radius_ = 5.f;
  float theta_ = -glm::pi<float>() / 4.f;
  float phi_ = glm::pi<float>() / 4.f;

  float translation_sensitivity_ = 0.003f;
  float rotation_sensitivity_ = 0.003f;
  float zoom_sensitivity_ = 0.1f;
  float zoom_wheel_sensitivity_ = 5.f;
  float zoom_multiplier_ = 0.01f;
  float move_speed_ = 1.f;
};
}
}

#endif // SUPERLUCENT_SCENE_CAMERA_CONTROL_H_
