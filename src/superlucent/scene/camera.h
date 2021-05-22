#ifndef SUPERLUCENT_SCENE_CAMERA_H_
#define SUPERLUCENT_SCENE_CAMERA_H_

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

namespace supl
{
namespace scene
{
class Camera
{
private:
  enum class Type
  {
    PERSPECTIVE,
    ORTHO,
  };

public:
  Camera() = default;
  ~Camera() = default;

  void SetPerspective();
  void SetOrtho();

  void SetFovy(float fovy);
  void SetZoom(float zoom);
  void SetNearFar(float near, float far);
  void SetScreenSize(int width, int height);

  void LookAt(const glm::vec3& eye, const glm::vec3& center, const glm::vec3& up);

  glm::mat4 ProjectionMatrix() const;
  glm::mat4 ViewMatrix() const;

  const auto& Eye() const { return eye_; }
  const auto& Center() const { return center_; }
  const auto& Up() const { return up_; }

private:
  Type type_ = Type::PERSPECTIVE;

  int width_ = 1;
  int height_ = 1;

  float near_ = 0.01f;
  float far_ = 100.f;

  float fovy_ = 60.f / 180.f * glm::pi<float>();

  float zoom_ = 1.f;

  glm::vec3 eye_{};
  glm::vec3 center_{};
  glm::vec3 up_{};
};
}
}

#endif // SUPERLUCENT_SCENE_CAMERA_H_
