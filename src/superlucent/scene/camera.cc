#include <superlucent/scene/camera.h>

namespace supl
{
namespace scene
{
void Camera::SetPerspective()
{
  type_ = Type::PERSPECTIVE;
}

void Camera::SetOrtho()
{
  type_ = Type::ORTHO;
}

void Camera::SetFovy(float fovy)
{
  fovy_ = fovy;
}

void Camera::SetZoom(float zoom)
{
  zoom_ = zoom;
}

void Camera::SetNearFar(float near, float far)
{
  near_ = near;
  far_ = far;
}

void Camera::SetScreenSize(int width, int height)
{
  width_ = width;
  height_ = height;
}

void Camera::LookAt(const glm::vec3& eye, const glm::vec3& center, const glm::vec3& up)
{
  eye_ = eye;
  center_ = center;
  up_ = up;
}

glm::mat4 Camera::ProjectionMatrix() const
{
  const auto aspect = static_cast<float>(width_) / height_;

  switch (type_)
  {
  case Type::PERSPECTIVE:
    return glm::perspective(fovy_, aspect, near_, far_);
  case Type::ORTHO:
    return glm::ortho(-aspect * zoom_, aspect * zoom_, -zoom_, zoom_, near_, far_);
  default:
    return glm::mat4(1.f);
  }
}

glm::mat4 Camera::ViewMatrix() const
{
  return glm::lookAt(eye_, center_, up_);
}
}
}
