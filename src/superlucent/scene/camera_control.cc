#include <superlucent/scene/camera_control.h>

#include <algorithm>

#include <superlucent/scene/camera.h>

namespace supl
{
namespace scene
{
CameraControl::CameraControl(std::shared_ptr<scene::Camera> camera)
  : camera_(camera)
{
  Update();
}

void CameraControl::Update()
{
  const auto cos_theta = std::cos(theta_);
  const auto sin_theta = std::sin(theta_);
  const auto cos_phi = std::cos(phi_);
  const auto sin_phi = std::sin(phi_);

  const glm::vec3 eye = center_ + radius_ * glm::vec3(cos_theta * cos_phi, sin_theta * cos_phi, sin_phi);

  camera_->LookAt(eye, center_, up_);
}

void CameraControl::TranslateByPixels(int dx, int dy)
{
  const auto cos_theta = std::cos(theta_);
  const auto sin_theta = std::sin(theta_);
  const auto cos_phi = std::cos(phi_);
  const auto sin_phi = std::sin(phi_);

  const glm::vec3 x = radius_ * glm::vec3(-sin_theta, cos_theta, 0.f);
  const glm::vec3 y = radius_ * glm::vec3(cos_theta * -sin_phi, sin_theta * -sin_phi, cos_phi);

  center_ += translation_sensitivity_ * (-x * static_cast<float>(dx) + y * static_cast<float>(dy));
}

void CameraControl::RotateByPixels(int dx, int dy)
{
  constexpr float epsilon = 1e-3f;
  constexpr auto phi_limit = glm::pi<float>() / 2.f - epsilon;

  theta_ -= rotation_sensitivity_ * dx;
  phi_ = std::clamp(phi_ + rotation_sensitivity_ * dy, -phi_limit, phi_limit);
}

void CameraControl::ZoomByPixels(int dx, int dy)
{
  radius_ *= std::pow(1.f + zoom_multiplier_, zoom_sensitivity_ * dy);
}

void CameraControl::ZoomByWheel(int scroll)
{
  radius_ /= std::pow(1.f + zoom_multiplier_, zoom_wheel_sensitivity_ * scroll);
}

void CameraControl::MoveForward(float dt)
{
  const auto cos_theta = std::cos(theta_);
  const auto sin_theta = std::sin(theta_);
  const auto cos_phi = std::cos(phi_);
  const auto sin_phi = std::sin(phi_);

  const auto forward = radius_ * -glm::vec3(cos_theta * cos_phi, sin_theta * cos_phi, sin_phi);

  center_ += move_speed_ * forward * dt;
}

void CameraControl::MoveRight(float dt)
{
  const auto cos_theta = std::cos(theta_);
  const auto sin_theta = std::sin(theta_);

  const glm::vec3 x = radius_ * glm::vec3(-sin_theta, cos_theta, 0.f);

  center_ += move_speed_ * x * dt;
}

void CameraControl::MoveUp(float dt)
{
  center_ += move_speed_ * (radius_ * up_) * dt;
}
}
}
