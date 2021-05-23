#ifndef SUPERLUCENT_SCENE_LIGHT_H_
#define SUPERLUCENT_SCENE_LIGHT_H_

#include <glm/glm.hpp>

namespace supl
{
namespace scene
{
class Light
{
private:
  enum class Type
  {
    DIRECTIONAL,
    POINT,
  };

public:
  Light() = default;
  ~Light() = default;

  void SetDirectionalLight() { type_ = Type::DIRECTIONAL; }
  void SetPointLight() { type_ = Type::POINT; }
  void SetPosition(const glm::vec3& position) { position_ = position; }
  void SetAmbient(const glm::vec3& ambient) { ambient_ = ambient; }
  void SetDiffuse(const glm::vec3& diffuse) { diffuse_ = diffuse; }
  void SetSpecular(const glm::vec3& specular) { specular_ = specular; }

  bool IsDirectionalLight() const { return type_ == Type::DIRECTIONAL; }
  bool IsPointLight() const { return type_ == Type::POINT; }
  const glm::vec3& Position() const { return position_; }
  const glm::vec3& Ambient() const { return ambient_; }
  const glm::vec3& Diffuse() const { return diffuse_; }
  const glm::vec3& Specular() const { return specular_; }

private:
  Type type_;
  glm::vec3 position_{ 0.f, 0.f, 1.f };
  glm::vec3 ambient_{ 0.f, 0.f, 0.f };
  glm::vec3 diffuse_{ 0.f, 0.f, 0.f };
  glm::vec3 specular_{ 0.f, 0.f, 0.f };
};
}
}

#endif // SUPERLUCENT_SCENE_LIGHT_H_
