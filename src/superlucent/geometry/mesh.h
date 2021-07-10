#ifndef SUPERLUCENT_GEOMETRY_MESH_H_
#define SUPERLUCENT_GEOMETRY_MESH_H_

#include <vector>

#include <glm/glm.hpp>

namespace supl
{
namespace geom
{
class Mesh
{
public:
  struct Vertex
  {
    glm::vec3 position{ 0.f };
    glm::vec3 normal{ 0.f };
    glm::vec2 tex_coord{ 0.f };
  };

public:
  Mesh();
  ~Mesh();

  Mesh& AddVertex(const Vertex& vertex);
  Mesh& AddVertex(Vertex&& vertex);

  Mesh& AddFace(const glm::uvec3& face);
  Mesh& AddFace(glm::uvec3&& face);

  void SetHasNormal() { has_normal_ = true; }
  void SetHasTexture() { has_texture_ = true; }

  const auto& Vertices() const { return vertices_; }
  const auto& Faces() const { return faces_; }
  auto HasNormal() const { return has_normal_; }
  auto HasTexture() const { return has_texture_; }

private:
  bool has_normal_ = false;
  bool has_texture_ = false;

  std::vector<Vertex> vertices_;
  std::vector<glm::uvec3> faces_;
};
}
}

#endif // SUPERLUCENT_GEOMETRY_MESH_H_
