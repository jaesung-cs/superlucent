#include <superlucent/geometry/mesh.h>

namespace supl
{
namespace geom
{
Mesh::Mesh() = default;

Mesh::~Mesh() = default;

Mesh& Mesh::AddVertex(const Vertex& vertex)
{
  vertices_.push_back(vertex);
  return *this;
}

Mesh& Mesh::AddVertex(Vertex&& vertex)
{
  vertices_.emplace_back(std::move(vertex));
  return *this;
}

Mesh& Mesh::AddFace(const glm::uvec3& face)
{
  faces_.push_back(face);
  return *this;
}

Mesh& Mesh::AddFace(glm::uvec3&& face)
{
  faces_.emplace_back(std::move(face));
  return *this;
}
}
}
