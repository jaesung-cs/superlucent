#ifndef SUPERLUCENT_GEOMETRY_MESH_LOADER_H_
#define SUPERLUCENT_GEOMETRY_MESH_LOADER_H_

#include <string>
#include <vector>

namespace supl
{
namespace geom
{
class Mesh;

class MeshLoader
{
public:
  MeshLoader() = delete;

  explicit MeshLoader(const std::string& filepath);

  ~MeshLoader();

  std::vector<Mesh> Load();

private:
  std::string filepath_;
};
}
}

#endif // SUPERLUCENT_GEOMETRY_MESH_LOADER_H_
