#ifndef SUPERLUDENTH_GEOMETRY_TETRAGRID_H_
#define SUPERLUDENTH_GEOMETRY_TETRAGRID_H_

#include <vector>

#include <glm/glm.hpp>

namespace supl
{
namespace geom
{
class Mesh;

class Tetragrid
{
public:
  struct Grid
  {
    glm::vec3 position;
  };

  struct MeshOptions
  {
    float distance;
  };

public:
  Tetragrid();
  ~Tetragrid();

  void FromMesh(const Mesh& mesh, const MeshOptions& options);

private:
  std::vector<Grid> grids_;
};
}
}

#endif // SUPERLUDENTH_GEOMETRY_TETRAGRID_H_
