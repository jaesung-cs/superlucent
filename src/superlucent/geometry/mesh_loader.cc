#include <superlucent/geometry/mesh_loader.h>

#include <assimp/Importer.hpp>
#include <assimp/postprocess.h>
#include <assimp/scene.h>

#include <glm/glm.hpp>

#include <superlucent/geometry/mesh.h>

namespace supl
{
namespace geom
{
namespace
{
Mesh LoadMesh(aiMesh* imported_mesh)
{
  Mesh mesh;

  if (imported_mesh->HasNormals())
    mesh.SetHasNormal();

  if (imported_mesh->HasTextureCoords(0))
    mesh.SetHasTexture();

  for (int i = 0; i < imported_mesh->mNumVertices; i++)
  {
    Mesh::Vertex vertex;

    vertex.position = {
      imported_mesh->mVertices[i].x,
      imported_mesh->mVertices[i].y,
      imported_mesh->mVertices[i].z,
    };

    if (imported_mesh->HasNormals())
    {
      vertex.normal = {
        imported_mesh->mNormals[i].x,
        imported_mesh->mNormals[i].y,
        imported_mesh->mNormals[i].z,
      };
    }

    // TODO: multiple texture coord sets
    if (imported_mesh->HasTextureCoords(0))
    {
      vertex.tex_coord = {
        imported_mesh->mTextureCoords[0][i].x,
        imported_mesh->mTextureCoords[0][i].y,
      };
    }

    mesh.AddVertex(vertex);
  }

  for (int i = 0; i < imported_mesh->mNumFaces; i++)
  {
    mesh.AddFace({
      imported_mesh->mFaces->mIndices[0],
      imported_mesh->mFaces->mIndices[1],
      imported_mesh->mFaces->mIndices[2],
    });
  }

  return mesh;
}
}

MeshLoader::MeshLoader(const std::string& filepath)
  : filepath_{ filepath }
{
}

MeshLoader::~MeshLoader() = default;

std::vector<Mesh> MeshLoader::Load()
{
  Assimp::Importer importer;
  const auto scene = importer.ReadFile(filepath_, aiPostProcessSteps::aiProcess_Triangulate);

  std::vector<Mesh> meshes;
  if (scene->HasMeshes())
  {
    for (int i = 0; i < scene->mNumMeshes; i++)
      meshes.emplace_back(LoadMesh(scene->mMeshes[i]));
  }

  return meshes;
}
}
}
