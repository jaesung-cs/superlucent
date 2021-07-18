#ifndef VKPBD_FLUID_UNIFORM_GRID_GLSL_
#define VKPBD_FLUID_UNIFORM_GRID_GLSL_

struct Node
{
  uint object_id;
  int next;
};

const int num_hash_buckets = 1000003;

layout (binding = 4) buffer GridSsbo
{
  vec3 cell_size;
  uint num_pairs;
  
  int hash_table_head[num_hash_buckets];
  int pad;

  Node object_grid_pairs[];
} grid;

// From Particle-based Fluid Simulation based Fluid Simulation by NVidia
uint GridHash(ivec3 cell_index)
{
  // TODO: why make positive?
  uvec3 ucell_index = uvec3(cell_index + vec3(100.f, 100.f, 100.f));
  const uint p1 = 73856093; // some large primes
  const uint p2 = 19349663;
  const uint p3 = 83492791;
  uint n = (p1 * ucell_index.x) ^ (p2 * ucell_index.y) ^ (p3 * ucell_index.z);
  n %= num_hash_buckets;

  // Allow hash collision
  return n;
}

ivec3 CellIndex(vec3 position)
{
  return ivec3(floor(position / grid.cell_size));
}

mat2x3 Bound(vec3 position)
{
  ivec3 cell_index = CellIndex(position);
  mat2x3 bound;
  bound[0] = cell_index * grid.cell_size;
  bound[1] = bound[0] + grid.cell_size;
  return bound;
}

void AddSphereToGrid(uint object_id, vec3 position, float d)
{
  ivec3 cell_index = CellIndex(position);
  mat2x3 bound = Bound(position);
  mat3 nearests = mat3(bound[0], position, bound[1]);

  for (int x = 0; x < 3; x++)
  {
    for (int y = 0; y < 3; y++)
    {
      for (int z = 0; z < 3; z++)
      {
        vec3 nearest = vec3(nearests[x].x, nearests[y].y, nearests[z].z);
        vec3 dist = position - nearest;
        if (dot(dist, dist) <= d * d)
        {
          ivec3 neighbor_cell_index = cell_index + ivec3(x - 1, y - 1, z - 1);
          uint hash_index = GridHash(neighbor_cell_index);

          // Append object-grid pair
          uint object_grid_pair_index = atomicAdd(grid.num_pairs, 1);
          grid.object_grid_pairs[object_grid_pair_index].object_id = object_id;

          // Exchange hash table head
          int next = atomicExchange(grid.hash_table_head[hash_index], int(object_grid_pair_index));
          grid.object_grid_pairs[object_grid_pair_index].next = next;
        }
      }
    }
  }
}

#endif // VKPBD_FLUID_UNIFORM_GRID_GLSL_
