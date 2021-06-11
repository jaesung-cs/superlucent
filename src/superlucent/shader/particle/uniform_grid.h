const int num_hash_buckets = 1000003;

layout (binding = 5) buffer GridSsbo
{
  vec3 cell_size;
  uint num_pairs;

  uvec2 object_grid_pairs[]; // [object_id, hash]
} grid;

// From Particle-based Fluid Simulation based Fluid Simulation by NVidia
uint GridHash(ivec3 cell_index)
{
  const uint p1 = 73856093; // some large primes
  const uint p2 = 19349663;
  const uint p3 = 83492791;
  uint n = (p1 * cell_index.x) ^ (p2 * cell_index.y) ^ (p3 * cell_index.z);
  n %= num_hash_buckets;

  // Allow hash collision
  return n;
}

ivec3 CellIndex(vec3 position)
{
  return ivec3(floor(position / grid.cell_size.xyz));
}

mat2x3 Bound(vec3 position)
{
  uvec3 cell_index = CellIndex(position);
  mat2x3 bound;
  bound[0] = cell_index * grid.cell_size.xyz;
  bound[1] = bound[0] + grid.cell_size.xyz;
  return bound;
}

void AddSphereToGrid(uint object_id, vec3 position, float radius)
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
        vec3 nearest = vec3(nearests[x][0], nearests[y][1], nearests[z][2]);
        vec3 d = position - nearest;
        if (dot(d, d) <= radius * radius)
        {
          ivec3 neighbor_cell_index = ivec3(cell_index.x + (x - 1), cell_index.y + (y - 1), cell_index.z + (z - 1));
          uint hash = GridHash(neighbor_cell_index);

          uint object_grid_pair_index = atomicAdd(grid.num_pairs, 1);
          grid.object_grid_pairs[object_grid_pair_index] = uvec2(object_id, hash);
        }
      }
    }
  }
}
