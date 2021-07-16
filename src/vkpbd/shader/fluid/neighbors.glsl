#ifndef VKPBD_FLUID_NEIGHBORS_GLSL_
#define VKPBD_FLUID_NEIGHBORS_GLSL_

#include "fluid_simulation_params.glsl"

layout (binding = 3) buffer NeighborsSsbo
{
  // [0, n): particles
  // [n, n + nm): particle ids
  int buf[];
} neighbors;

int NeighborCount(uint particle_index)
{
  return neighbors.buf[particle_index];
}

int NeighborIndex(uint particle_index, uint index)
{
  if (neighbors.buf[particle_index] < params.max_num_neighbors)
    return neighbors.buf[params.num_particles + particle_index * params.max_num_neighbors + index];
  else
    return -1;
}

void AddNeighbor(uint particle_index, uint neighbor_index)
{
  uint index = atomicAdd(neighbors.buf[particle_index], 1);
  if (index < params.max_num_neighbors)
    neighbors.buf[params.num_particles + particle_index * params.max_num_neighbors + index] = int(neighbor_index);
}

#endif // VKPBD_FLUID_NEIGHBORS_GLSL_
