struct Neighbor
{
  int index;
  int next; // index in neighbors[] pointing next neighbor
  ivec2 pad;
};

layout (binding = 2) buffer NeighborsSsbo
{
  int num_neighbors;
  Neighbor neighbors[];
};

layout (binding = 3) buffer NeighborsHeadsSsbo
{
  int neighbors_heads[];
};
