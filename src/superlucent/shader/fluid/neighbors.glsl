struct Neighbor
{
  int index;
  float r; // distance between particles

  int next; // index in neighbors[] pointing next neighbor
  int pad;
};

layout (binding = 2) uniform NeighborsSsbo
{
  int neighbors_count;
  Neighbor neighbors[];
};

layout (binding = 3) uniform NeighborsHeadsSsbo
{
  int neighbors_heads[];
};
