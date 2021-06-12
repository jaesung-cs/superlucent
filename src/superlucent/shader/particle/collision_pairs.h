struct CollisionPairs
{
  ivec2 ids; // [i0, i1]
  int next;
  int pad;
  mat2x4 r; // r[0/1] = [rx, ry, rz, 0]
  vec4 n; // [nx, ny, nz, 0]
};

layout (binding = 2) buffer CollisionPairsSsbo
{
  uint num_collisions;
  CollisionPairs collisions[];
};

layout (binding = 7) buffer CollisionPairsLinkedListSsbo
{
  int head[];
} collision_chain;
