struct CollisionPairs
{
  ivec2 ids; // [i0, i1]
  ivec2 next; // [i0 < i1, i0 > i1]
  mat2x4 r; // r[0/1] = [rx, ry, rz, 0]
  vec4 n; // [nx, ny, nz, 0]
};

layout (binding = 3) buffer CollisionPairsSsbo
{
  uint num_collisions;
  CollisionPairs collisions[];
};

layout (binding = 4) buffer CollisionPairsLinkedListSsbo
{
  ivec2 head[]; // [i0 < i1, i0 > i1]
} collision_chain;
