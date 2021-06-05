layout (binding = 3) buffer SolverSsbo
{
  // rows = num_collisions
  // cols = num_particles * 3
  // J: [rows * cols], C: [cols], lambda: [rows], x: [cols]
  // Pack by vec4
  vec4 matrix[];
} solver;

uint Rows()
{
  return num_collisions;
}

uint Cols()
{
  return params.num_particles * 3;
}

uvec2 LambdaIndex(uint row)
{
  uint index = row;
  return uvec2(index / 4, index % 4);
}

uvec2 XIndex(uint row)
{
  uint index = Rows() + row;
  return uvec2(index / 4, index % 4);
}

uvec2 DeltaLambdaIndex(uint row)
{
  uint index = Rows() + Cols() + row;
  return uvec2(index / 4, index % 4);
}

uvec2 DeltaXIndex(uint row)
{
  uint index = Rows() + Cols() + Rows() + row;
  return uvec2(index / 4, index % 4);
}
