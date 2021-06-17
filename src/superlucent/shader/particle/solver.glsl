layout (binding = 3) buffer SolverSsbo
{
  // rows = num_collisions
  // cols = num_particles * 3
  // J: [rows * cols], C: [cols], lambda: [rows], x: [cols]
  float matrix[];
} solver;

uint Rows()
{
  return num_collisions;
}

uint Cols()
{
  return params.num_particles * 3;
}

uint LambdaIndex(uint row)
{
  return row;
}

uint XIndex(uint row)
{
  return Rows() + row;
}

uint DeltaLambdaIndex(uint row)
{
  return Rows() + Cols() + row;
}

uint DeltaXIndex(uint row)
{
  return Rows() + Cols() + Rows() + row;
}
