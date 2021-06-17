// Fluid solver
layout (binding = 4) buffer SolverSsbo
{
  // rows = num_particles * 3
  float matrix[];
} solver;

uint Size()
{
  return uint(params.num_particles * 3);
}
