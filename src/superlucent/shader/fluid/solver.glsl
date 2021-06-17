struct SolverComponent
{
  float lambda;
  vec3 dp;
};

layout (binding = 4) buffer SolverSsbo
{
  SolverComponent solver[];
};
