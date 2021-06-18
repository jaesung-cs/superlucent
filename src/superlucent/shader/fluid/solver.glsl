struct SolverComponent
{
  vec3 dp;
  float lambda;
};

layout (binding = 4) buffer SolverSsbo
{
  SolverComponent solver[];
};
