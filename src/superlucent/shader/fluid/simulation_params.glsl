layout (binding = 0) uniform FluidSimulatinoParamsSsbo
{
  float dt;
  int num_particles;
  float epsilon;
  float wall_offset;

  float h;
  float radius;
  float rest_density;
  float c;

  float k;
  int n;
  int max_num_neighbors;
} params;
