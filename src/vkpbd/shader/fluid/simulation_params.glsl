layout (binding = 7) uniform SimulationParamsUbo
{
	float dt;
	int num_particles;
  float alpha;
  float wall_offset;

  float radius;
} params;
