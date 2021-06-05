layout (binding = 1) uniform SimulationParamsUbo
{
	float dt;
	int num_particles;
  float alpha;
} params;
