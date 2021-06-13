#ifndef SUPERLUCENT_ENGINE_PARTICLE_SIMULATION_H_
#define SUPERLUCENT_ENGINE_PARTICLE_SIMULATION_H_

#include <vulkan/vulkan.hpp>

namespace supl
{
namespace engine
{
// Compute binding 1
struct SimulationParamsUbo
{
  alignas(16) float dt;
  int num_particles;
  float alpha; // compliance of the constraints
  float wall_offset; // wall x direction distance is added with this value
};

class ParticleSimulation
{
public:

  // TODO: change to private members
public:
  struct Uniform
  {
    vk::DeviceSize offset;
    vk::DeviceSize size;
  };

  vk::DescriptorSetLayout descriptor_set_layout;
  vk::PipelineLayout pipeline_layout;
  vk::Pipeline forward_pipeline;
  vk::Pipeline initialize_uniform_grid_pipeline;
  vk::Pipeline add_uniform_grid_pipeline;
  vk::Pipeline initialize_collision_detection_pipeline;
  vk::Pipeline collision_detection_pipeline;
  vk::Pipeline initialize_dispatch_pipeline;
  vk::Pipeline initialize_solver_pipeline;
  vk::Pipeline solve_delta_lambda_pipeline;
  vk::Pipeline solve_delta_x_pipeline;
  vk::Pipeline solve_x_lambda_pipeline;
  vk::Pipeline velocity_update_pipeline;

  vk::Buffer particle_buffer;
  uint32_t num_particles;

  uint32_t num_collisions;

  // Storage buffer (with only storage buffer usage) has
  // - Collision pairs
  // - Solver matrices: lambda, x, delta_lambda, delta_x
  // - Uniform grid
  vk::Buffer storage_buffer;
  vk::DeviceSize collision_pairs_buffer_offset;
  vk::DeviceSize collision_pairs_buffer_size;
  vk::DeviceSize solver_buffer_offset;
  vk::DeviceSize solver_buffer_size;
  vk::DeviceSize grid_buffer_offset;
  vk::DeviceSize grid_buffer_size;
  static constexpr int num_hash_buckets = 1000003;
  vk::DeviceSize hash_table_buffer_offset;
  static constexpr vk::DeviceSize hash_table_buffer_size = num_hash_buckets * sizeof(int32_t);
  vk::DeviceSize collision_chain_buffer_offset;
  vk::DeviceSize collision_chain_buffer_size;

  // Dispatch indirect buffer
  vk::Buffer dispatch_indirect;

  std::vector<vk::DescriptorSet> descriptor_sets;
  std::vector<Uniform> simulation_params_ubos;

  SimulationParamsUbo simulation_params;
};
}
}

#endif // SUPERLUCENT_ENGINE_PARTICLE_SIMULATION_H_
