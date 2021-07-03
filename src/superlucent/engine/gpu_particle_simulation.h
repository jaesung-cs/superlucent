#ifndef SUPERLUCENT_ENGINE_GPU_PARTICLE_SIMULATION_H_
#define SUPERLUCENT_ENGINE_GPU_PARTICLE_SIMULATION_H_

#include <vulkan/vulkan.hpp>

#include <superlucent/engine/ubo/particle_simulation_params.h>

namespace supl
{
namespace engine
{
class Engine;
class Uniform;

class GpuParticleSimulation
{
public:
  GpuParticleSimulation() = delete;

  explicit GpuParticleSimulation(Engine* engine, int num_ubos);

  ~GpuParticleSimulation();

  const auto& SimulationParams() const { return simulation_params_; }
  auto ParticleBuffer() const { return particle_buffer_; }
  auto NumParticles() const { return num_particles_; }

  void RecordComputeWithGraphicsBarriers(vk::CommandBuffer& command_buffer, int ubo_index);
  void UpdateSimulationParams(double dt, double animation_time, int ubo_index);

private:
  void CreateComputePipelines();
  void DestroyComputePipelines();

  void PrepareResources();
  void DestroyResources();

  vk::Pipeline CreateComputePipeline(vk::ComputePipelineCreateInfo& create_info);

  Engine* const engine_;

  vk::DescriptorSetLayout descriptor_set_layout_;
  vk::PipelineLayout pipeline_layout_;
  vk::PipelineCache pipeline_cache_;
  vk::Pipeline forward_pipeline_;
  vk::Pipeline initialize_uniform_grid_pipeline_;
  vk::Pipeline add_uniform_grid_pipeline_;
  vk::Pipeline initialize_collision_detection_pipeline_;
  vk::Pipeline collision_detection_pipeline_;
  vk::Pipeline initialize_dispatch_pipeline_;
  vk::Pipeline initialize_solver_pipeline_;vk::Pipeline solve_delta_lambda_pipeline_;
  vk::Pipeline solve_delta_x_pipeline_;
  vk::Pipeline solve_x_lambda_pipeline_;
  vk::Pipeline velocity_update_pipeline_;

  vk::Buffer particle_buffer_;
  uint32_t num_particles_;

  uint32_t num_collisions_;

  // Storage buffer (with only storage buffer usage) has
  // - Collision pairs
  // - Solver matrices: lambda, x, delta_lambda, delta_x
  // - Uniform grid
  vk::Buffer storage_buffer_;
  vk::DeviceSize collision_pairs_buffer_offset_;
  vk::DeviceSize collision_pairs_buffer_size_;
  vk::DeviceSize solver_buffer_offset_;
  vk::DeviceSize solver_buffer_size_;
  vk::DeviceSize grid_buffer_offset_;
  vk::DeviceSize grid_buffer_size_;
  static constexpr int num_hash_buckets = 1000003;
  vk::DeviceSize hash_table_buffer_offset_;
  static constexpr vk::DeviceSize hash_table_buffer_size_ = num_hash_buckets * sizeof(int32_t);
  vk::DeviceSize collision_chain_buffer_offset_;
  vk::DeviceSize collision_chain_buffer_size_;

    // Dispatch indirect buffer
  vk::Buffer dispatch_indirect_;

  // Uniform buffer
  uint32_t num_ubos_ = 0;
  std::vector<vk::DescriptorSet> descriptor_sets_;
  std::vector<Uniform> simulation_params_ubos_;
  ParticleSimulationParamsUbo simulation_params_;
};
}
}

#endif // SUPERLUCENT_ENGINE_GPU_PARTICLE_SIMULATION_H_
