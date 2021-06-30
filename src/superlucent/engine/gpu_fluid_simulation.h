#ifndef SUPERLUCENT_ENGINE_GPU_FLUID_SIMULATION_H_
#define SUPERLUCENT_ENGINE_GPU_FLUID_SIMULATION_H_

#include <vulkan/vulkan.hpp>

#include <superlucent/engine/ubo/fluid_simulation_params.h>

namespace supl
{
namespace engine
{
class Engine;
class Uniform;

class GpuFluidSimulation
{
public:
  GpuFluidSimulation() = delete;

  explicit GpuFluidSimulation(Engine* engine, int num_ubos);

  ~GpuFluidSimulation();

  auto ParticleBuffer() const { return particle_buffer_; }
  auto NumParticles() const { return fluid_simulation_params_.num_particles; }
  const auto& SimulationParams() const { return fluid_simulation_params_; }

  void RecordComputeWithGraphicsBarriers(vk::CommandBuffer& command_buffer, int ubo_index);
  void UpdateSimulationParams(double dt, double animation_time, int ubo_index);

private:
  void CreatePipelines();
  void DestroyPipelines();
  void PrepareResources();
  void DestroyResources();

  vk::Pipeline CreateComputePipeline(vk::ComputePipelineCreateInfo& create_info);

  Engine* const engine_;
  const int num_ubos_;

  // Pipelines
  vk::DescriptorSetLayout descriptor_set_layout_;
  vk::PipelineLayout pipeline_layout_;
  vk::PipelineCache pipeline_cache_;
  vk::Pipeline forward_pipeline_;
  vk::Pipeline initialize_uniform_grid_pipeline_;
  vk::Pipeline add_uniform_grid_pipeline_;
  vk::Pipeline initialize_neighbors_pipeline_;
  vk::Pipeline find_neighbors_pipeline_;
  vk::Pipeline calculate_lambda_pipeline_;
  vk::Pipeline calculate_dp_collision_response_pipeline_;
  vk::Pipeline update_p_pipeline_;
  vk::Pipeline update_v_pipeline_;

  // Resources
  struct Buffer
  {
    vk::DeviceSize offset;
    vk::DeviceSize size;
  };

  // Binding 0: simulation params
  // Binding 1: particles
  // Binding 2: neighbors
  // Binding 3: neighbor heads
  // Binding 4: solver
  // Binding 5: grid
  // Binding 6: hash table heads

  vk::Buffer particle_buffer_;
  vk::Buffer storage_buffer_;
  Buffer neighbors_buffer_;
  Buffer neighbors_heads_buffer_;
  Buffer solver_buffer_;
  Buffer grid_buffer_;
  static constexpr uint32_t num_hash_buckets = 1000003;
  Buffer hash_table_buffer_;

  std::vector<vk::DescriptorSet> descriptor_sets_;

  std::vector<Uniform> fluid_simulation_params_ubos_;
  FluidSimulationParamsUbo fluid_simulation_params_;
};
}
}

#endif // SUPERLUCENT_ENGINE_GPU_FLUID_SIMULATION_H_