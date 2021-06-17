#ifndef SUPERLUCENT_ENGINE_FLUID_SIMULATION_H_
#define SUPERLUCENT_ENGINE_FLUID_SIMULATION_H_

#include <vulkan/vulkan.hpp>

#include <superlucent/engine/ubo/fluid_simulation_params.h>

namespace supl
{
namespace engine
{
class Engine;
class Uniform;

class FluidSimulation
{
public:
  FluidSimulation() = delete;

  explicit FluidSimulation(Engine* engine, int num_ubos);

  ~FluidSimulation();

  void RecordComputeWithGraphicsBarriers(vk::CommandBuffer& command_buffer, int ubo_index);
  void UpdateSimulationParams(double dt, double animation_time, int ubo_index);

private:
  void CreatePipelines();
  void DestroyPipelines();
  void PrepareResources();
  void DestroyResources();

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
  Buffer neighbors_buffer;
  Buffer neighbors_heads_buffer;
  Buffer solver_buffer;
  Buffer grid_buffer;
  static constexpr uint32_t num_hash_buckets = 1000003;
  Buffer hash_table_buffer;

  std::vector<Uniform> fluid_simulation_params_ubos_;
  FluidSimulationParamsUbo fluid_simulation_params_;
};
}
}

#endif // SUPERLUCENT_ENGINE_FLUID_SIMULATION_H_
