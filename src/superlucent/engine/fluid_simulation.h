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

  // Resources
  vk::Buffer particle_buffer_;
  vk::Buffer storage_buffer_;

  std::vector<Uniform> fluid_simulation_params_ubos_;
  FluidSimulationParamsUbo fluid_simulation_params_;
};
}
}

#endif // SUPERLUCENT_ENGINE_FLUID_SIMULATION_H_
