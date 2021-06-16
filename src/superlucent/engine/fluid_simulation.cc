#include <superlucent/engine/fluid_simulation.h>

#include <superlucent/engine/engine.h>
#include <superlucent/engine/uniform_buffer.h>

namespace supl
{
namespace engine
{
FluidSimulation::FluidSimulation(Engine* engine, int num_ubos)
  : engine_(engine)
  , num_ubos_(num_ubos)
{
  CreatePipelines();
  PrepareResources();
}

FluidSimulation::~FluidSimulation()
{
  DestroyResources();
  DestroyPipelines();
}

void FluidSimulation::RecordComputeWithGraphicsBarriers(vk::CommandBuffer& command_buffer, int ubo_index)
{
}

void FluidSimulation::UpdateSimulationParams(double dt, double animation_time, int ubo_index)
{
}

void FluidSimulation::CreatePipelines()
{
  const auto device = engine_->Device();

  // Create pipeline cache
  pipeline_cache_ = device.createPipelineCache({});
}

void FluidSimulation::DestroyPipelines()
{
  const auto device = engine_->Device();

  device.destroyPipelineCache(pipeline_cache_);
}

void FluidSimulation::PrepareResources()
{
  const auto device = engine_->Device();
}

void FluidSimulation::DestroyResources()
{
  const auto device = engine_->Device();
}
}
}
