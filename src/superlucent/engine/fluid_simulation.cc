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

  // Descriptor set layout
  std::vector<vk::DescriptorSetLayoutBinding> bindings(7);
  bindings[0]
    .setBinding(0)
    .setStageFlags(vk::ShaderStageFlagBits::eCompute)
    .setDescriptorType(vk::DescriptorType::eUniformBuffer)
    .setDescriptorCount(1);

  bindings[1]
    .setBinding(1)
    .setStageFlags(vk::ShaderStageFlagBits::eCompute)
    .setDescriptorType(vk::DescriptorType::eStorageBuffer)
    .setDescriptorCount(1);

  bindings[2]
    .setBinding(2)
    .setStageFlags(vk::ShaderStageFlagBits::eCompute)
    .setDescriptorType(vk::DescriptorType::eStorageBuffer)
    .setDescriptorCount(1);

  bindings[3]
    .setBinding(3)
    .setStageFlags(vk::ShaderStageFlagBits::eCompute)
    .setDescriptorType(vk::DescriptorType::eStorageBuffer)
    .setDescriptorCount(1);

  bindings[4]
    .setBinding(4)
    .setStageFlags(vk::ShaderStageFlagBits::eCompute)
    .setDescriptorType(vk::DescriptorType::eStorageBuffer)
    .setDescriptorCount(1);

  bindings[5]
    .setBinding(5)
    .setStageFlags(vk::ShaderStageFlagBits::eCompute)
    .setDescriptorType(vk::DescriptorType::eStorageBuffer)
    .setDescriptorCount(1);

  bindings[6]
    .setBinding(6)
    .setStageFlags(vk::ShaderStageFlagBits::eCompute)
    .setDescriptorType(vk::DescriptorType::eStorageBuffer)
    .setDescriptorCount(1);

  vk::DescriptorSetLayoutCreateInfo descriptor_set_layout_create_info;
  descriptor_set_layout_create_info
    .setBindings(bindings);
  descriptor_set_layout_ = device.createDescriptorSetLayout(descriptor_set_layout_create_info);

  // Pipeline layout
  vk::PipelineLayoutCreateInfo pipeline_layout_create_info;
  pipeline_layout_create_info
    .setSetLayouts(descriptor_set_layout_);
  pipeline_layout_ = device.createPipelineLayout(pipeline_layout_create_info);

  // Pipelines
  const std::string base_directory = "C:\\workspace\\superlucent\\src\\superlucent\\shader\\";
  auto forward_module = engine_->CreateShaderModule(base_directory + "fluid_forward.comp.spv");
  auto initialize_uniform_grid_module = engine_->CreateShaderModule(base_directory + "fluid_initialize_uniform_grid.comp.spv");
  auto add_uniform_grid_module = engine_->CreateShaderModule(base_directory + "fluid_add_uniform_grid.comp.spv");
  auto initialize_neighbors_module = engine_->CreateShaderModule(base_directory + "fluid_initialize_neighbors.comp.spv");
  auto find_neighbors_module = engine_->CreateShaderModule(base_directory + "fluid_find_neighbors.comp.spv");
  auto calculate_lambda_module = engine_->CreateShaderModule(base_directory + "fluid_calculate_lambda.comp.spv");
  auto calculate_dp_collision_response_module = engine_->CreateShaderModule(base_directory + "fluid_calculate_dp_collision_response.comp.spv");
  auto update_p_module = engine_->CreateShaderModule(base_directory + "fluid_update_p.comp.spv");
  auto update_v_module = engine_->CreateShaderModule(base_directory + "fluid_update_v.comp.spv");

  vk::PipelineShaderStageCreateInfo shader_stage_create_info;
  shader_stage_create_info
    .setPName("main")
    .setStage(vk::ShaderStageFlagBits::eCompute);

  vk::ComputePipelineCreateInfo pipeline_create_info;
  pipeline_create_info
    .setLayout(pipeline_layout_);

  shader_stage_create_info.setModule(forward_module);
  pipeline_create_info.setStage(shader_stage_create_info);
  forward_pipeline_ = CreateComputePipeline(pipeline_create_info);

  shader_stage_create_info.setModule(initialize_uniform_grid_module);
  pipeline_create_info.setStage(shader_stage_create_info);
  initialize_uniform_grid_pipeline_ = CreateComputePipeline(pipeline_create_info);

  shader_stage_create_info.setModule(add_uniform_grid_module);
  pipeline_create_info.setStage(shader_stage_create_info);
  add_uniform_grid_pipeline_ = CreateComputePipeline(pipeline_create_info);

  shader_stage_create_info.setModule(initialize_neighbors_module);
  pipeline_create_info.setStage(shader_stage_create_info);
  initialize_neighbors_pipeline_ = CreateComputePipeline(pipeline_create_info);

  shader_stage_create_info.setModule(find_neighbors_module);
  pipeline_create_info.setStage(shader_stage_create_info);
  find_neighbors_pipeline_ = CreateComputePipeline(pipeline_create_info);

  shader_stage_create_info.setModule(calculate_lambda_module);
  pipeline_create_info.setStage(shader_stage_create_info);
  calculate_lambda_pipeline_ = CreateComputePipeline(pipeline_create_info);

  shader_stage_create_info.setModule(calculate_dp_collision_response_module);
  pipeline_create_info.setStage(shader_stage_create_info);
  calculate_dp_collision_response_pipeline_ = CreateComputePipeline(pipeline_create_info);

  shader_stage_create_info.setModule(update_p_module);
  pipeline_create_info.setStage(shader_stage_create_info);
  update_p_pipeline_ = CreateComputePipeline(pipeline_create_info);

  shader_stage_create_info.setModule(update_v_module);
  pipeline_create_info.setStage(shader_stage_create_info);
  update_v_pipeline_ = CreateComputePipeline(pipeline_create_info);

  device.destroyShaderModule(forward_module);
  device.destroyShaderModule(initialize_uniform_grid_module);
  device.destroyShaderModule(add_uniform_grid_module);
  device.destroyShaderModule(initialize_neighbors_module);
  device.destroyShaderModule(find_neighbors_module);
  device.destroyShaderModule(calculate_lambda_module);
  device.destroyShaderModule(calculate_dp_collision_response_module);
  device.destroyShaderModule(update_p_module);
  device.destroyShaderModule(update_v_module);
}

void FluidSimulation::DestroyPipelines()
{
  const auto device = engine_->Device();

  device.destroyPipeline(forward_pipeline_);
  device.destroyPipeline(initialize_uniform_grid_pipeline_);
  device.destroyPipeline(add_uniform_grid_pipeline_);
  device.destroyPipeline(initialize_neighbors_pipeline_);
  device.destroyPipeline(find_neighbors_pipeline_);
  device.destroyPipeline(calculate_lambda_pipeline_);
  device.destroyPipeline(calculate_dp_collision_response_pipeline_);
  device.destroyPipeline(update_p_pipeline_);
  device.destroyPipeline(update_v_pipeline_);

  device.destroyDescriptorSetLayout(descriptor_set_layout_);
  device.destroyPipelineLayout(pipeline_layout_);
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

vk::Pipeline FluidSimulation::CreateComputePipeline(vk::ComputePipelineCreateInfo& create_info)
{
  const auto device = engine_->Device();

  auto result = device.createComputePipeline(pipeline_cache_, create_info);
  if (result.result != vk::Result::eSuccess)
    throw std::runtime_error("Failed to create compute pipeline, with error code: " + vk::to_string(result.result));
  return result.value;
}
}
}
