#include <superlucent/engine/fluid_simulation.h>

#include <superlucent/engine/engine.h>
#include <superlucent/engine/uniform_buffer.h>
#include <superlucent/engine/data/particle.h>
#include <superlucent/utils/rng.h>

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
  // Barrier to make sure previous rendering command
  // TODO: triple buffering as well as for particle buffers
  vk::BufferMemoryBarrier particle_buffer_memory_barrier;
  particle_buffer_memory_barrier
    .setSrcAccessMask(vk::AccessFlagBits::eVertexAttributeRead)
    .setDstAccessMask(vk::AccessFlagBits::eShaderWrite)
    .setSrcQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
    .setDstQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
    .setBuffer(particle_buffer_)
    .setOffset(0)
    .setSize(NumParticles() * sizeof(float) * 24);
  command_buffer.pipelineBarrier(vk::PipelineStageFlagBits::eVertexInput, vk::PipelineStageFlagBits::eComputeShader, {},
    {}, particle_buffer_memory_barrier, {});

  // Prepare compute shaders
  command_buffer.bindDescriptorSets(vk::PipelineBindPoint::eCompute, pipeline_layout_, 0u,
    descriptor_sets_[ubo_index], {});

  // Forward
  command_buffer.bindPipeline(vk::PipelineBindPoint::eCompute, forward_pipeline_);
  command_buffer.dispatch((NumParticles() + 255) / 256, 1, 1);

  particle_buffer_memory_barrier
    .setSrcAccessMask(vk::AccessFlagBits::eShaderWrite)
    .setDstAccessMask(vk::AccessFlagBits::eShaderRead);

  command_buffer.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eComputeShader, {},
    {}, particle_buffer_memory_barrier, {});

  // Initialize uniform grid
  command_buffer.bindPipeline(vk::PipelineBindPoint::eCompute, initialize_uniform_grid_pipeline_);
  command_buffer.dispatch((num_hash_buckets + 255) / 256, 1, 1);

  vk::BufferMemoryBarrier grid_barrier;
  grid_barrier
    .setSrcAccessMask(vk::AccessFlagBits::eShaderWrite)
    .setDstAccessMask(vk::AccessFlagBits::eShaderRead)
    .setSrcQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
    .setDstQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
    .setBuffer(storage_buffer_)
    .setOffset(grid_buffer_.offset)
    .setSize(grid_buffer_.size);

  vk::BufferMemoryBarrier hash_table_barrier;
  hash_table_barrier
    .setSrcAccessMask(vk::AccessFlagBits::eShaderWrite)
    .setDstAccessMask(vk::AccessFlagBits::eShaderRead)
    .setSrcQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
    .setDstQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
    .setBuffer(storage_buffer_)
    .setOffset(hash_table_buffer_.offset)
    .setSize(hash_table_buffer_.size);

  command_buffer.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eComputeShader, {},
    {}, { grid_barrier, hash_table_barrier }, {});

  // Add uniform grid
  command_buffer.bindPipeline(vk::PipelineBindPoint::eCompute, add_uniform_grid_pipeline_);
  command_buffer.dispatch((NumParticles() + 255) / 256, 1, 1);

  command_buffer.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eComputeShader, {},
    {}, { grid_barrier, hash_table_barrier }, {});

  // DEBUG particle color changed
  command_buffer.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eComputeShader, {},
    {}, particle_buffer_memory_barrier, {});

  // Initialize neighbors
  command_buffer.bindPipeline(vk::PipelineBindPoint::eCompute, initialize_neighbors_pipeline_);
  command_buffer.dispatch((fluid_simulation_params_.max_num_neighbors + 255) / 256, 1, 1);

  vk::BufferMemoryBarrier neighbors_barrier;
  neighbors_barrier
    .setSrcAccessMask(vk::AccessFlagBits::eShaderWrite)
    .setDstAccessMask(vk::AccessFlagBits::eShaderRead)
    .setSrcQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
    .setDstQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
    .setBuffer(storage_buffer_)
    .setOffset(neighbors_buffer_.offset)
    .setSize(neighbors_buffer_.size);

  vk::BufferMemoryBarrier neighbors_heads_barrier;
  neighbors_heads_barrier
    .setSrcAccessMask(vk::AccessFlagBits::eShaderWrite)
    .setDstAccessMask(vk::AccessFlagBits::eShaderRead)
    .setSrcQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
    .setDstQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
    .setBuffer(storage_buffer_)
    .setOffset(neighbors_heads_buffer_.offset)
    .setSize(neighbors_heads_buffer_.size);

  command_buffer.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eComputeShader, {},
    {}, { neighbors_barrier, neighbors_heads_barrier }, {});

  // Find neighbors
  /*
  command_buffer.bindPipeline(vk::PipelineBindPoint::eCompute, find_neighbors_pipeline_);
  // command_buffer.dispatch((NumParticles() + 255) / 256, 1, 1);
  command_buffer.dispatch(1, 1, 1);

  command_buffer.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eComputeShader, {},
    {}, { neighbors_barrier, neighbors_heads_barrier }, {});
    */

  // Update v
  command_buffer.bindPipeline(vk::PipelineBindPoint::eCompute, update_v_pipeline_);
  command_buffer.dispatch((NumParticles() + 255) / 256, 1, 1);

  particle_buffer_memory_barrier
    .setSrcAccessMask(vk::AccessFlagBits::eShaderWrite)
    .setDstAccessMask(vk::AccessFlagBits::eVertexAttributeRead);

  command_buffer.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eVertexInput, {},
    {}, particle_buffer_memory_barrier, {});
}

void FluidSimulation::UpdateSimulationParams(double dt, double animation_time, int ubo_index)
{
  constexpr auto wall_offset_speed = 5.;
  constexpr auto wall_offset_magnitude = 0.5;

  fluid_simulation_params_.dt = dt;
  fluid_simulation_params_.epsilon = 1e-6f;
  fluid_simulation_params_.wall_offset = static_cast<float>(wall_offset_magnitude * std::sin(animation_time * wall_offset_speed));

  fluid_simulation_params_ubos_[ubo_index] = fluid_simulation_params_;
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

  // Particles
  fluid_simulation_params_.radius = 0.03f;
  fluid_simulation_params_.h = 0.15f;

  constexpr int cell_count = 40;
  const float radius = fluid_simulation_params_.radius;
  constexpr float density = 1000.f; // water
  const float mass = radius * radius * radius * density;
  constexpr glm::vec2 wall_distance = glm::vec2(3.f, 1.5f);
  const glm::vec3 particle_offset = glm::vec3(-wall_distance + glm::vec2(radius * 1.1f), radius * 1.1f);
  const glm::vec3 particle_stride = glm::vec3(radius * 2.2f);

  fluid_simulation_params_.rest_density = density;

  utils::Rng rng;
  constexpr float noise_range = 1e-2f;
  const auto noise = [&rng, noise_range]() { return rng.Uniform(-noise_range, noise_range); };

  glm::vec3 gravity = glm::vec3(0.f, 0.f, -9.8f);
  std::vector<Particle> particle_buffer;
  for (int i = 0; i < cell_count; i++)
  {
    for (int j = 0; j < cell_count; j++)
    {
      for (int k = 0; k < cell_count; k++)
      {
        glm::vec4 position{
          particle_offset.x + particle_stride.x * i + noise(),
          particle_offset.y + particle_stride.y * j + noise(),
          particle_offset.z + particle_stride.z * k + noise(),
          0.f
        };
        glm::vec4 velocity{ 0.f };
        glm::vec4 properties{ mass, 0.f, 0.f, 0.f };
        glm::vec4 external_force{
          gravity.x * mass,
          gravity.y * mass,
          gravity.z * mass,
          0.f
        };
        glm::vec4 color{ 0.5f, 0.5f, 0.5f, 0.f };

        // Struct initialization
        particle_buffer.push_back({ position, position, velocity, properties, external_force, color });
      }
    }
  }
  const auto particle_buffer_size = particle_buffer.size() * sizeof(Particle);
  fluid_simulation_params_.num_particles = particle_buffer.size();

  constexpr auto max_num_neighbors_per_perticle = 30;
  fluid_simulation_params_.max_num_neighbors = max_num_neighbors_per_perticle * fluid_simulation_params_.num_particles;
  neighbors_buffer_.offset = 0;
  neighbors_buffer_.size = sizeof(uint32_t) + sizeof(glm::vec4) * fluid_simulation_params_.max_num_neighbors;

  const auto ssbo_alignment = engine_->SsboAlignment();
  neighbors_heads_buffer_.offset = engine_->Align(neighbors_buffer_.offset + neighbors_buffer_.size, ssbo_alignment);
  neighbors_heads_buffer_.size = sizeof(uint32_t) * fluid_simulation_params_.num_particles;

  solver_buffer_.offset = engine_->Align(neighbors_heads_buffer_.offset + neighbors_heads_buffer_.size, ssbo_alignment);
  solver_buffer_.size = sizeof(glm::vec4) * fluid_simulation_params_.num_particles;

  grid_buffer_.offset = engine_->Align(solver_buffer_.offset + solver_buffer_.size, ssbo_alignment);
  grid_buffer_.size = sizeof(glm::vec4) + sizeof(glm::vec2) * fluid_simulation_params_.num_particles * 8;

  hash_table_buffer_.offset = engine_->Align(grid_buffer_.offset + grid_buffer_.size, ssbo_alignment);
  hash_table_buffer_.size = sizeof(int32_t) * num_hash_buckets;

  const auto storage_buffer_size = hash_table_buffer_.offset + hash_table_buffer_.size;

  // Buffers
  vk::BufferCreateInfo buffer_create_info;
  buffer_create_info
    .setUsage(vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eStorageBuffer)
    .setSize(particle_buffer_size);
  particle_buffer_ = device.createBuffer(buffer_create_info);

  buffer_create_info
    .setUsage(vk::BufferUsageFlagBits::eStorageBuffer)
    .setSize(storage_buffer_size);
  storage_buffer_ = device.createBuffer(buffer_create_info);

  // Memory binding
  const auto particle_buffer_memory = engine_->AcquireDeviceMemory(particle_buffer_);
  device.bindBufferMemory(particle_buffer_, particle_buffer_memory.memory, particle_buffer_memory.offset);

  const auto storage_buffer_memory = engine_->AcquireDeviceMemory(storage_buffer_);
  device.bindBufferMemory(storage_buffer_, storage_buffer_memory.memory, storage_buffer_memory.offset);

  // Staging
  engine_->ToDeviceMemory(particle_buffer, particle_buffer_);

  // Descriptor set
  const auto uniform_buffer = engine_->UniformBuffer();
  fluid_simulation_params_ubos_ = uniform_buffer->Allocate<FluidSimulationParamsUbo>(num_ubos_);

  std::vector<vk::DescriptorSetLayout> layouts(num_ubos_, descriptor_set_layout_);
  vk::DescriptorSetAllocateInfo decriptor_set_allocate_info;
  decriptor_set_allocate_info
    .setDescriptorPool(engine_->DescriptorPool())
    .setSetLayouts(layouts);
  descriptor_sets_ = device.allocateDescriptorSets(decriptor_set_allocate_info);

  for (int i = 0; i < num_ubos_; i++)
  {
    std::vector<vk::DescriptorBufferInfo> buffer_infos(7);
    buffer_infos[0]
      .setBuffer(uniform_buffer->Buffer())
      .setOffset(fluid_simulation_params_ubos_[i].offset)
      .setRange(fluid_simulation_params_ubos_[i].size);

    buffer_infos[1]
      .setBuffer(particle_buffer_)
      .setOffset(0)
      .setRange(fluid_simulation_params_.num_particles * sizeof(Particle));

    buffer_infos[2]
      .setBuffer(storage_buffer_)
      .setOffset(neighbors_buffer_.offset)
      .setRange(neighbors_buffer_.size);

    buffer_infos[3]
      .setBuffer(storage_buffer_)
      .setOffset(neighbors_heads_buffer_.offset)
      .setRange(neighbors_heads_buffer_.size);

    buffer_infos[4]
      .setBuffer(storage_buffer_)
      .setOffset(solver_buffer_.offset)
      .setRange(solver_buffer_.size);

    buffer_infos[5]
      .setBuffer(storage_buffer_)
      .setOffset(grid_buffer_.offset)
      .setRange(grid_buffer_.size);

    buffer_infos[6]
      .setBuffer(storage_buffer_)
      .setOffset(hash_table_buffer_.offset)
      .setRange(hash_table_buffer_.size);

    std::vector<vk::WriteDescriptorSet> descriptor_writes(7);
    descriptor_writes[0]
      .setDstSet(descriptor_sets_[i])
      .setDstBinding(0)
      .setBufferInfo(buffer_infos[0])
      .setDescriptorType(vk::DescriptorType::eUniformBuffer)
      .setDstArrayElement(0)
      .setDescriptorCount(1);

    descriptor_writes[1]
      .setDstSet(descriptor_sets_[i])
      .setDstBinding(1)
      .setBufferInfo(buffer_infos[1])
      .setDescriptorType(vk::DescriptorType::eStorageBuffer)
      .setDstArrayElement(0)
      .setDescriptorCount(1);

    descriptor_writes[2]
      .setDstSet(descriptor_sets_[i])
      .setDstBinding(2)
      .setBufferInfo(buffer_infos[2])
      .setDescriptorType(vk::DescriptorType::eStorageBuffer)
      .setDstArrayElement(0)
      .setDescriptorCount(1);

    descriptor_writes[3]
      .setDstSet(descriptor_sets_[i])
      .setDstBinding(3)
      .setBufferInfo(buffer_infos[3])
      .setDescriptorType(vk::DescriptorType::eStorageBuffer)
      .setDstArrayElement(0)
      .setDescriptorCount(1);

    descriptor_writes[4]
      .setDstSet(descriptor_sets_[i])
      .setDstBinding(4)
      .setBufferInfo(buffer_infos[4])
      .setDescriptorType(vk::DescriptorType::eStorageBuffer)
      .setDstArrayElement(0)
      .setDescriptorCount(1);

    descriptor_writes[5]
      .setDstSet(descriptor_sets_[i])
      .setDstBinding(5)
      .setBufferInfo(buffer_infos[5])
      .setDescriptorType(vk::DescriptorType::eStorageBuffer)
      .setDstArrayElement(0)
      .setDescriptorCount(1);

    descriptor_writes[6]
      .setDstSet(descriptor_sets_[i])
      .setDstBinding(6)
      .setBufferInfo(buffer_infos[6])
      .setDescriptorType(vk::DescriptorType::eStorageBuffer)
      .setDstArrayElement(0)
      .setDescriptorCount(1);

    device.updateDescriptorSets(descriptor_writes, {});
  }
}

void FluidSimulation::DestroyResources()
{
  const auto device = engine_->Device();

  descriptor_sets_.clear();
  device.destroyBuffer(particle_buffer_);
  device.destroyBuffer(storage_buffer_);
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
