#include <superlucent/engine/particle_simulation.h>

#include <superlucent/engine/engine.h>
#include <superlucent/engine/uniform_buffer.h>
#include <superlucent/utils/rng.h>

namespace supl
{
namespace engine
{
ParticleSimulation::ParticleSimulation(Engine* engine, int num_ubos)
  : engine_(engine)
  , num_ubos_(num_ubos)
{
  CreateComputePipelines();
  PrepareResources();
}

ParticleSimulation::~ParticleSimulation()
{
  DestroyResources();
  DestroyComputePipelines();
}

void ParticleSimulation::UpdateSimulationParams(double dt, double animation_time, int ubo_index)
{
  const auto uniform_buffer = engine_->UniformBuffer();

  constexpr auto wall_offset_speed = 5.;
  constexpr auto wall_offset_magnitude = 0.5;

  simulation_params_.dt = static_cast<float>(dt);
  simulation_params_.num_particles = num_particles_;
  simulation_params_.alpha = 0.001f;
  simulation_params_.wall_offset = static_cast<float>(wall_offset_magnitude * std::sin(animation_time * wall_offset_speed));

  std::memcpy(uniform_buffer->Map() + simulation_params_ubos_[ubo_index].offset, &simulation_params_, sizeof(SimulationParamsUbo));
}

void ParticleSimulation::RecordComputeWithGraphicsBarriers(vk::CommandBuffer& command_buffer, int ubo_index)
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
    .setSize(num_particles_ * sizeof(float) * 24);
  command_buffer.pipelineBarrier(vk::PipelineStageFlagBits::eVertexInput, vk::PipelineStageFlagBits::eComputeShader, {},
    {}, particle_buffer_memory_barrier, {});

  // Prepare compute shaders
  command_buffer.bindDescriptorSets(vk::PipelineBindPoint::eCompute, pipeline_layout_, 0u,
    descriptor_sets_[ubo_index], {});

  // Forward
  command_buffer.bindPipeline(vk::PipelineBindPoint::eCompute, forward_pipeline_);
  command_buffer.dispatch((num_particles_ + 255) / 256, 1, 1);

  particle_buffer_memory_barrier
    .setSrcAccessMask(vk::AccessFlagBits::eShaderWrite)
    .setDstAccessMask(vk::AccessFlagBits::eShaderRead);

  command_buffer.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eComputeShader, {},
    {}, particle_buffer_memory_barrier, {});

  // Initialize uniform grid
  command_buffer.bindPipeline(vk::PipelineBindPoint::eCompute, initialize_uniform_grid_pipeline_);
  command_buffer.dispatch((num_hash_buckets + 255) / 256, 1, 1);

  vk::BufferMemoryBarrier grid_buffer_memory_barrier;
  grid_buffer_memory_barrier
    .setSrcAccessMask(vk::AccessFlagBits::eShaderWrite)
    .setDstAccessMask(vk::AccessFlagBits::eShaderRead)
    .setSrcQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
    .setDstQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
    .setBuffer(storage_buffer_)
    .setOffset(grid_buffer_offset_)
    .setSize(grid_buffer_size_);

  vk::BufferMemoryBarrier hash_table_buffer_memory_barrier;
  hash_table_buffer_memory_barrier
    .setSrcAccessMask(vk::AccessFlagBits::eShaderWrite)
    .setDstAccessMask(vk::AccessFlagBits::eShaderRead)
    .setSrcQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
    .setDstQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
    .setBuffer(storage_buffer_)
    .setOffset(hash_table_buffer_offset_)
    .setSize(hash_table_buffer_size_);

  command_buffer.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eComputeShader, {},
    {}, { grid_buffer_memory_barrier, hash_table_buffer_memory_barrier }, {});

  // Add to uniform grid
  command_buffer.bindPipeline(vk::PipelineBindPoint::eCompute, add_uniform_grid_pipeline_);
  command_buffer.dispatch((num_particles_ + 255) / 256, 1, 1);

  command_buffer.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eComputeShader, {},
    {}, { grid_buffer_memory_barrier, hash_table_buffer_memory_barrier }, {});

  // Initialize collision detection
  command_buffer.bindPipeline(vk::PipelineBindPoint::eCompute, initialize_collision_detection_pipeline_);
  command_buffer.dispatch((num_particles_ + 255) / 256, 1, 1);

  vk::BufferMemoryBarrier collision_pairs_buffer_memory_barrier;
  collision_pairs_buffer_memory_barrier
    .setSrcAccessMask(vk::AccessFlagBits::eShaderWrite)
    .setDstAccessMask(vk::AccessFlagBits::eShaderRead)
    .setSrcQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
    .setDstQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
    .setBuffer(storage_buffer_)
    .setOffset(collision_pairs_buffer_offset_)
    .setSize(collision_pairs_buffer_size_);

  vk::BufferMemoryBarrier collision_chain_buffer_memory_barrier;
  collision_chain_buffer_memory_barrier
    .setSrcAccessMask(vk::AccessFlagBits::eShaderWrite)
    .setDstAccessMask(vk::AccessFlagBits::eShaderRead)
    .setSrcQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
    .setDstQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
    .setBuffer(storage_buffer_)
    .setOffset(collision_chain_buffer_offset_)
    .setSize(collision_chain_buffer_size_);

  command_buffer.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eComputeShader, {},
    {}, { collision_pairs_buffer_memory_barrier, collision_chain_buffer_memory_barrier }, {});

  // Collision detection
  command_buffer.bindPipeline(vk::PipelineBindPoint::eCompute, collision_detection_pipeline_);
  command_buffer.dispatch((num_particles_ + 255) / 256, 1, 1);

  command_buffer.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eComputeShader, {},
    {}, { collision_pairs_buffer_memory_barrier, collision_chain_buffer_memory_barrier }, {});

  // In collision detection, particle color is written for debug purpose
  command_buffer.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eComputeShader, {},
    {}, particle_buffer_memory_barrier, {});

  // Initialize dispatch
  command_buffer.bindPipeline(vk::PipelineBindPoint::eCompute, initialize_dispatch_pipeline_);
  command_buffer.dispatch(1, 1, 1);

  vk::MemoryBarrier dispatch_indirect_barrier;
  dispatch_indirect_barrier
    .setSrcAccessMask(vk::AccessFlagBits::eShaderWrite)
    .setDstAccessMask(vk::AccessFlagBits::eIndirectCommandRead);

  // Why draw indirect stage, not top of pipe or compute?
  command_buffer.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eDrawIndirect, {},
    dispatch_indirect_barrier, {}, {});

  // Initialize solver
  command_buffer.bindPipeline(vk::PipelineBindPoint::eCompute, initialize_solver_pipeline_);
  command_buffer.dispatchIndirect(dispatch_indirect_, 0);

  vk::BufferMemoryBarrier solver_buffer_memory_barrier;
  solver_buffer_memory_barrier
    .setSrcAccessMask(vk::AccessFlagBits::eShaderWrite)
    .setDstAccessMask(vk::AccessFlagBits::eShaderRead)
    .setSrcQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
    .setDstQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
    .setBuffer(storage_buffer_)
    .setOffset(solver_buffer_offset_)
    .setSize(solver_buffer_size_);

  command_buffer.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eComputeShader, {},
    {}, solver_buffer_memory_barrier, {});

  // Solve
  constexpr int solver_iterations = 1;
  for (int i = 0; i < solver_iterations; i++)
  {
    // Solve delta lambda
    command_buffer.bindPipeline(vk::PipelineBindPoint::eCompute, solve_delta_lambda_pipeline_);
    command_buffer.dispatchIndirect(dispatch_indirect_, sizeof(uint32_t) * 4);

    command_buffer.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eComputeShader, {},
      {}, solver_buffer_memory_barrier, {});

    // Solve delta x
    command_buffer.bindPipeline(vk::PipelineBindPoint::eCompute, solve_delta_x_pipeline_);
    command_buffer.dispatch((num_particles_ + 255) / 256, 1, 1);

    command_buffer.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eComputeShader, {},
      {}, solver_buffer_memory_barrier, {});

    // Solve x and lambda
    command_buffer.bindPipeline(vk::PipelineBindPoint::eCompute, solve_x_lambda_pipeline_);
    command_buffer.dispatchIndirect(dispatch_indirect_, 0);

    command_buffer.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eComputeShader, {},
      {}, solver_buffer_memory_barrier, {});
  }

  // Velocity update
  command_buffer.bindPipeline(vk::PipelineBindPoint::eCompute, velocity_update_pipeline_);
  command_buffer.dispatch((num_particles_ + 255) / 256, 1, 1);

  particle_buffer_memory_barrier
    .setSrcAccessMask(vk::AccessFlagBits::eShaderWrite)
    .setDstAccessMask(vk::AccessFlagBits::eVertexAttributeRead);
  command_buffer.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eVertexInput, {},
    {}, particle_buffer_memory_barrier, {});
}

void ParticleSimulation::CreateComputePipelines()
{
  const auto device = engine_->Device();

  // Create pipeline cache
  pipeline_cache_ = device.createPipelineCache({});

  // Descriptor set layout
  std::vector<vk::DescriptorSetLayoutBinding> bindings(8);
  bindings[0]
    .setBinding(0)
    .setStageFlags(vk::ShaderStageFlagBits::eCompute)
    .setDescriptorType(vk::DescriptorType::eStorageBuffer)
    .setDescriptorCount(1);

  bindings[1]
    .setBinding(1)
    .setStageFlags(vk::ShaderStageFlagBits::eCompute)
    .setDescriptorType(vk::DescriptorType::eUniformBuffer)
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

  bindings[7]
    .setBinding(7)
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

  // Shader modules
  const std::string base_dir = "C:\\workspace\\superlucent\\src\\superlucent\\shader";
  vk::ShaderModule particle_forward_module = engine_->CreateShaderModule(base_dir + "\\particle_forward.comp.spv");
  vk::ShaderModule particle_initialize_uniform_grid_module = engine_->CreateShaderModule(base_dir + "\\particle_initialize_uniform_grid.comp.spv");
  vk::ShaderModule particle_add_uniform_grid_module = engine_->CreateShaderModule(base_dir + "\\particle_add_uniform_grid.comp.spv");
  vk::ShaderModule particle_initialize_collision_detection_module = engine_->CreateShaderModule(base_dir + "\\particle_initialize_collision_detection.comp.spv");
  vk::ShaderModule particle_collision_detection_module = engine_->CreateShaderModule(base_dir + "\\particle_collision_detection.comp.spv");
  vk::ShaderModule particle_initialize_dispatch_module = engine_->CreateShaderModule(base_dir + "\\particle_initialize_dispatch.comp.spv");
  vk::ShaderModule particle_initialize_solver_module = engine_->CreateShaderModule(base_dir + "\\particle_initialize_solver.comp.spv");
  vk::ShaderModule particle_solve_delta_lambda_module = engine_->CreateShaderModule(base_dir + "\\particle_solve_delta_lambda.comp.spv");
  vk::ShaderModule particle_solve_delta_x_module = engine_->CreateShaderModule(base_dir + "\\particle_solve_delta_x.comp.spv");
  vk::ShaderModule particle_solve_x_lambda_module = engine_->CreateShaderModule(base_dir + "\\particle_solve_x_lambda.comp.spv");
  vk::ShaderModule particle_velocity_update_module = engine_->CreateShaderModule(base_dir + "\\particle_velocity_update.comp.spv");

  // Forward
  vk::PipelineShaderStageCreateInfo shader_stage;
  shader_stage
    .setStage(vk::ShaderStageFlagBits::eCompute)
    .setModule(particle_forward_module)
    .setPName("main");

  vk::ComputePipelineCreateInfo compute_pipeline_create_info;
  compute_pipeline_create_info
    .setStage(shader_stage)
    .setLayout(pipeline_layout_);
  forward_pipeline_ = CreateComputePipeline(compute_pipeline_create_info);

  // Initialize uniform grid
  shader_stage.setModule(particle_initialize_uniform_grid_module);
  compute_pipeline_create_info.setStage(shader_stage);
  initialize_uniform_grid_pipeline_ = CreateComputePipeline(compute_pipeline_create_info);

  // Add uniform grid
  shader_stage.setModule(particle_add_uniform_grid_module);
  compute_pipeline_create_info.setStage(shader_stage);
  add_uniform_grid_pipeline_ = CreateComputePipeline(compute_pipeline_create_info);

  // Initialize collision detection
  shader_stage.setModule(particle_initialize_collision_detection_module);
  compute_pipeline_create_info.setStage(shader_stage);
  initialize_collision_detection_pipeline_ = CreateComputePipeline(compute_pipeline_create_info);

  // Collision detection
  shader_stage.setModule(particle_collision_detection_module);
  compute_pipeline_create_info.setStage(shader_stage);
  collision_detection_pipeline_ = CreateComputePipeline(compute_pipeline_create_info);

  // Initialize dispatch
  shader_stage.setModule(particle_initialize_dispatch_module);
  compute_pipeline_create_info.setStage(shader_stage);
  initialize_dispatch_pipeline_ = CreateComputePipeline(compute_pipeline_create_info);

  // Initialize solver
  shader_stage.setModule(particle_initialize_solver_module);
  compute_pipeline_create_info.setStage(shader_stage);
  initialize_solver_pipeline_ = CreateComputePipeline(compute_pipeline_create_info);

  // Solve delta lambda
  shader_stage.setModule(particle_solve_delta_lambda_module);
  compute_pipeline_create_info.setStage(shader_stage);
  solve_delta_lambda_pipeline_ = CreateComputePipeline(compute_pipeline_create_info);

  // Solve delta x
  shader_stage.setModule(particle_solve_delta_x_module);
  compute_pipeline_create_info.setStage(shader_stage);
  solve_delta_x_pipeline_ = CreateComputePipeline(compute_pipeline_create_info);

  // Solve x lambda
  shader_stage.setModule(particle_solve_x_lambda_module);
  compute_pipeline_create_info.setStage(shader_stage);
  solve_x_lambda_pipeline_ = CreateComputePipeline(compute_pipeline_create_info);

  // Velocity update
  shader_stage.setModule(particle_velocity_update_module);
  compute_pipeline_create_info.setStage(shader_stage);
  velocity_update_pipeline_ = CreateComputePipeline(compute_pipeline_create_info);

  device.destroyShaderModule(particle_forward_module);
  device.destroyShaderModule(particle_initialize_uniform_grid_module);
  device.destroyShaderModule(particle_add_uniform_grid_module);
  device.destroyShaderModule(particle_initialize_collision_detection_module);
  device.destroyShaderModule(particle_collision_detection_module);
  device.destroyShaderModule(particle_initialize_dispatch_module);
  device.destroyShaderModule(particle_initialize_solver_module);
  device.destroyShaderModule(particle_solve_delta_lambda_module);
  device.destroyShaderModule(particle_solve_delta_x_module);
  device.destroyShaderModule(particle_solve_x_lambda_module);
  device.destroyShaderModule(particle_velocity_update_module);
}

void ParticleSimulation::DestroyComputePipelines()
{
  const auto device = engine_->Device();

  // Destroy pipelines
  device.destroyDescriptorSetLayout(descriptor_set_layout_);
  device.destroyPipelineLayout(pipeline_layout_);

  device.destroyPipeline(forward_pipeline_);
  device.destroyPipeline(initialize_uniform_grid_pipeline_);
  device.destroyPipeline(add_uniform_grid_pipeline_);
  device.destroyPipeline(initialize_collision_detection_pipeline_);
  device.destroyPipeline(collision_detection_pipeline_);
  device.destroyPipeline(initialize_dispatch_pipeline_);
  device.destroyPipeline(initialize_solver_pipeline_);
  device.destroyPipeline(solve_delta_lambda_pipeline_);
  device.destroyPipeline(solve_delta_x_pipeline_);
  device.destroyPipeline(solve_x_lambda_pipeline_);
  device.destroyPipeline(velocity_update_pipeline_);

  device.destroyPipelineCache(pipeline_cache_);
}

void ParticleSimulation::PrepareResources()
{
  const auto device = engine_->Device();

  // Prticles
  constexpr int cell_count = 40;
  constexpr float radius = 0.03f;
  constexpr float density = 1000.f; // water
  constexpr float mass = radius * radius * radius * density;
  constexpr glm::vec2 wall_distance = glm::vec2(3.f, 1.5f);
  constexpr glm::vec3 particle_offset = glm::vec3(-wall_distance + glm::vec2(radius * 1.1f), radius * 1.1f);
  constexpr glm::vec3 particle_stride = glm::vec3(radius * 2.2f);

  utils::Rng rng;
  constexpr float noise_range = 1e-2f;
  const auto noise = [&rng, noise_range]() { return rng.Uniform(-noise_range, noise_range); };

  glm::vec3 gravity = glm::vec3(0.f, 0.f, -9.8f);
  std::vector<float> particle_buffer;
  uint32_t num_particles = 0;
  for (int i = 0; i < cell_count; i++)
  {
    for (int j = 0; j < cell_count; j++)
    {
      for (int k = 0; k < cell_count; k++)
      {
        num_particles++;

        // prev_position
        particle_buffer.push_back(particle_offset.x + particle_stride.x * i + noise());
        particle_buffer.push_back(particle_offset.y + particle_stride.y * j + noise());
        particle_buffer.push_back(particle_offset.z + particle_stride.z * k + noise());
        particle_buffer.push_back(0.f);

        // position
        particle_buffer.push_back(particle_offset.x + particle_stride.x * i + noise());
        particle_buffer.push_back(particle_offset.y + particle_stride.y * j + noise());
        particle_buffer.push_back(particle_offset.z + particle_stride.z * k + noise());
        particle_buffer.push_back(0.f);

        // velocity
        particle_buffer.push_back(0.f);
        particle_buffer.push_back(0.f);
        particle_buffer.push_back(0.f);
        particle_buffer.push_back(0.f);

        // properties
        particle_buffer.push_back(radius);
        particle_buffer.push_back(mass);
        particle_buffer.push_back(0.f);
        particle_buffer.push_back(0.f);

        // external_force
        particle_buffer.push_back(gravity.x * mass);
        particle_buffer.push_back(gravity.y * mass);
        particle_buffer.push_back(gravity.z * mass);
        particle_buffer.push_back(0.f);

        // color
        particle_buffer.push_back(0.5f);
        particle_buffer.push_back(0.5f);
        particle_buffer.push_back(0.5f);
        particle_buffer.push_back(0.f);
      }
    }
  }
  const auto particle_buffer_size = particle_buffer.size() * sizeof(float);

  // Collision and solver size
  num_collisions_ =
    num_particles + 5 // walls
    + num_particles * 12; // max 12 collisions for each sphere
  collision_pairs_buffer_size_ = sizeof(uint32_t) + num_collisions_ * (sizeof(int32_t) * 4 + sizeof(float) * 12);

  solver_buffer_size_ =
    (num_collisions_ // lambda
      + num_particles * 3) // x
    * 2 // delta
    * sizeof(float);

  grid_buffer_size_ =
    16 // 4-element header
    + (sizeof(uint32_t) + sizeof(int32_t)) * (num_particles * 8); // object grid pairs

  collision_chain_buffer_size_ =
    (sizeof(int32_t) * 2) * num_particles;

  collision_pairs_buffer_offset_ = 0;
  solver_buffer_offset_ = engine_->Align(collision_pairs_buffer_offset_ + collision_pairs_buffer_size_, engine_->SsboAlignment());
  grid_buffer_offset_ = engine_->Align(solver_buffer_offset_ + solver_buffer_size_, engine_->SsboAlignment());
  hash_table_buffer_offset_ = engine_->Align(grid_buffer_offset_ + grid_buffer_size_, engine_->SsboAlignment());
  collision_chain_buffer_offset_ = engine_->Align(hash_table_buffer_offset_ + hash_table_buffer_size_, engine_->SsboAlignment());
  const auto storage_buffer_size = collision_chain_buffer_offset_ + collision_chain_buffer_size_;
  
  const auto dispatch_indirect_size = sizeof(uint32_t) * 8;

  // Vulkan buffers
  vk::BufferCreateInfo buffer_create_info;
  buffer_create_info
    .setSize(particle_buffer_size)
    .setUsage(vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eStorageBuffer);
  particle_buffer_ = device.createBuffer(buffer_create_info);
  num_particles_ = num_particles;

  buffer_create_info
    .setSize(storage_buffer_size)
    .setUsage(vk::BufferUsageFlagBits::eStorageBuffer);
  storage_buffer_ = device.createBuffer(buffer_create_info);

  buffer_create_info
    .setSize(dispatch_indirect_size)
    .setUsage(vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eIndirectBuffer);
  dispatch_indirect_ = device.createBuffer(buffer_create_info);

  const auto uniform_buffer = engine_->UniformBuffer();

  // Bind to memory
  const auto particle_memory = engine_->AcquireDeviceMemory(particle_buffer_);
  device.bindBufferMemory(particle_buffer_, particle_memory.memory, particle_memory.offset);

  const auto storage_buffer_memory = engine_->AcquireDeviceMemory(storage_buffer_);
  device.bindBufferMemory(storage_buffer_, storage_buffer_memory.memory, storage_buffer_memory.offset);

  const auto dispatch_indirect_memory = engine_->AcquireDeviceMemory(dispatch_indirect_);
  device.bindBufferMemory(dispatch_indirect_, dispatch_indirect_memory.memory, dispatch_indirect_memory.offset);

  // Staging
  engine_->ToDeviceMemory(particle_buffer, particle_buffer_);

  // Particle descriptor set
  vk::DeviceSize uniform_offset = 0;
  simulation_params_ubos_ = uniform_buffer->Allocate(sizeof(SimulationParamsUbo), num_ubos_);

  std::vector<vk::DescriptorSetLayout> set_layouts(num_ubos_, descriptor_set_layout_);
  vk::DescriptorSetAllocateInfo descriptor_set_allocate_info;
  descriptor_set_allocate_info
    .setDescriptorPool(engine_->DescriptorPool())
    .setSetLayouts(set_layouts);
  descriptor_sets_ = device.allocateDescriptorSets(descriptor_set_allocate_info);

  for (int i = 0; i < num_ubos_; i++)
  {
    std::vector<vk::DescriptorBufferInfo> buffer_infos(8);
    buffer_infos[0]
      .setBuffer(particle_buffer_)
      .setOffset(0)
      .setRange(num_particles * sizeof(float) * 24);

    buffer_infos[1]
      .setBuffer(uniform_buffer->Buffer())
      .setOffset(simulation_params_ubos_[i].offset)
      .setRange(simulation_params_ubos_[i].size);

    buffer_infos[2]
      .setBuffer(storage_buffer_)
      .setOffset(collision_pairs_buffer_offset_)
      .setRange(collision_pairs_buffer_size_);

    buffer_infos[3]
      .setBuffer(storage_buffer_)
      .setOffset(solver_buffer_offset_)
      .setRange(solver_buffer_size_);

    buffer_infos[4]
      .setBuffer(dispatch_indirect_)
      .setOffset(0)
      .setRange(sizeof(uint32_t) * 8);

    buffer_infos[5]
      .setBuffer(storage_buffer_)
      .setOffset(grid_buffer_offset_)
      .setRange(grid_buffer_size_);

    buffer_infos[6]
      .setBuffer(storage_buffer_)
      .setOffset(hash_table_buffer_offset_)
      .setRange(hash_table_buffer_size_);

    buffer_infos[7]
      .setBuffer(storage_buffer_)
      .setOffset(collision_chain_buffer_offset_)
      .setRange(collision_chain_buffer_size_);

    std::vector<vk::WriteDescriptorSet> descriptor_writes(8);
    descriptor_writes[0]
      .setDstSet(descriptor_sets_[i])
      .setDstBinding(0)
      .setDstArrayElement(0)
      .setDescriptorType(vk::DescriptorType::eStorageBuffer)
      .setBufferInfo(buffer_infos[0]);

    descriptor_writes[1]
      .setDstSet(descriptor_sets_[i])
      .setDstBinding(1)
      .setDstArrayElement(0)
      .setDescriptorType(vk::DescriptorType::eUniformBuffer)
      .setBufferInfo(buffer_infos[1]);

    descriptor_writes[2]
      .setDstSet(descriptor_sets_[i])
      .setDstBinding(2)
      .setDstArrayElement(0)
      .setDescriptorType(vk::DescriptorType::eStorageBuffer)
      .setBufferInfo(buffer_infos[2]);

    descriptor_writes[3]
      .setDstSet(descriptor_sets_[i])
      .setDstBinding(3)
      .setDstArrayElement(0)
      .setDescriptorType(vk::DescriptorType::eStorageBuffer)
      .setBufferInfo(buffer_infos[3]);

    descriptor_writes[4]
      .setDstSet(descriptor_sets_[i])
      .setDstBinding(4)
      .setDstArrayElement(0)
      .setDescriptorType(vk::DescriptorType::eStorageBuffer)
      .setBufferInfo(buffer_infos[4]);

    descriptor_writes[5]
      .setDstSet(descriptor_sets_[i])
      .setDstBinding(5)
      .setDstArrayElement(0)
      .setDescriptorType(vk::DescriptorType::eStorageBuffer)
      .setBufferInfo(buffer_infos[5]);

    descriptor_writes[6]
      .setDstSet(descriptor_sets_[i])
      .setDstBinding(6)
      .setDstArrayElement(0)
      .setDescriptorType(vk::DescriptorType::eStorageBuffer)
      .setBufferInfo(buffer_infos[6]);

    descriptor_writes[7]
      .setDstSet(descriptor_sets_[i])
      .setDstBinding(7)
      .setDstArrayElement(0)
      .setDescriptorType(vk::DescriptorType::eStorageBuffer)
      .setBufferInfo(buffer_infos[7]);

    device.updateDescriptorSets(descriptor_writes, {});
  }
}

void ParticleSimulation::DestroyResources()
{
  const auto device = engine_->Device();

  descriptor_sets_.clear();
  device.destroyBuffer(particle_buffer_);
  device.destroyBuffer(storage_buffer_);
  device.destroyBuffer(dispatch_indirect_);
}

vk::Pipeline ParticleSimulation::CreateComputePipeline(vk::ComputePipelineCreateInfo& create_info)
{
  const auto device = engine_->Device();

  auto result = device.createComputePipeline(pipeline_cache_, create_info);
  if (result.result != vk::Result::eSuccess)
    throw std::runtime_error("Failed to create compute pipeline, with error code: " + vk::to_string(result.result));
  return result.value;
}
}
}
