#include <vkpbd/particle_simulator.hpp>

#include <string>
#include <fstream>

#include <vkpbd/particle.h>
#include <vkpbd/simulation_params.h>

namespace vkpbd
{
BufferRequirements ParticleSimulator::getParticleBufferRequirements()
{
  BufferRequirements requirements;
  requirements.usage = vk::BufferUsageFlagBits::eStorageBuffer;
  requirements.size = particleBufferRequiredSize_;
  return requirements;
}

BufferRequirements ParticleSimulator::getInternalBufferRequirements()
{
  BufferRequirements requirements;
  requirements.usage = vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eIndirectBuffer;
  requirements.size = internalBufferRequiredSize_;
  return requirements;
}

BufferRequirements ParticleSimulator::getUniformBufferRequirements()
{
  BufferRequirements requirements;
  requirements.usage = vk::BufferUsageFlagBits::eUniformBuffer;
  requirements.size = uniformBufferRequiredSize_;
  return requirements;
}

void ParticleSimulator::cmdBindSrcParticleBuffer(vk::Buffer buffer, vk::DeviceSize offset)
{
  srcBuffer_.buffer = buffer;
  srcBuffer_.offset = offset;
  srcBuffer_.size = particleBufferRequiredSize_;
}

void ParticleSimulator::cmdBindDstParticleBuffer(vk::Buffer buffer, vk::DeviceSize offset)
{
  dstBuffer_.buffer = buffer;
  dstBuffer_.offset = offset;
  dstBuffer_.size = particleBufferRequiredSize_;
}

void ParticleSimulator::cmdBindInternalBuffer(vk::Buffer buffer, vk::DeviceSize offset)
{
  internalBuffer_.buffer = buffer;
  internalBuffer_.offset = offset;
  internalBuffer_.size = internalBufferRequiredSize_;
}

void ParticleSimulator::cmdBindUniformBuffer(vk::Buffer buffer, vk::DeviceSize offset, uint8_t* map)
{
  uniformBuffer_.buffer = buffer;
  uniformBuffer_.offset = offset;
  uniformBuffer_.size = uniformBufferRequiredSize_;
  uniformBuffer_.map = map;
}

void ParticleSimulator::cmdStep(vk::CommandBuffer commandBuffer, int cmdIndex, float animationTime, float dt)
{
  cmdStep(commandBuffer, cmdIndex, particleCount_, animationTime, dt);
}

void ParticleSimulator::cmdStep(vk::CommandBuffer commandBuffer, int cmdIndex, uint32_t particleCount, float animationTime, float dt)
{
  constexpr auto radius = 0.03f;

  constexpr auto wallOffsetSpeed = 5.f;
  constexpr auto wallOffsetMagnitude = 0.5f;

  // Set uniform
  SimulationParams params;
  params.dt = dt;
  params.num_particles = particleCount;
  params.radius = radius;
  params.alpha = 1e-3f;
  params.wall_offset = static_cast<float>(wallOffsetMagnitude * std::sin(animationTime * wallOffsetSpeed));

  std::memcpy(uniformBuffer_.map, &params, sizeof(SimulationParams));

  // Descriptor set update
  // Binding 0: input
  // Binding 1: output
  // Binding 2: uniform grid and hash table
  // Binding 3: collision pairs
  // Binding 4: collision pairs linked list
  // Binding 5: solver
  // Binding 6: indirect dispatch
  // Binding 7: uniform params

  std::vector<vk::DescriptorBufferInfo> bufferInfos(8);
  bufferInfos[0]
    .setBuffer(srcBuffer_.buffer)
    .setOffset(srcBuffer_.offset)
    .setRange(srcBuffer_.size);

  bufferInfos[1]
    .setBuffer(dstBuffer_.buffer)
    .setOffset(dstBuffer_.offset)
    .setRange(dstBuffer_.size);

  bufferInfos[2]
    .setBuffer(internalBuffer_.buffer)
    .setOffset(internalBuffer_.offset + gridBufferRange_.offset)
    .setRange(gridBufferRange_.size);

  bufferInfos[3]
    .setBuffer(internalBuffer_.buffer)
    .setOffset(internalBuffer_.offset + collisionPairsBufferRange_.offset)
    .setRange(collisionPairsBufferRange_.size);

  bufferInfos[4]
    .setBuffer(internalBuffer_.buffer)
    .setOffset(internalBuffer_.offset + collisionChainBufferRange_.offset)
    .setRange(collisionChainBufferRange_.size);

  bufferInfos[5]
    .setBuffer(internalBuffer_.buffer)
    .setOffset(internalBuffer_.offset + solveBufferRange_.offset)
    .setRange(solveBufferRange_.size);

  bufferInfos[6]
    .setBuffer(internalBuffer_.buffer)
    .setOffset(internalBuffer_.offset + dispatchIndirectBufferRange_.offset)
    .setRange(dispatchIndirectBufferRange_.size);

  bufferInfos[7]
    .setBuffer(uniformBuffer_.buffer)
    .setOffset(uniformBuffer_.offset)
    .setRange(uniformBuffer_.size);

  std::vector<vk::WriteDescriptorSet> descriptorWrites(8);
  for (int i = 0; i < 7; i++)
  {
    descriptorWrites[i]
      .setDstSet(descriptorSets_[cmdIndex])
      .setDstBinding(i)
      .setDstArrayElement(0)
      .setDescriptorType(vk::DescriptorType::eStorageBuffer)
      .setBufferInfo(bufferInfos[i]);
  }

  descriptorWrites[7]
    .setDstSet(descriptorSets_[cmdIndex])
    .setDstBinding(7)
    .setDstArrayElement(0)
    .setDescriptorType(vk::DescriptorType::eUniformBuffer)
    .setBufferInfo(bufferInfos[7]);

  device_.updateDescriptorSets(descriptorWrites, {});

  // Prepare compute shaders
  commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eCompute, pipelineLayout_, 0u,
    descriptorSets_[cmdIndex], {});

  // Forward
  commandBuffer.bindPipeline(vk::PipelineBindPoint::eCompute, forwardPipeline_);
  commandBuffer.dispatch((particleCount + 255) / 256, 1, 1);

  vk::BufferMemoryBarrier particleBufferMemoryBarrier;
  particleBufferMemoryBarrier
    .setSrcAccessMask(vk::AccessFlagBits::eShaderWrite)
    .setDstAccessMask(vk::AccessFlagBits::eShaderRead)
    .setSrcQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
    .setDstQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
    .setBuffer(dstBuffer_.buffer)
    .setOffset(dstBuffer_.offset)
    .setSize(dstBuffer_.size);

  commandBuffer.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eComputeShader, {},
    {}, particleBufferMemoryBarrier, {});

  // Initialize uniform grid
  commandBuffer.bindPipeline(vk::PipelineBindPoint::eCompute, initializeUniformGridPipeline_);
  commandBuffer.dispatch((hashBucketCount_ + 255) / 256, 1, 1);

  vk::BufferMemoryBarrier gridBufferMemoryBarrier;
  gridBufferMemoryBarrier
    .setSrcAccessMask(vk::AccessFlagBits::eShaderWrite)
    .setDstAccessMask(vk::AccessFlagBits::eShaderRead)
    .setSrcQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
    .setDstQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
    .setBuffer(internalBuffer_.buffer)
    .setOffset(internalBuffer_.offset + gridBufferRange_.offset)
    .setSize(gridBufferRange_.size);

  commandBuffer.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eComputeShader, {},
    {}, gridBufferMemoryBarrier, {});

  // Add to uniform grid
  commandBuffer.bindPipeline(vk::PipelineBindPoint::eCompute, addUniformGridPipeline_);
  commandBuffer.dispatch((particleCount + 255) / 256, 1, 1);

  commandBuffer.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eComputeShader, {},
    {}, gridBufferMemoryBarrier, {});

  // Initialize collision detection
  commandBuffer.bindPipeline(vk::PipelineBindPoint::eCompute, initializeCollisionDetectionPipeline_);
  commandBuffer.dispatch((particleCount + 255) / 256, 1, 1);

  vk::BufferMemoryBarrier collisionPairsBufferMemoryBarrier;
  collisionPairsBufferMemoryBarrier
    .setSrcAccessMask(vk::AccessFlagBits::eShaderWrite)
    .setDstAccessMask(vk::AccessFlagBits::eShaderRead)
    .setSrcQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
    .setDstQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
    .setBuffer(internalBuffer_.buffer)
    .setOffset(internalBuffer_.offset + collisionPairsBufferRange_.offset)
    .setSize(collisionPairsBufferRange_.size);

  vk::BufferMemoryBarrier collisionChainBufferMemoryBarrier;
  collisionChainBufferMemoryBarrier
    .setSrcAccessMask(vk::AccessFlagBits::eShaderWrite)
    .setDstAccessMask(vk::AccessFlagBits::eShaderRead)
    .setSrcQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
    .setDstQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
    .setBuffer(internalBuffer_.buffer)
    .setOffset(internalBuffer_.offset + collisionChainBufferRange_.offset)
    .setSize(collisionChainBufferRange_.size);

  commandBuffer.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eComputeShader, {},
    {}, { collisionPairsBufferMemoryBarrier, collisionChainBufferMemoryBarrier }, {});

  // Collision detection
  commandBuffer.bindPipeline(vk::PipelineBindPoint::eCompute, collisionDetectionPipeline_);
  commandBuffer.dispatch((particleCount + 255) / 256, 1, 1);

  commandBuffer.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eComputeShader, {},
    {}, { collisionPairsBufferMemoryBarrier, collisionChainBufferMemoryBarrier }, {});

  // In collision detection, particle color is written for debug purpose
  commandBuffer.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eComputeShader, {},
    {}, particleBufferMemoryBarrier, {});

  // Initialize dispatch
  commandBuffer.bindPipeline(vk::PipelineBindPoint::eCompute, initializeDispatchPipeline_);
  commandBuffer.dispatch(1, 1, 1);

  vk::MemoryBarrier dispatch_indirect_barrier;
  dispatch_indirect_barrier
    .setSrcAccessMask(vk::AccessFlagBits::eShaderWrite)
    .setDstAccessMask(vk::AccessFlagBits::eIndirectCommandRead);

  // Why draw indirect stage, not top of pipe or compute?
  commandBuffer.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eDrawIndirect, {},
    dispatch_indirect_barrier, {}, {});

  // Initialize solver
  commandBuffer.bindPipeline(vk::PipelineBindPoint::eCompute, initializeSolverPipeline_);
  commandBuffer.dispatchIndirect(internalBuffer_.buffer, internalBuffer_.offset + dispatchIndirectBufferRange_.offset);

  vk::BufferMemoryBarrier solverBufferMemoryBarrier;
  solverBufferMemoryBarrier
    .setSrcAccessMask(vk::AccessFlagBits::eShaderWrite)
    .setDstAccessMask(vk::AccessFlagBits::eShaderRead)
    .setSrcQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
    .setDstQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
    .setBuffer(internalBuffer_.buffer)
    .setOffset(internalBuffer_.offset + solveBufferRange_.offset)
    .setSize(solveBufferRange_.size);

  commandBuffer.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eComputeShader, {},
    {}, solverBufferMemoryBarrier, {});

  // Solve
  constexpr int solver_iterations = 1;
  for (int i = 0; i < solver_iterations; i++)
  {
    // Solve delta lambda
    commandBuffer.bindPipeline(vk::PipelineBindPoint::eCompute, solveDeltaLambdaPipeline_);
    commandBuffer.dispatchIndirect(internalBuffer_.buffer, internalBuffer_.offset + dispatchIndirectBufferRange_.offset + sizeof(uint32_t) * 4);

    commandBuffer.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eComputeShader, {},
      {}, solverBufferMemoryBarrier, {});

    // Solve delta x
    commandBuffer.bindPipeline(vk::PipelineBindPoint::eCompute, solveDeltaXPipeline_);
    commandBuffer.dispatch((particleCount + 255) / 256, 1, 1);

    commandBuffer.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eComputeShader, {},
      {}, solverBufferMemoryBarrier, {});

    // Solve x and lambda
    commandBuffer.bindPipeline(vk::PipelineBindPoint::eCompute, solveXLambdaPipeline_);
    commandBuffer.dispatchIndirect(internalBuffer_.buffer, internalBuffer_.offset + dispatchIndirectBufferRange_.offset);

    commandBuffer.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eComputeShader, {},
      {}, solverBufferMemoryBarrier, {});
  }

  // Velocity update
  commandBuffer.bindPipeline(vk::PipelineBindPoint::eCompute, velocityUpdatePipeline_);
  commandBuffer.dispatch((particleCount + 255) / 256, 1, 1);
}

void ParticleSimulator::destroy()
{
  device_.destroyPipeline(forwardPipeline_);
  device_.destroyPipeline(initializeUniformGridPipeline_);
  device_.destroyPipeline(addUniformGridPipeline_);
  device_.destroyPipeline(initializeCollisionDetectionPipeline_);
  device_.destroyPipeline(collisionDetectionPipeline_);
  device_.destroyPipeline(initializeDispatchPipeline_);
  device_.destroyPipeline(initializeSolverPipeline_);
  device_.destroyPipeline(solveDeltaLambdaPipeline_);
  device_.destroyPipeline(solveDeltaXPipeline_);
  device_.destroyPipeline(solveXLambdaPipeline_);
  device_.destroyPipeline(velocityUpdatePipeline_);

  device_.destroyPipelineLayout(pipelineLayout_);
  device_.destroyDescriptorSetLayout(descriptorSetLayout_);

  descriptorSets_.clear();
}

ParticleSimulator createParticleSimulator(const ParticleSimulatorCreateInfo& createInfo)
{
  ParticleSimulator simulator;
  simulator.device_ = createInfo.device;
  simulator.descriptorPool_ = createInfo.descriptorPool;
  simulator.particleCount_ = createInfo.particleCount;

  auto device = createInfo.device;
  auto physical_device = createInfo.physicalDevice;

  // Create descriptor set layout
  std::vector<vk::DescriptorSetLayoutBinding> bindings(8);

  for (int i = 0; i < 7; i++)
  {
    bindings[i]
      .setBinding(i)
      .setStageFlags(vk::ShaderStageFlagBits::eCompute)
      .setDescriptorType(vk::DescriptorType::eStorageBuffer)
      .setDescriptorCount(1);
  }

  bindings[7]
    .setBinding(7)
    .setStageFlags(vk::ShaderStageFlagBits::eCompute)
    .setDescriptorType(vk::DescriptorType::eUniformBuffer)
    .setDescriptorCount(1);

  vk::DescriptorSetLayoutCreateInfo descriptorSetLayoutCreateInfo;
  descriptorSetLayoutCreateInfo.setBindings(bindings);
  simulator.descriptorSetLayout_ = device.createDescriptorSetLayout(descriptorSetLayoutCreateInfo);

  // Create descriptor sets
  std::vector<vk::DescriptorSetLayout> layouts(createInfo.commandCount, simulator.descriptorSetLayout_);
  vk::DescriptorSetAllocateInfo descriptorSetAllocateInfo;
  descriptorSetAllocateInfo
    .setDescriptorPool(simulator.descriptorPool_)
    .setSetLayouts(layouts);
  simulator.descriptorSets_ = device.allocateDescriptorSets(descriptorSetAllocateInfo);

  // Create pipeline layout
  vk::PipelineLayoutCreateInfo pipelineLayoutCreateInfo;
  pipelineLayoutCreateInfo.setSetLayouts(simulator.descriptorSetLayout_);
  simulator.pipelineLayout_ = device.createPipelineLayout(pipelineLayoutCreateInfo);

  // Create pipelines
  auto pipelineCache = device.createPipelineCache({});
  const auto createComputePipeline = [&device, &pipelineCache, pipelineLayout = simulator.pipelineLayout_](const std::string& filepath)
  {
    std::ifstream file(filepath, std::ios::ate | std::ios::binary);
    if (!file.is_open())
      throw std::runtime_error("Failed to open file: " + filepath);

    size_t fileSize = (size_t)file.tellg();
    std::vector<char> buffer(fileSize);
    file.seekg(0);
    file.read(buffer.data(), fileSize);
    file.close();

    std::vector<uint32_t> code;
    auto* int_ptr = reinterpret_cast<uint32_t*>(buffer.data());
    for (int i = 0; i < fileSize / 4; i++)
      code.push_back(int_ptr[i]);

    // Create shader module
    vk::ShaderModuleCreateInfo shaderModuleCreateInfo;
    shaderModuleCreateInfo
      .setCode(code);
    auto module = device.createShaderModule(shaderModuleCreateInfo);

    vk::PipelineShaderStageCreateInfo shaderStage;
    shaderStage
      .setStage(vk::ShaderStageFlagBits::eCompute)
      .setModule(module)
      .setPName("main");

    vk::ComputePipelineCreateInfo computePipelineCreateInfo;
    computePipelineCreateInfo
      .setStage(shaderStage)
      .setLayout(pipelineLayout);
    auto pipeline = device.createComputePipeline(pipelineCache, computePipelineCreateInfo);

    // Destroy shader module after creating pipeline
    device.destroyShaderModule(module);

    return pipeline.value;
  };

  const std::string baseDir = "C:\\workspace\\superlucent\\src\\vkpbd\\shader\\particle";
  simulator.forwardPipeline_ = createComputePipeline(baseDir + "\\forward.comp.spv");
  simulator.initializeUniformGridPipeline_ = createComputePipeline(baseDir + "\\initialize_uniform_grid.comp.spv");
  simulator.addUniformGridPipeline_ = createComputePipeline(baseDir + "\\add_uniform_grid.comp.spv");
  simulator.initializeCollisionDetectionPipeline_ = createComputePipeline(baseDir + "\\initialize_collision_detection.comp.spv");
  simulator.collisionDetectionPipeline_ = createComputePipeline(baseDir + "\\collision_detection.comp.spv");
  simulator.initializeDispatchPipeline_ = createComputePipeline(baseDir + "\\initialize_dispatch.comp.spv");
  simulator.initializeSolverPipeline_ = createComputePipeline(baseDir + "\\initialize_solver.comp.spv");
  simulator.solveDeltaLambdaPipeline_ = createComputePipeline(baseDir + "\\solve_delta_lambda.comp.spv");
  simulator.solveDeltaXPipeline_ = createComputePipeline(baseDir + "\\solve_delta_x.comp.spv");
  simulator.solveXLambdaPipeline_ = createComputePipeline(baseDir + "\\solve_x_lambda.comp.spv");
  simulator.velocityUpdatePipeline_ = createComputePipeline(baseDir + "\\velocity_update.comp.spv");

  device.destroyPipelineCache(pipelineCache);

  // Requirements
  const auto uboAlignment = physical_device.getProperties().limits.minUniformBufferOffsetAlignment;
  const auto ssboAlignment = physical_device.getProperties().limits.minStorageBufferOffsetAlignment;

  const auto align = [](vk::DeviceSize offset, vk::DeviceSize alignment)
  {
    return (offset + alignment - 1) & ~(alignment - 1);
  };

  // Sub buffer ranges
  const auto collisionCount =
    simulator.particleCount_ + 5 // walls
    + simulator.particleCount_ * 6; // max 12 collisions for each sphere, 6 pairs in average

  const auto solverBufferSize =
    (collisionCount // lambda
      + simulator.particleCount_ * 3) // x
    * 2 // delta
    * sizeof(float);

  const auto collisionChainBufferSize =
    (sizeof(int32_t) * 2) * simulator.particleCount_;

  const auto gridBufferSize =
    16 // 4-element header
    + ParticleSimulator::hashBucketCount_ * sizeof(int32_t) + sizeof(int32_t) // hash bucket plus pad
    + (sizeof(uint32_t) + sizeof(int32_t)) * (simulator.particleCount_ * 8); // object grid pairs

  simulator.gridBufferRange_.offset = 0;
  simulator.gridBufferRange_.size = gridBufferSize;

  simulator.collisionPairsBufferRange_.offset = align(simulator.gridBufferRange_.offset + simulator.gridBufferRange_.size, ssboAlignment);
  simulator.collisionPairsBufferRange_.size = sizeof(uint32_t) + collisionCount * (sizeof(int32_t) * 4 + sizeof(float) * 12);

  simulator.collisionChainBufferRange_.offset = align(simulator.collisionPairsBufferRange_.offset + simulator.collisionPairsBufferRange_.size, ssboAlignment);
  simulator.collisionChainBufferRange_.size = collisionChainBufferSize;

  simulator.solveBufferRange_.offset = align(simulator.collisionChainBufferRange_.offset + simulator.collisionChainBufferRange_.size, ssboAlignment);
  simulator.solveBufferRange_.size = solverBufferSize;

  simulator.dispatchIndirectBufferRange_.offset = align(simulator.solveBufferRange_.offset + simulator.solveBufferRange_.size, ssboAlignment);
  simulator.dispatchIndirectBufferRange_.size = sizeof(uint32_t) * 8;

  // Requirements
  simulator.particleBufferRequiredSize_ = sizeof(Particle) * simulator.particleCount_;
  simulator.internalBufferRequiredSize_ = simulator.dispatchIndirectBufferRange_.offset + simulator.dispatchIndirectBufferRange_.size;
  simulator.uniformBufferRequiredSize_ = sizeof(SimulationParams);

  return simulator;
}
}
