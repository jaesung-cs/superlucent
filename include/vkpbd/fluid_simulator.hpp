#ifndef VKPBD_FLUID_SIMULATOR_HPP_
#define VKPBD_FLUID_SIMULATOR_HPP_

#include <vector>

#include <vulkan/vulkan.hpp>

#include <vkpbd/buffer_requirements.h>

namespace vkpbd
{
class FluidSimulatorCreateInfo;
class FluidSimulator;

FluidSimulator createFluidSimulator(const FluidSimulatorCreateInfo& createInfo);

class FluidSimulator
{
  friend FluidSimulator createFluidSimulator(const FluidSimulatorCreateInfo& createInfo);

private:
  struct SubBufferRange
  {
    vk::DeviceSize offset = 0;
    vk::DeviceSize size = 0;
  };

  struct BufferRange
  {
    vk::Buffer buffer;
    vk::DeviceSize offset = 0;
    vk::DeviceSize size = 0;
  };

  struct UniformBufferRange
  {
    vk::Buffer buffer;
    vk::DeviceSize offset = 0;
    vk::DeviceSize size = 0;
    uint8_t* map = nullptr;
  };

public:
  FluidSimulator() = default;

  ~FluidSimulator() = default;

  auto getParticleCount() const
  {
    return particleCount_;
  }

  BufferRequirements getParticleBufferRequirements();
  BufferRequirements getInternalBufferRequirements();
  BufferRequirements getUniformBufferRequirements();

  void cmdBindSrcParticleBuffer(vk::Buffer buffer, vk::DeviceSize offset);
  void cmdBindDstParticleBuffer(vk::Buffer buffer, vk::DeviceSize offset);
  void cmdBindInternalBuffer(vk::Buffer buffer, vk::DeviceSize offset);
  void cmdBindUniformBuffer(vk::Buffer buffer, vk::DeviceSize offset, uint8_t* map);

  void cmdStep(vk::CommandBuffer commandBuffer, int cmdIndex, float animationTime, float dt);
  void cmdStep(vk::CommandBuffer commandBuffer, int cmdIndex, uint32_t particleCount, float animationTime, float dt);

  void destroy();

private:
  vk::Device device_;

  static constexpr int hashBucketCount_ = 1000003;

  // Descriptor set
  // Binding 0: input
  // Binding 1: output
  // Binding 2: uniform grid and hash table
  // Binding 3: collision pairs
  // Binding 4: collision pairs linked list
  // Binding 5: solver
  // Binding 6: indirect dispatch
  // Binding 7: uniform params
  vk::DescriptorSetLayout descriptorSetLayout_;
  vk::DescriptorPool descriptorPool_;

  // Descriptor sets will be created on demand and reused
  std::vector<vk::DescriptorSet> descriptorSets_;

  // Pipelines
  vk::PipelineLayout pipelineLayout_;
  vk::Pipeline forwardPipeline_;
  vk::Pipeline initializeUniformGridPipeline_;
  vk::Pipeline addUniformGridPipeline_;
  vk::Pipeline initializeCollisionDetectionPipeline_;
  vk::Pipeline collisionDetectionPipeline_;
  vk::Pipeline initializeDispatchPipeline_;
  vk::Pipeline initializeSolverPipeline_;
  vk::Pipeline solveDeltaLambdaPipeline_;
  vk::Pipeline solveDeltaXPipeline_;
  vk::Pipeline solveXLambdaPipeline_;
  vk::Pipeline velocityUpdatePipeline_;

  // Internal buffer ranges
  SubBufferRange gridBufferRange_;
  SubBufferRange collisionPairsBufferRange_;
  SubBufferRange collisionChainBufferRange_;
  SubBufferRange solveBufferRange_;
  SubBufferRange dispatchIndirectBufferRange_;

  // Requirements
  vk::DeviceSize particleBufferRequiredSize_ = 0;
  vk::DeviceSize internalBufferRequiredSize_ = 0;
  vk::DeviceSize uniformBufferRequiredSize_ = 0;

  // Bound buffer ranges
  BufferRange srcBuffer_;
  BufferRange dstBuffer_;
  BufferRange internalBuffer_;
  UniformBufferRange uniformBuffer_;

  uint32_t particleCount_ = 0;
};

class FluidSimulatorCreateInfo
{
public:
  FluidSimulatorCreateInfo() = default;
  ~FluidSimulatorCreateInfo() = default;

public:
  vk::Device device;
  vk::PhysicalDevice physicalDevice;
  vk::DescriptorPool descriptorPool;
  uint32_t particleCount = 0;
  uint32_t commandCount = 0;
};
}

#endif // VKPBD_FLUID_SIMULATOR_HPP_
