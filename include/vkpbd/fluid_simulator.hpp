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
  BufferRequirements getBoundaryBufferRequirements();

  void cmdBindSrcParticleBuffer(vk::Buffer buffer, vk::DeviceSize offset);
  void cmdBindDstParticleBuffer(vk::Buffer buffer, vk::DeviceSize offset);
  void cmdBindBoundaryBuffer(vk::Buffer buffer, vk::DeviceSize offset);
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
  // Binding 2: boundary
  // Binding 3: boundary volume
  // Binding 4: uniform grid and hash table
  // Binding 5: neighbors
  // Binding 6: solver
  // Binding 7: uniform params
  vk::DescriptorSetLayout descriptorSetLayout_;
  vk::DescriptorPool descriptorPool_;

  // Descriptor sets will be created on demand and reused
  std::vector<vk::DescriptorSet> descriptorSets_;

  // Pipelines
  vk::PipelineLayout pipelineLayout_;
  vk::Pipeline forwardPipeline_;
  vk::Pipeline initializeUniformGridPipeline_;
  vk::Pipeline initializeBoudnaryPipeline_;
  vk::Pipeline addUniformGridPipeline_;
  vk::Pipeline initializeNeighborSearchPipeline_;
  vk::Pipeline neighborSearchPipeline_;
  vk::Pipeline computeBoundaryPipeline_;
  vk::Pipeline computeDensityPipeline_;
  vk::Pipeline solveDensityPipeline_;
  vk::Pipeline updatePositionPipeline_;
  vk::Pipeline solveViscosityPipeline_;
  vk::Pipeline updateViscosityPipeline_;
  vk::Pipeline velocityUpdatePipeline_;

  // Internal buffer ranges
  SubBufferRange gridBufferRange_;
  SubBufferRange neighborsBufferRange_;
  SubBufferRange boundaryBufferRange_;
  SubBufferRange solverBufferRange_;

  // Requirements
  vk::DeviceSize particleBufferRequiredSize_ = 0;
  vk::DeviceSize internalBufferRequiredSize_ = 0;
  vk::DeviceSize uniformBufferRequiredSize_ = 0;

  // Bound buffer ranges
  BufferRange srcBuffer_;
  BufferRange dstBuffer_;
  BufferRange boundaryBuffer_;
  BufferRange internalBuffer_;
  UniformBufferRange uniformBuffer_;

  uint32_t particleCount_ = 0;
  uint32_t boundaryCount_ = 0;
  uint32_t maxNeighborCount_ = 0;

  // Simulation constants
  float restDensity_ = 0.f;
  float viscosity_ = 0.f;
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
  uint32_t boundaryCount = 0;
  uint32_t commandCount = 0;
  uint32_t maxNeighborCount = 30u;
  float restDensity = 1000.f;
  float viscosity = 0.02f;
};
}

#endif // VKPBD_FLUID_SIMULATOR_HPP_
