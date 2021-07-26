#include <superlucent/engine/mesh_renderer.h>

#include <fstream>

namespace supl
{
namespace engine
{
namespace
{
vk::ShaderModule createShaderModule(vk::Device device, const std::string& filepath)
{
  std::ifstream file(filepath, std::ios::ate | std::ios::binary);
  if (!file.is_open())
    throw std::runtime_error("Failed to open file: " + filepath);

  size_t file_size = (size_t)file.tellg();
  std::vector<char> buffer(file_size);
  file.seekg(0);
  file.read(buffer.data(), file_size);
  file.close();

  std::vector<uint32_t> code;
  auto* int_ptr = reinterpret_cast<uint32_t*>(buffer.data());
  for (int i = 0; i < file_size / 4; i++)
    code.push_back(int_ptr[i]);

  vk::ShaderModuleCreateInfo shader_module_create_info;
  shader_module_create_info
    .setCode(code);
  return device.createShaderModule(shader_module_create_info);
}
}

MeshRenderer::MeshRenderer()
{
}

MeshRenderer::~MeshRenderer()
{
}

void MeshRenderer::destroy()
{
  device_.destroyPipelineLayout(pipelineLayout_);
  device_.destroyDescriptorSetLayout(descriptorSetLayout_);
  device_.destroyRenderPass(renderPass_);
}

void MeshRenderer::resize(uint32_t width, uint32_t height)
{
  width_ = width;
  height_ = height;

  // TODO: Recreate swapchains and framebuffers
}

void MeshRenderer::updateLights(const LightUbo& lights, int imageIndex)
{
  // TODO
}

void MeshRenderer::updateCamera(const CameraUbo& camera, int imageIndex)
{
  // TODO
}

void MeshRenderer::begin(vk::CommandBuffer& commandBuffer, int imageIndex)
{
  vk::Rect2D renderArea{ {0u, 0u}, {width_, height_} };

  // TODO: Begin render pass
  commandBuffer.setViewport(0, vk::Viewport(0.f, 0.f, width_, height_, 0.f, 1.f));
  commandBuffer.setScissor(0, renderArea);

  // TODO: clear flag
  std::vector<vk::ClearValue> clearValues{
    vk::ClearColorValue{ std::array<float, 4>{ 0.f, 0.f, 0.f, 1.f } },
    vk::ClearDepthStencilValue{ 1.f, 0u }
  };
  vk::RenderPassBeginInfo renderPassBeginInfo;
  renderPassBeginInfo
    .setRenderPass(renderPass_)
    // TODO: setFramebuffer()
    .setRenderArea(renderArea)
    .setClearValues(clearValues);
  commandBuffer.beginRenderPass(renderPassBeginInfo, vk::SubpassContents::eInline);
}

void MeshRenderer::end(vk::CommandBuffer& commandBuffer)
{
  commandBuffer.endRenderPass();
}

MeshRenderer createMeshRenderer(const MeshRendererCreateInfo& createInfo)
{
  const auto& device = createInfo.device;
  const auto& descriptorPool = createInfo.descriptorPool;
  const auto imageCount = createInfo.imageCount;
  const auto format = createInfo.format;
  const auto width = createInfo.width;
  const auto height = createInfo.height;
  const auto finalLayout = createInfo.finalLayout;

  MeshRenderer renderer;
  renderer.width_ = width;
  renderer.height_ = height;
  renderer.device_ = device;

  // Descriptor set
  std::vector<vk::DescriptorSetLayoutBinding> bindings;
  bindings[0]
    .setBinding(0)
    .setDescriptorCount(1)
    .setDescriptorType(vk::DescriptorType::eUniformBuffer)
    .setStageFlags(vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment);

  bindings[1]
    .setBinding(1)
    .setDescriptorCount(1)
    .setDescriptorType(vk::DescriptorType::eUniformBuffer)
    .setStageFlags(vk::ShaderStageFlagBits::eFragment);

  vk::DescriptorSetLayoutCreateInfo descriptorSetLayoutCreateInfo;
  descriptorSetLayoutCreateInfo
    .setBindings(bindings);
  renderer.descriptorSetLayout_ = device.createDescriptorSetLayout(descriptorSetLayoutCreateInfo);

  std::vector<vk::DescriptorSetLayout> layouts(imageCount, renderer.descriptorSetLayout_);
  vk::DescriptorSetAllocateInfo descriptorSetAllocateInfo;
  descriptorSetAllocateInfo
    .setDescriptorPool(descriptorPool)
    .setSetLayouts(layouts);
  renderer.descriptorSets_ = device.allocateDescriptorSets(descriptorSetAllocateInfo);

  // Render pass
  std::vector<vk::AttachmentDescription> attachments(3);
  attachments[0]
    .setFormat(format)
    .setSamples(vk::SampleCountFlagBits::e4)
    .setLoadOp(vk::AttachmentLoadOp::eClear)
    .setStoreOp(vk::AttachmentStoreOp::eStore)
    .setStencilLoadOp(vk::AttachmentLoadOp::eDontCare)
    .setStencilStoreOp(vk::AttachmentStoreOp::eDontCare)
    .setInitialLayout(vk::ImageLayout::eUndefined)
    .setFinalLayout(vk::ImageLayout::eColorAttachmentOptimal);

  attachments[1]
    .setFormat(vk::Format::eD24UnormS8Uint)
    .setSamples(vk::SampleCountFlagBits::e4)
    .setLoadOp(vk::AttachmentLoadOp::eClear)
    .setStoreOp(vk::AttachmentStoreOp::eDontCare)
    .setStencilLoadOp(vk::AttachmentLoadOp::eDontCare)
    .setStencilStoreOp(vk::AttachmentStoreOp::eDontCare)
    .setInitialLayout(vk::ImageLayout::eUndefined)
    .setFinalLayout(vk::ImageLayout::eDepthStencilAttachmentOptimal);

  attachments[2]
    .setFormat(format)
    .setSamples(vk::SampleCountFlagBits::e1)
    .setLoadOp(vk::AttachmentLoadOp::eDontCare)
    .setStoreOp(vk::AttachmentStoreOp::eStore)
    .setStencilLoadOp(vk::AttachmentLoadOp::eDontCare)
    .setStencilStoreOp(vk::AttachmentStoreOp::eDontCare)
    .setInitialLayout(vk::ImageLayout::eUndefined)
    .setFinalLayout(finalLayout);

  std::vector<vk::AttachmentReference> attachmentReferences(3);
  attachmentReferences[0]
    .setAttachment(0)
    .setLayout(vk::ImageLayout::eColorAttachmentOptimal);

  attachmentReferences[1]
    .setAttachment(1)
    .setLayout(vk::ImageLayout::eDepthStencilAttachmentOptimal);

  attachmentReferences[2]
    .setAttachment(2)
    .setLayout(vk::ImageLayout::eColorAttachmentOptimal);

  std::vector<vk::SubpassDescription> subpasses(1);
  subpasses[0]
    .setPipelineBindPoint(vk::PipelineBindPoint::eGraphics)
    .setColorAttachments(attachmentReferences[0])
    .setPDepthStencilAttachment(&attachmentReferences[1])
    .setResolveAttachments(attachmentReferences[2]);

  std::vector<vk::SubpassDependency> dependencies(1);
  dependencies[0]
    .setSrcSubpass(VK_SUBPASS_EXTERNAL)
    .setDstSubpass(0)
    .setSrcStageMask(vk::PipelineStageFlagBits::eColorAttachmentOutput | vk::PipelineStageFlagBits::eEarlyFragmentTests)
    .setDstStageMask(vk::PipelineStageFlagBits::eColorAttachmentOutput | vk::PipelineStageFlagBits::eEarlyFragmentTests)
    .setSrcAccessMask({})
    .setDstAccessMask(vk::AccessFlagBits::eColorAttachmentWrite | vk::AccessFlagBits::eDepthStencilAttachmentWrite);

  vk::RenderPassCreateInfo renderPassCreateInfo;
  renderPassCreateInfo
    .setAttachments(attachments)
    .setSubpasses(subpasses)
    .setDependencies(dependencies);

  // Pipeline layout
  // Push constants: model matrix
  vk::PushConstantRange pushConstantRange;
  pushConstantRange
    .setOffset(0)
    .setSize(sizeof(glm::mat4))
    .setStageFlags(vk::ShaderStageFlagBits::eVertex);

  vk::PipelineLayoutCreateInfo pipelineLayoutCreateInfo;
  pipelineLayoutCreateInfo
    .setSetLayouts(renderer.descriptorSetLayout_)
    .setPushConstantRanges(pushConstantRange);
  renderer.pipelineLayout_ = device.createPipelineLayout(pipelineLayoutCreateInfo);

  // TODO: Create pipeline
  const std::string baseDir = "C:\\workspace\\superlucent\\src\\superlucent\\shader\\rendering";
  vk::ShaderModule meshVertexShader = createShaderModule(device, baseDir + "\\mesh.vert.spv");
  vk::ShaderModule meshFragmentShader = createShaderModule(device, baseDir + "\\mesh.frag.spv");
  std::vector<vk::PipelineShaderStageCreateInfo> stages(2);
  stages[0]
    .setModule(meshVertexShader)
    .setStage(vk::ShaderStageFlagBits::eVertex)
    .setPName("main");

  stages[1]
    .setModule(meshFragmentShader)
    .setStage(vk::ShaderStageFlagBits::eFragment)
    .setPName("main");

  vk::PipelineVertexInputStateCreateInfo vertexInput; // TODO
  vk::PipelineInputAssemblyStateCreateInfo inputAssembly; // TODO
  vk::PipelineViewportStateCreateInfo viewport; // TODO
  vk::PipelineRasterizationStateCreateInfo rasterization; // TODO
  vk::PipelineMultisampleStateCreateInfo multisample; // TODO
  vk::PipelineDepthStencilStateCreateInfo depthStencil; // TODO
  vk::PipelineColorBlendStateCreateInfo colorBlend; // TODO
  vk::PipelineDynamicStateCreateInfo dynamicState; // TODO

  vk::GraphicsPipelineCreateInfo graphicsPipelineCreateInfo;
  graphicsPipelineCreateInfo
    .setLayout(renderer.pipelineLayout_)
    .setRenderPass(renderer.renderPass_)
    .setStages(stages)
    .setPVertexInputState(&vertexInput)
    .setPInputAssemblyState(&inputAssembly)
    .setPViewportState(&viewport)
    .setPRasterizationState(&rasterization)
    .setPMultisampleState(&multisample)
    .setPDepthStencilState(&depthStencil)
    .setPColorBlendState(&colorBlend)
    .setPDynamicState(&dynamicState)
    .setSubpass(0);

  device.destroyShaderModule(meshVertexShader);
  device.destroyShaderModule(meshFragmentShader);

  return renderer;
}
}
}
