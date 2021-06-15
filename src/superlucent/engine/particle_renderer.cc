#include <superlucent/engine/particle_renderer.h>

#include <glm/gtc/matrix_transform.hpp>

#include <superlucent/engine/engine.h>
#include <superlucent/engine/uniform_buffer.h>
#include <superlucent/engine/data/particle.h>
#include <superlucent/scene/camera.h>
#include <superlucent/scene/light.h>

namespace supl
{
namespace engine
{
ParticleRenderer::ParticleRenderer(Engine* engine, uint32_t width, uint32_t height)
  : engine_(engine)
  , width_(width)
  , height_(height)
{
  CreateSampler();
  CreateFramebuffer();
  CreateGraphicsPipelines();
  PrepareResources();
}

ParticleRenderer::~ParticleRenderer()
{
  DestroyResources();
  DestroyGraphicsPipelines();
  DestroyFramebuffer();
  DestroySampler();
}

void ParticleRenderer::UpdateLights(const LightUbo& lights, int image_index)
{
  const auto uniform_buffer = engine_->UniformBuffer();

  light_ubos_[image_index] = lights;
}

void ParticleRenderer::UpdateCamera(const CameraUbo& camera, int image_index)
{
  const auto uniform_buffer = engine_->UniformBuffer();

  camera_ubos_[image_index] = camera;
}

void ParticleRenderer::RecordRenderCommands(vk::CommandBuffer command_buffer, vk::Buffer particle_buffer, uint32_t num_particles, int image_index)
{
  // Begin render pass 
  command_buffer.setViewport(0, vk::Viewport{ 0.f, 0.f, static_cast<float>(width_), static_cast<float>(height_), 0.f, 1.f });

  command_buffer.setScissor(0, vk::Rect2D{ {0u, 0u}, {width_, height_} });

  std::vector<vk::ClearValue> clear_values{
    vk::ClearColorValue{ std::array<float, 4>{0.8f, 0.8f, 0.8f, 1.f} },
    vk::ClearDepthStencilValue{ 1.f, 0u }
  };
  vk::RenderPassBeginInfo render_pass_begin_info;
  render_pass_begin_info
    .setRenderPass(render_pass_)
    .setFramebuffer(swapchain_framebuffers_[image_index])
    .setRenderArea(vk::Rect2D{ {0u, 0u}, {width_, height_} })
    .setClearValues(clear_values);
  command_buffer.beginRenderPass(render_pass_begin_info, vk::SubpassContents::eInline);

  // Bind a shared descriptor set
  command_buffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipeline_layout_, 0u,
    descriptor_sets_[image_index], {});

  // Draw cells
  command_buffer.bindPipeline(vk::PipelineBindPoint::eGraphics, cell_sphere_pipeline_);

  command_buffer.bindVertexBuffers(0u,
    { cells_buffer_.buffer, particle_buffer },
    { 0ull, 0ull });

  command_buffer.bindIndexBuffer(cells_buffer_.buffer, cells_buffer_.index_offset, vk::IndexType::eUint32);

  command_buffer.drawIndexed(cells_buffer_.num_indices, num_particles, 0u, 0u, 0u);

  // Draw floor model
  command_buffer.bindPipeline(vk::PipelineBindPoint::eGraphics, floor_pipeline_);

  command_buffer.bindVertexBuffers(0u, { floor_buffer_.buffer }, { 0ull });

  command_buffer.bindIndexBuffer(floor_buffer_.buffer, floor_buffer_.index_offset, vk::IndexType::eUint32);

  command_buffer.drawIndexed(floor_buffer_.num_indices, 1u, 0u, 0u, 0u);

  command_buffer.endRenderPass();
}

void ParticleRenderer::CreateSampler()
{
  const auto device = engine_->Device();

  vk::SamplerCreateInfo sampler_create_info;
  sampler_create_info
    .setMagFilter(vk::Filter::eLinear)
    .setMinFilter(vk::Filter::eLinear)
    .setMipmapMode(vk::SamplerMipmapMode::eLinear)
    .setAddressModeU(vk::SamplerAddressMode::eRepeat)
    .setAddressModeV(vk::SamplerAddressMode::eRepeat)
    .setAddressModeW(vk::SamplerAddressMode::eRepeat)
    .setMipLodBias(0.f)
    .setAnisotropyEnable(true)
    .setMaxAnisotropy(16.f)
    .setCompareEnable(false)
    .setMinLod(0.f)
    .setMaxLod(mipmap_level_)
    .setBorderColor(vk::BorderColor::eFloatTransparentBlack)
    .setUnnormalizedCoordinates(false);
  sampler_ = device.createSampler(sampler_create_info);
}

void ParticleRenderer::DestroySampler()
{
  const auto device = engine_->Device();

  device.destroySampler(sampler_);
}

void ParticleRenderer::CreateFramebuffer()
{
  const auto device = engine_->Device();
  const auto swapchain_image_format = engine_->SwapchainImageFormat();

  // Attachment descriptions
  vk::AttachmentDescription color_attachment_description;
  color_attachment_description
    .setFormat(swapchain_image_format)
    .setSamples(vk::SampleCountFlagBits::e4)
    .setLoadOp(vk::AttachmentLoadOp::eClear)
    .setStoreOp(vk::AttachmentStoreOp::eStore)
    .setStencilLoadOp(vk::AttachmentLoadOp::eDontCare)
    .setStencilStoreOp(vk::AttachmentStoreOp::eDontCare)
    .setInitialLayout(vk::ImageLayout::eUndefined)
    .setFinalLayout(vk::ImageLayout::eColorAttachmentOptimal);

  vk::AttachmentDescription depth_attachment_description;
  depth_attachment_description
    .setFormat(vk::Format::eD24UnormS8Uint)
    .setSamples(vk::SampleCountFlagBits::e4)
    .setLoadOp(vk::AttachmentLoadOp::eClear)
    .setStoreOp(vk::AttachmentStoreOp::eDontCare)
    .setStencilLoadOp(vk::AttachmentLoadOp::eDontCare)
    .setStencilStoreOp(vk::AttachmentStoreOp::eDontCare)
    .setInitialLayout(vk::ImageLayout::eUndefined)
    .setFinalLayout(vk::ImageLayout::eDepthStencilAttachmentOptimal);

  vk::AttachmentDescription resolve_attachment_description;
  resolve_attachment_description
    .setFormat(swapchain_image_format)
    .setSamples(vk::SampleCountFlagBits::e1)
    .setLoadOp(vk::AttachmentLoadOp::eDontCare)
    .setStoreOp(vk::AttachmentStoreOp::eStore)
    .setStencilLoadOp(vk::AttachmentLoadOp::eDontCare)
    .setStencilStoreOp(vk::AttachmentStoreOp::eDontCare)
    .setInitialLayout(vk::ImageLayout::eUndefined)
    .setFinalLayout(vk::ImageLayout::ePresentSrcKHR);

  std::vector<vk::AttachmentDescription> attachment_descriptions{
    color_attachment_description,
    depth_attachment_description,
    resolve_attachment_description,
  };

  // Attachment references
  vk::AttachmentReference color_attachment_reference;
  color_attachment_reference
    .setAttachment(0)
    .setLayout(vk::ImageLayout::eColorAttachmentOptimal);

  vk::AttachmentReference depth_attachment_reference;
  depth_attachment_reference
    .setAttachment(1)
    .setLayout(vk::ImageLayout::eDepthStencilAttachmentOptimal);

  vk::AttachmentReference resolve_attachment_reference;
  resolve_attachment_reference
    .setAttachment(2)
    .setLayout(vk::ImageLayout::eColorAttachmentOptimal);

  // Subpasses
  vk::SubpassDescription subpass;
  subpass
    .setPipelineBindPoint(vk::PipelineBindPoint::eGraphics)
    .setColorAttachments(color_attachment_reference)
    .setResolveAttachments(resolve_attachment_reference)
    .setPDepthStencilAttachment(&depth_attachment_reference);

  // Dependencies
  vk::SubpassDependency dependency;
  dependency
    .setSrcSubpass(VK_SUBPASS_EXTERNAL)
    .setDstSubpass(0)
    .setSrcStageMask(vk::PipelineStageFlagBits::eColorAttachmentOutput | vk::PipelineStageFlagBits::eEarlyFragmentTests)
    .setDstStageMask(vk::PipelineStageFlagBits::eColorAttachmentOutput | vk::PipelineStageFlagBits::eEarlyFragmentTests)
    .setSrcAccessMask({})
    .setDstAccessMask(vk::AccessFlagBits::eColorAttachmentWrite | vk::AccessFlagBits::eDepthStencilAttachmentWrite);

  // Render pass
  vk::RenderPassCreateInfo render_pass_create_info;
  render_pass_create_info
    .setAttachments(attachment_descriptions)
    .setSubpasses(subpass)
    .setDependencies(dependency);
  render_pass_ = device.createRenderPass(render_pass_create_info);

  // Framebuffer
  const auto rendertarget = engine_->Rendertarget();
  const auto swapchain_image_count = engine_->SwapchainImageCount();
  const auto& swapchain_image_views = engine_->SwapchainImageViews();
  swapchain_framebuffers_.resize(swapchain_image_count);
  for (uint32_t i = 0; i < swapchain_image_count; i++)
  {
    std::vector<vk::ImageView> attachments{
      rendertarget.color_image_view,
      rendertarget.depth_image_view,
      swapchain_image_views[i],
    };

    vk::FramebufferCreateInfo framebuffer_create_info;
    framebuffer_create_info
      .setRenderPass(render_pass_)
      .setAttachments(attachments)
      .setWidth(width_)
      .setHeight(height_)
      .setLayers(1);
    swapchain_framebuffers_[i] = device.createFramebuffer(framebuffer_create_info);
  }
}

void ParticleRenderer::DestroyFramebuffer()
{
  const auto device = engine_->Device();

  for (auto& framebuffer : swapchain_framebuffers_)
    device.destroyFramebuffer(framebuffer);
  swapchain_framebuffers_.clear();

  device.destroyRenderPass(render_pass_);
}

void ParticleRenderer::CreateGraphicsPipelines()
{
  const auto device = engine_->Device();

  // Create pipeline cache
  pipeline_cache_ = device.createPipelineCache({});

  // Descriptor set layout
  std::vector<vk::DescriptorSetLayoutBinding> descriptor_set_layout_bindings;
  vk::DescriptorSetLayoutBinding descriptor_set_layout_binding;

  descriptor_set_layout_binding
    .setBinding(0)
    .setDescriptorType(vk::DescriptorType::eUniformBuffer)
    .setDescriptorCount(1)
    .setStageFlags(vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment);
  descriptor_set_layout_bindings.push_back(descriptor_set_layout_binding);

  descriptor_set_layout_binding
    .setBinding(1)
    .setDescriptorType(vk::DescriptorType::eUniformBuffer)
    .setDescriptorCount(1)
    .setStageFlags(vk::ShaderStageFlagBits::eFragment);
  descriptor_set_layout_bindings.push_back(descriptor_set_layout_binding);

  descriptor_set_layout_binding
    .setBinding(2)
    .setDescriptorType(vk::DescriptorType::eCombinedImageSampler)
    .setDescriptorCount(1)
    .setStageFlags(vk::ShaderStageFlagBits::eFragment);
  descriptor_set_layout_bindings.push_back(descriptor_set_layout_binding);

  vk::DescriptorSetLayoutCreateInfo descriptor_set_layout_create_info;
  descriptor_set_layout_create_info
    .setBindings(descriptor_set_layout_bindings);
  descriptor_set_layout_ = device.createDescriptorSetLayout(descriptor_set_layout_create_info);

  // Pipeline layout
  vk::PipelineLayoutCreateInfo pipeline_layout_create_info;
  pipeline_layout_create_info
    .setSetLayouts(descriptor_set_layout_);
  pipeline_layout_ = device.createPipelineLayout(pipeline_layout_create_info);

  // Shader modules
  const std::string base_dir = "C:\\workspace\\superlucent\\src\\superlucent\\shader";
  vk::ShaderModule vert_module = engine_->CreateShaderModule(base_dir + "\\floor.vert.spv");
  vk::ShaderModule frag_module = engine_->CreateShaderModule(base_dir + "\\floor.frag.spv");

  // Shader stages
  std::vector<vk::PipelineShaderStageCreateInfo> shader_stages(2);

  shader_stages[0]
    .setStage(vk::ShaderStageFlagBits::eVertex)
    .setModule(vert_module)
    .setPName("main");

  shader_stages[1]
    .setStage(vk::ShaderStageFlagBits::eFragment)
    .setModule(frag_module)
    .setPName("main");

  // Vertex input
  vk::VertexInputBindingDescription vertex_binding_description;
  vertex_binding_description
    .setBinding(0)
    .setStride(sizeof(float) * 2)
    .setInputRate(vk::VertexInputRate::eVertex);

  std::vector<vk::VertexInputAttributeDescription> vertex_attribute_descriptions(1);
  vertex_attribute_descriptions[0]
    .setLocation(0)
    .setBinding(0)
    .setFormat(vk::Format::eR32G32B32Sfloat)
    .setOffset(0);

  vk::PipelineVertexInputStateCreateInfo vertex_input;
  vertex_input
    .setVertexBindingDescriptions(vertex_binding_description)
    .setVertexAttributeDescriptions(vertex_attribute_descriptions);

  // Input assembly
  vk::PipelineInputAssemblyStateCreateInfo input_assembly;
  input_assembly
    .setTopology(vk::PrimitiveTopology::eTriangleStrip)
    .setPrimitiveRestartEnable(false);

  // Viewport
  vk::Viewport viewport_area;
  viewport_area
    .setX(0.f)
    .setY(0.f)
    .setWidth(width_)
    .setHeight(height_)
    .setMinDepth(0.f)
    .setMaxDepth(1.f);

  vk::Rect2D scissor{ { 0u, 0u }, { width_, height_ } };

  vk::PipelineViewportStateCreateInfo viewport;
  viewport
    .setViewports(viewport_area)
    .setScissors(scissor);

  // Rasterization
  vk::PipelineRasterizationStateCreateInfo rasterization;
  rasterization
    .setDepthClampEnable(false)
    .setRasterizerDiscardEnable(false)
    .setPolygonMode(vk::PolygonMode::eFill)
    .setCullMode(vk::CullModeFlagBits::eNone)
    .setFrontFace(vk::FrontFace::eCounterClockwise)
    .setDepthBiasEnable(false)
    .setLineWidth(1.f);

  // Multisample
  vk::PipelineMultisampleStateCreateInfo multisample;
  multisample
    .setRasterizationSamples(vk::SampleCountFlagBits::e4);

  // Depth stencil
  vk::PipelineDepthStencilStateCreateInfo depth_stencil;
  depth_stencil
    .setDepthTestEnable(true)
    .setDepthWriteEnable(true)
    .setDepthCompareOp(vk::CompareOp::eLess)
    .setDepthBoundsTestEnable(false)
    .setStencilTestEnable(false);

  // Color blend
  vk::PipelineColorBlendAttachmentState color_blend_attachment;
  color_blend_attachment
    .setBlendEnable(true)
    .setSrcColorBlendFactor(vk::BlendFactor::eSrcAlpha)
    .setDstColorBlendFactor(vk::BlendFactor::eOneMinusSrcAlpha)
    .setColorBlendOp(vk::BlendOp::eAdd)
    .setSrcAlphaBlendFactor(vk::BlendFactor::eOne)
    .setDstAlphaBlendFactor(vk::BlendFactor::eZero)
    .setColorBlendOp(vk::BlendOp::eAdd)
    .setColorWriteMask(vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA);

  vk::PipelineColorBlendStateCreateInfo color_blend;
  color_blend
    .setLogicOpEnable(false)
    .setAttachments(color_blend_attachment)
    .setBlendConstants({ 0.f, 0.f, 0.f, 0.f });

  // Dynamic states
  std::vector<vk::DynamicState> dynamic_states{
    vk::DynamicState::eViewport,
    vk::DynamicState::eScissor,
  };
  vk::PipelineDynamicStateCreateInfo dynamic_state;
  dynamic_state
    .setDynamicStates(dynamic_states);

  vk::GraphicsPipelineCreateInfo pipeline_create_info;
  pipeline_create_info
    .setStages(shader_stages)
    .setPVertexInputState(&vertex_input)
    .setPInputAssemblyState(&input_assembly)
    .setPViewportState(&viewport)
    .setPRasterizationState(&rasterization)
    .setPMultisampleState(&multisample)
    .setPDepthStencilState(&depth_stencil)
    .setPColorBlendState(&color_blend)
    .setPDynamicState(&dynamic_state)
    .setLayout(pipeline_layout_)
    .setRenderPass(render_pass_)
    .setSubpass(0);
  floor_pipeline_ = CreateGraphicsPipeline(pipeline_create_info);
  device.destroyShaderModule(vert_module);
  device.destroyShaderModule(frag_module);

  // Cell sphere graphics pipeline
  vert_module = engine_->CreateShaderModule(base_dir + "\\cell_sphere.vert.spv");
  frag_module = engine_->CreateShaderModule(base_dir + "\\cell_sphere.frag.spv");

  shader_stages.resize(2);
  shader_stages[0]
    .setStage(vk::ShaderStageFlagBits::eVertex)
    .setModule(vert_module)
    .setPName("main");

  shader_stages[1]
    .setStage(vk::ShaderStageFlagBits::eFragment)
    .setModule(frag_module)
    .setPName("main");

  std::vector<vk::VertexInputBindingDescription> vertex_binding_descriptions(2);
  vertex_binding_descriptions[0]
    .setBinding(0)
    .setStride(sizeof(float) * 6)
    .setInputRate(vk::VertexInputRate::eVertex);

  // Binding 1 shared with particle compute shader
  vertex_binding_descriptions[1]
    .setBinding(1)
    .setStride(sizeof(float) * 24)
    .setInputRate(vk::VertexInputRate::eInstance);

  vertex_attribute_descriptions.resize(5);
  vertex_attribute_descriptions[0]
    .setLocation(0)
    .setBinding(0)
    .setFormat(vk::Format::eR32G32B32Sfloat)
    .setOffset(0);

  vertex_attribute_descriptions[1]
    .setLocation(1)
    .setBinding(0)
    .setFormat(vk::Format::eR32G32B32Sfloat)
    .setOffset(sizeof(float) * 3);

  vertex_attribute_descriptions[2]
    .setLocation(2)
    .setBinding(1)
    .setFormat(vk::Format::eR32G32B32Sfloat)
    .setOffset(offsetof(Particle, position));

  vertex_attribute_descriptions[3]
    .setLocation(3)
    .setBinding(1)
    .setFormat(vk::Format::eR32Sfloat)
    .setOffset(offsetof(Particle, properties));

  vertex_attribute_descriptions[4]
    .setLocation(4)
    .setBinding(1)
    .setFormat(vk::Format::eR32G32B32Sfloat)
    .setOffset(offsetof(Particle, color));

  vertex_input
    .setVertexBindingDescriptions(vertex_binding_descriptions)
    .setVertexAttributeDescriptions(vertex_attribute_descriptions);

  input_assembly
    .setTopology(vk::PrimitiveTopology::eTriangleStrip)
    .setPrimitiveRestartEnable(true);

  rasterization.setCullMode(vk::CullModeFlagBits::eBack);

  pipeline_create_info.setStages(shader_stages);

  cell_sphere_pipeline_ = CreateGraphicsPipeline(pipeline_create_info);
  device.destroyShaderModule(vert_module);
  device.destroyShaderModule(frag_module);
}

void ParticleRenderer::DestroyGraphicsPipelines()
{
  const auto device = engine_->Device();

  device.destroyDescriptorSetLayout(descriptor_set_layout_);
  device.destroyPipeline(floor_pipeline_);
  device.destroyPipeline(cell_sphere_pipeline_);
  device.destroyPipelineLayout(pipeline_layout_);

  device.destroyPipelineCache(pipeline_cache_);
}

void ParticleRenderer::PrepareResources()
{
  const auto device = engine_->Device();
  const auto swapchain_image_count = engine_->SwapchainImageCount();

  // Floor buffer
  constexpr float floor_range = 20.f;
  std::vector<float> floor_vertex_buffer{
    -floor_range, -floor_range,
    floor_range, -floor_range,
    -floor_range, floor_range,
    floor_range, floor_range,
  };
  std::vector<uint32_t> floor_index_buffer{
    0, 1, 2, 3
  };

  const auto floor_vertex_buffer_size = floor_vertex_buffer.size() * sizeof(float);
  const auto floor_index_buffer_size = floor_index_buffer.size() * sizeof(uint32_t);
  const auto floor_buffer_size = floor_vertex_buffer_size + floor_index_buffer_size;

  vk::BufferCreateInfo buffer_create_info;
  buffer_create_info
    .setSize(floor_buffer_size)
    .setUsage(vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eIndexBuffer);

  floor_buffer_.buffer = device.createBuffer(buffer_create_info);
  floor_buffer_.index_offset = floor_vertex_buffer_size;
  floor_buffer_.num_indices = static_cast<uint32_t>(floor_index_buffer.size());

  // Floor texture
  constexpr int floor_texture_length = 128;
  std::vector<uint8_t> floor_texture(floor_texture_length * floor_texture_length * 4);
  for (int u = 0; u < floor_texture_length; u++)
  {
    for (int v = 0; v < floor_texture_length; v++)
    {
      uint8_t color = (255 - 64) + 64 * !((u < floor_texture_length / 2) ^ (v < floor_texture_length / 2));
      floor_texture[(v * floor_texture_length + u) * 4 + 0] = color;
      floor_texture[(v * floor_texture_length + u) * 4 + 1] = color;
      floor_texture[(v * floor_texture_length + u) * 4 + 2] = color;
      floor_texture[(v * floor_texture_length + u) * 4 + 3] = 255;
    }
  }
  constexpr int floor_texture_size = floor_texture_length * floor_texture_length * sizeof(uint8_t) * 4;

  vk::ImageCreateInfo image_create_info;
  image_create_info
    .setImageType(vk::ImageType::e2D)
    .setFormat(vk::Format::eR8G8B8A8Srgb)
    .setExtent(vk::Extent3D{ floor_texture_length, floor_texture_length, 1 })
    .setMipLevels(mipmap_level_)
    .setArrayLayers(1)
    .setSamples(vk::SampleCountFlagBits::e1)
    .setTiling(vk::ImageTiling::eOptimal)
    .setUsage(vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eSampled)
    .setSharingMode(vk::SharingMode::eExclusive)
    .setInitialLayout(vk::ImageLayout::eUndefined);
  floor_texture_.image = device.createImage(image_create_info);

  // Cells buffer
  constexpr int sphere_segments = 8;
  std::vector<float> sphere_buffer;
  std::vector<std::vector<uint32_t>> sphere_indices;
  std::vector<uint32_t> sphere_index_buffer;

  sphere_buffer.push_back(0.f);
  sphere_buffer.push_back(0.f);
  sphere_buffer.push_back(1.f);
  sphere_buffer.push_back(0.f);
  sphere_buffer.push_back(0.f);
  sphere_buffer.push_back(1.f);

  sphere_buffer.push_back(0.f);
  sphere_buffer.push_back(0.f);
  sphere_buffer.push_back(-1.f);
  sphere_buffer.push_back(0.f);
  sphere_buffer.push_back(0.f);
  sphere_buffer.push_back(-1.f);

  uint32_t sphere_index = 2;

  sphere_indices.resize(sphere_segments);
  constexpr auto pi = glm::pi<float>();
  for (int i = 0; i < sphere_segments; i++)
  {
    sphere_indices[i].resize(sphere_segments);

    const auto theta = static_cast<float>(i) / sphere_segments * 2.f * pi;
    const auto cos_theta = std::cos(theta);
    const auto sin_theta = std::sin(theta);
    for (int j = 1; j < sphere_segments; j++)
    {
      sphere_indices[i][j] = sphere_index++;

      const auto phi = (0.5f - static_cast<float>(j) / sphere_segments) * pi;
      const auto cos_phi = std::cos(phi);
      const auto sin_phi = std::sin(phi);

      sphere_buffer.push_back(cos_theta * cos_phi);
      sphere_buffer.push_back(sin_theta * cos_phi);
      sphere_buffer.push_back(sin_phi);
      sphere_buffer.push_back(cos_theta * cos_phi);
      sphere_buffer.push_back(sin_theta * cos_phi);
      sphere_buffer.push_back(sin_phi);
    }
  }
  const auto sphere_vertex_buffer_size = sphere_buffer.size() * sizeof(float);

  // Sphere indices
  for (int i = 0; i < sphere_segments; i++)
  {
    sphere_index_buffer.push_back(0);
    for (int j = 1; j < sphere_segments; j++)
    {
      sphere_index_buffer.push_back(sphere_indices[i][j]);
      sphere_index_buffer.push_back(sphere_indices[(i + 1) % sphere_segments][j]);
    }
    sphere_index_buffer.push_back(1);
    sphere_index_buffer.push_back(-1);
  }
  const auto sphere_index_buffer_size = sphere_index_buffer.size() * sizeof(uint32_t);

  buffer_create_info
    .setSize(sphere_vertex_buffer_size + sphere_index_buffer_size)
    .setUsage(vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eIndexBuffer);
  cells_buffer_.buffer = device.createBuffer(buffer_create_info);
  cells_buffer_.index_offset = sphere_vertex_buffer_size;
  cells_buffer_.num_indices = sphere_index_buffer.size();

  // Memory binding
  const auto floor_memory = engine_->AcquireDeviceMemory(floor_buffer_.buffer);
  device.bindBufferMemory(floor_buffer_.buffer, floor_memory.memory, floor_memory.offset);

  const auto floor_texture_memory = engine_->AcquireDeviceMemory(floor_texture_.image);
  device.bindImageMemory(floor_texture_.image, floor_texture_memory.memory, floor_texture_memory.offset);

  const auto cells_vertex_memory = engine_->AcquireDeviceMemory(cells_buffer_.buffer);
  device.bindBufferMemory(cells_buffer_.buffer, cells_vertex_memory.memory, cells_vertex_memory.offset);

  // Create image view for floor texture
  vk::ImageSubresourceRange subresource_range;
  subresource_range
    .setAspectMask(vk::ImageAspectFlagBits::eColor)
    .setBaseMipLevel(0)
    .setLevelCount(mipmap_level_)
    .setBaseArrayLayer(0)
    .setLayerCount(1);

  vk::ImageViewCreateInfo image_view_create_info;
  image_view_create_info
    .setImage(floor_texture_.image)
    .setViewType(vk::ImageViewType::e2D)
    .setFormat(vk::Format::eR8G8B8A8Srgb)
    .setSubresourceRange(subresource_range);

  floor_texture_.image_view = device.createImageView(image_view_create_info);

  // Transfer
  engine_->ToDeviceMemory(floor_vertex_buffer, floor_buffer_.buffer);
  engine_->ToDeviceMemory(floor_index_buffer, floor_buffer_.buffer, floor_vertex_buffer_size);

  engine_->ToDeviceMemory(sphere_buffer, cells_buffer_.buffer);
  engine_->ToDeviceMemory(sphere_index_buffer, cells_buffer_.buffer, sphere_vertex_buffer_size);

  engine_->ToDeviceMemory(floor_texture, floor_texture_.image, floor_texture_length, floor_texture_length, mipmap_level_);

  // Allocate uniform memory ranges
  const auto uniform_buffer = engine_->UniformBuffer();
  camera_ubos_ = uniform_buffer->Allocate<CameraUbo>(swapchain_image_count);
  light_ubos_ = uniform_buffer->Allocate<LightUbo>(swapchain_image_count);

  // Descriptor set
  std::vector<vk::DescriptorSetLayout> set_layouts(swapchain_image_count, descriptor_set_layout_);
  vk::DescriptorSetAllocateInfo descriptor_set_allocate_info;
  descriptor_set_allocate_info
    .setDescriptorPool(engine_->DescriptorPool())
    .setSetLayouts(set_layouts);
  descriptor_sets_ = device.allocateDescriptorSets(descriptor_set_allocate_info);

  for (int i = 0; i < descriptor_sets_.size(); i++)
  {
    std::vector<vk::DescriptorBufferInfo> buffer_infos(2);
    buffer_infos[0]
      .setBuffer(uniform_buffer->Buffer())
      .setOffset(camera_ubos_[i].offset)
      .setRange(camera_ubos_[i].size);

    buffer_infos[1]
      .setBuffer(uniform_buffer->Buffer())
      .setOffset(light_ubos_[i].offset)
      .setRange(light_ubos_[i].size);

    std::vector<vk::DescriptorImageInfo> image_infos(1);
    image_infos[0]
      .setSampler(sampler_)
      .setImageView(floor_texture_.image_view)
      .setImageLayout(vk::ImageLayout::eShaderReadOnlyOptimal);

    std::vector<vk::WriteDescriptorSet> descriptor_writes(3);
    descriptor_writes[0]
      .setDstSet(descriptor_sets_[i])
      .setDstBinding(0)
      .setDstArrayElement(0)
      .setDescriptorType(vk::DescriptorType::eUniformBuffer)
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
      .setDescriptorType(vk::DescriptorType::eCombinedImageSampler)
      .setImageInfo(image_infos[0]);

    device.updateDescriptorSets(descriptor_writes, {});
  }
}

void ParticleRenderer::DestroyResources()
{
  const auto device = engine_->Device();

  descriptor_sets_.clear();

  device.destroyBuffer(floor_buffer_.buffer);
  device.destroyBuffer(cells_buffer_.buffer);
  device.destroyImage(floor_texture_.image);
  device.destroyImageView(floor_texture_.image_view);
}

vk::Pipeline ParticleRenderer::CreateGraphicsPipeline(vk::GraphicsPipelineCreateInfo& create_info)
{
  const auto device = engine_->Device();
  auto result = device.createGraphicsPipeline(pipeline_cache_, create_info);
  if (result.result != vk::Result::eSuccess)
    throw std::runtime_error("Failed to create graphics pipeline, with error code: " + vk::to_string(result.result));
  return result.value;
}
}
}
