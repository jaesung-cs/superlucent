#ifndef VKPBD_BUFFER_REQUIREMENTS_H_
#define VKPBD_BUFFER_REQUIREMENTS_H_

#include <vulkan/vulkan.hpp>

namespace vkpbd
{
struct BufferRequirements
{
  vk::BufferUsageFlags usage;
  vk::DeviceSize size = 0;
};
}

#endif // VKPBD_BUFFER_REQUIREMENTS_H_
