#ifndef VKPBD_FLUID_INDIRECT_GLSL_
#define VKPBD_FLUID_INDIRECT_GLSL_

layout(std140, binding = 5) buffer DispatchIndirectoCommandSsbo
{
  uvec4 dispatch_indirect_commands[];
};

#endif // VKPBD_FLUID_INDIRECT_GLSL_
