#ifndef VKPBD_FLUID_KERNEL_GLSL_
#define VKPBD_FLUID_KERNEL_GLSL_

#include "fluid_simulation_params.glsl"

// See SPHKernels.h in PositionBasedDynamics repo
float KernelWZero()
{
  return 0.f;
}

float KernelW(vec3 r)
{
  const float k = params.kernel_constants[0];

  const float rl = length(r);
  const float q = rl / params.radius;
  const float q2 = q * q;
  const float q3 = q2 * q;

  if (q <= 0.5f)
    return k * (6.f * q3 - 6.f * q2 + 1.f);
  else
    return k * (2.f * pow(1.f - q, 3));
}

vec3 KernelGradW(vec3 r, float h)
{
  const float l = params.kernel_constants[1];

  const float rl = length(r);
  const float q = rl / params.radius;
  const float q2 = q * q;
  const float q3 = q2 * q;

  if (rl > 1e-6f)
  {
    const vec3 gradQ = r / (rl * h);
    if (q <= 0.5f)
      return l * q * (3.f * q - 2.f) * gradQ;
    else
    {
      const float factor = 1.f - f;
      return l * (-factor * factor) * gradQ;
    }
  }
  else
    return vec3(0.f);
}

#endif // VKPBD_FLUID_KERNEL_GLSL_
