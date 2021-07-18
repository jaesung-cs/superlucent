#ifndef VKPBD_FLUID_KERNEL_GLSL_
#define VKPBD_FLUID_KERNEL_GLSL_

#include "fluid_simulation_params.glsl"

// See SPHKernels.h in PositionBasedDynamics repo
float KernelW(vec3 r)
{
  const float k = params.kernel_constants[0];

  const float rl = length(r);
  const float q = rl / (4.f * params.radius);
  const float q2 = q * q;
  const float q3 = q2 * q;

  if (q <= 0.5f)
    return k * (6.f * q3 - 6.f * q2 + 1.f);
  else if (q <= 1.f)
    return k * (2.f * pow(1.f - q, 3));
  else
    return 0.f;
}

float KernelWZero()
{
  return KernelW(vec3(0.f));
}

vec3 KernelGradW(vec3 r)
{
  const float l = params.kernel_constants[1];

  const float rl = length(r);
  const float q = rl / (4.f * params.radius);
  const float q2 = q * q;
  const float q3 = q2 * q;

  if (rl > 1e-6f)
  {
    const vec3 grad_q = r / (rl * (4.f * params.radius));
    if (q <= 0.5f)
      return l * q * (3.f * q - 2.f) * grad_q;
    else if (q <= 1.f)
    {
      const float factor = 1.f - q;
      return l * (-factor * factor) * grad_q;
    }
    else
      return vec3(0.f);
  }
  else
    return vec3(0.f);
}

#endif // VKPBD_FLUID_KERNEL_GLSL_
