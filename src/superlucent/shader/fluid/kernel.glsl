#define SUPERLUCENT_PI 3.1415926535897932384626433832795f

// Kernel defined in Position-Based Fluid Simulation paper
//   r: distance between particles

float pow2(float x)
{
  return x * x;
}

float pow3(float x)
{
  return x * x * x;
}

float pow6(float x)
{
  return pow2(pow3(x));
}

float pow9(float x)
{
  return pow3(pow3(x));
}

float KernelPoly6(float r, float h)
{
  return r <= h ? 315.f / (64.f * SUPERLUCENT_PI * pow9(h)) * pow3(h * h - r * r) : 0.f;
}

float KernelSpiky(float r, float h)
{
  return r <= h ? 15.f / (SUPERLUCENT_PI * pow6(h)) * pow3(h - r) : 0.f;
}

float KernelViscosity(float r, float h)
{
  return r <= h ? 15.f / (2.f * SUPERLUCENT_PI * pow3(h)) * (-pow3(r) / (2.f * pow3(h)) + pow2(r) / pow2(h) + h / (2.f * r) - 1.f) : 0.f;
}
