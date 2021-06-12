#ifndef SUPERLUCENT_UTILS_RNG_H_
#define SUPERLUCENT_UTILS_RNG_H_

#include <random>

namespace supl
{
namespace utils
{
class Rng
{
public:
  Rng();

  template <typename T>
  T Uniform(T a, T b) const
  {
    return a + (b - a) * static_cast<T>(dis_(gen_));
  }

private:
  std::random_device rd_;
  mutable std::mt19937 gen_;
  std::uniform_real_distribution<> dis_;
};
}
}

#endif // SUPERLUCENT_UTILS_RNG_H_
