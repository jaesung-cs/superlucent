#ifndef SUPERLUCENT_VKSYNC_VECTOR_H_
#define SUPERLUCENT_VKSYNC_VECTOR_H_

#include <superlucent/vksync/sync.h>

namespace supl
{
namespace vksync
{
template <typename T>
class Vector
{
public:
  Vector() = delete;

  Vector(Sync* sync, int size)
    : sync_(sync)
  {
  }

private:
  Sync* const sync_ = nullptr;
};
}
}

#endif // SUPERLUCENT_VKSYNC_VECTOR_H_
