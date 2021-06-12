#include <superlucent/utils/rng.h>

namespace supl
{
namespace utils
{
Rng::Rng()
  : rd_{}
  , gen_{ rd_() }
  , dis_{ 0., 1. }
{
}
}
}
