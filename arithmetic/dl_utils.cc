#include "arithmetic/dl_utils.h"
#include <cassert>

namespace alu {

int conv_output_size(const int& in, const int& w, int pad, int stride) {
  assert(stride > 0);
  int o = (in + (2 * pad) - w) / stride + 1;
  return o;
}

}  // namespace alu
