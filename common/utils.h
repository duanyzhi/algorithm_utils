#ifndef ALU_COMMON_UTILS_H_
#define ALU_COMMON_UTILS_H_

#include "core/tensor.h"
#include "interface/common.h"

namespace alu {

Tensor load(const std::string &image);
Tensor load(const char *image);
void save_image(const Tensor &tensor, const std::string &name);

} // namespace alu

#endif // ALU_COMMON_UTILS_H_
