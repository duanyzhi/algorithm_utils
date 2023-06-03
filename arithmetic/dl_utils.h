#ifndef ALU_ARITHMETIC_DL_UTILS_H_
#define ALU_ARITHMETIC_DL_UTILS_H_

#include "core/tensor.h"

namespace alu {

int conv_output_size(const int& in, const int& w, int pad = 0, int stride = 1);

}  // namespace alu

#endif  //  ALU_ARITHMETIC_CONVOLUTION_H_
