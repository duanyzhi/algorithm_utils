#ifndef ALU_ARITHMETIC_CONVOLUTION_H_
#define ALU_ARITHMETIC_CONVOLUTION_H_

#include "core/tensor.h"

namespace alu {

Tensor Convolution2D(const Tensor& input, const Tensor& weight);

}  // namespace alu

#endif  //  ALU_ARITHMETIC_CONVOLUTION_H_
