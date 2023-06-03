#ifndef ALU_ARITHMETIC_CONVOLUTION_H_
#define ALU_ARITHMETIC_CONVOLUTION_H_

#include "core/tensor.h"
#include "interface/common.h"

namespace alu {

Tensor Convolution2D(const Tensor& input, const Tensor& weight);

Scalar Convolution2DWithRoi(const Tensor& input, const Tensor& weight, alu::rect roi = alu::rect());

}  // namespace alu

#endif  //  ALU_ARITHMETIC_CONVOLUTION_H_
