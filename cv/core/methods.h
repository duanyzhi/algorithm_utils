#ifndef ALU_CV_CORE_METHODS_H_
#define ALU_CV_CORE_METHODS_H_

#include "interface/common.h"
#include "core/tensor.h"

namespace alu {
namespace cv {

Tensor Get2DGaussianKernel(const int& kernel_size, double deviation = 1.0);

void GaussianBlur();

}  // namespace cv
}  // namespace alu

#endif  // ALU_CV_CORE_METHODS_H_
