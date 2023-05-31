#include "cv/core/methods.h"
#include "arithmetic/mathematic.h"
#include "common/logging.h"

namespace alu {
namespace cv {

Tensor Get2DGaussianKernel(const int& kernel_size, double deviation) {
  Tensor kernel_tensor(kernel_size, kernel_size, alu::AluType::ADOUBLE);
  double* data_ptr = static_cast<double*>(kernel_tensor.GetTensorBuffer()->data());
  int range = kernel_size / 2;
  LOG("range " << range);
  for (int x = -range; x < range + 1; x++) {
    for (int y = -range; y < range + 1; y++) {
      auto value = alu::math::GaussianDistribution2D(x, y, deviation);
      LOG(value);
      *data_ptr = value;
      data_ptr++;
    }
  }
  return kernel_tensor;
}


}  // namespace cv
}  // namespace alu


