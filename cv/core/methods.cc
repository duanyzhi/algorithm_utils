#include "cv/core/methods.h"
#include "arithmetic/mathematic.h"
#include "common/logging.h"

namespace alu {
namespace cv {

Tensor Get2DGaussianKernel(const int& kernel_size, double deviation) {
  Tensor kernel_tensor(kernel_size, kernel_size, alu::AluType::ADOUBLE);
  double* data_ptr = static_cast<double*>(kernel_tensor.data());
  int range = kernel_size / 2;
  int cc = kernel_size % 2 == 0 ? 0 : 1;
  for (int x = -range; x < range + cc; x++) {
    for (int y = -range; y < range + cc; y++) {
      *data_ptr = alu::math::GaussianDistribution2D(x, y, deviation);
      data_ptr++;
    }
  }
  return kernel_tensor;
}


}  // namespace cv
}  // namespace alu


