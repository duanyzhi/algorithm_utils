#include "cv/core/methods.h"
#include "arithmetic/mathematic.h"
#include "common/logging.h"

namespace alu {
namespace cv {

Tensor Get2DGaussianKernel(const int &kernel_size, double deviation) {
  Tensor kernel_tensor(kernel_size, kernel_size, alu::AluType::ADOUBLE);
  double *data_ptr = static_cast<double *>(kernel_tensor.data());
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

Tensor threshold(const Tensor &input, double lower, double upper) {
  Tensor output = input;
  lower = lower < 0 ? 0 : lower;
  upper = upper > 255.0 ? 255.0 : upper;
  Scalar lv(lower), uv(upper);
  for (int index = 0; index < output.info().numel; index++) {
    if ((output[index] < lv) || (output[index] > uv)) {
      output.set(index, Scalar(0));
    }
  }
  return output;
}

Region Tensor2Region(const Tensor &input) {
  Region region;
  for (int h = 0; h < input.info().height; h++) {
    int st{0}, et{0};
    for (int w = 0; w < input.info().width; w++) {
      if (!input.data(h, w) && 0 == st)
        continue;
      if (!input.data(h, w) && 0 != st) {
        LineType line(st, et, h);
        region.lines.push_back(line);
        st = 0;
        et = 0;
        continue;
      }
      if (st == 0) {
        st = w;
        et = w;
      } else {
        et++;
      }
    }
    if (et != 0) {
      LineType line(st, et, h);
      region.lines.push_back(line);
    }
  }
  return region;
}

} // namespace cv
} // namespace alu
