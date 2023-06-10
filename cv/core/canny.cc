#include "arithmetic/convolution.h"
#include "arithmetic/mathematic.h"
#include "cv/core/methods.h"

namespace alu {
namespace cv {

Tensor Canny::smooth(const Tensor &input) {
  // 1. smoothing with gaussian filter
  int kernel_size = 3;
  auto kernel_tensor = Get2DGaussianKernel(kernel_size);
  auto output = alu::Convolution2D(input, kernel_tensor);
  return output;
}

void Canny::finding_gradients(const Tensor &input) {
  Tensor KGX(3, 3);
  Tensor KGY(3, 3);
  for (int i = 0; i < KGX.info().numel; i++) {
    KGX.set(i, Scalar(sobelgx[i]));
  }
  for (int i = 0; i < KGY.info().numel; i++) {
    KGY.set(i, Scalar(sobelgy[i]));
  }
  // std::cout << "KGXANDGY " << KGX << " " << KGY << "\n";
  auto GX = alu::Convolution2D(input, KGX);
  auto GY = alu::Convolution2D(input, KGY);
  // std::cout << "GX " << GX << "\n";
  // std::cout << "GY " << GY << "\n";
  // compute gradient magnitude with euclidean distance
  magnitude_ = alu::sqrt(GX.mul(GX) + GY.mul(GY));
  auto radian = arctan(GY.abs(), GX.abs());
  theta_ = radian * 180 / ALU_PI;
}

void Canny::nms() {
  // common pixel from 0 to height - 1
  for (int y = 1; y < magnitude_.info().height - 2; y++) {
    for (int x = 1; x < magnitude_.info().width - 2; x++) {
      // get 8-connected neighbourhood
      alu::rect roi(x, y, 3, 3);
      auto mag_neighbor = magnitude_(roi);
      auto mag_center = mag_neighbor.data(1, 1);
      auto theta = theta_.data(y, x);
      Scalar max_mag;
      if (((theta >= 0) && (theta < 22.5)) ||
          (theta >= 157.5) && (theta <= 180)) {
        max_mag = alu::max(mag_neighbor.data(1, 0), mag_neighbor.data(1, 2));
      } else if ((theta >= 22.5) && (theta < 67.5)) {
        max_mag = alu::max(mag_neighbor.data(0, 0), mag_neighbor.data(2, 2));
      } else if ((theta >= 67.5) && (theta < 112.5)) {
        max_mag = alu::max(mag_neighbor.data(0, 1), mag_neighbor.data(2, 1));
      } else if ((theta >= 112.5) && (theta < 157.5)) {
        max_mag = alu::max(mag_neighbor.data(0, 2), mag_neighbor.data(2, 0));
      }
      if (max_mag > mag_center) {
        magnitude_.set(y, x, Scalar(0.0));
      }
    }
  }
}

Tensor Canny::detection(const Tensor &input) {
  auto smooth_tensor = smooth(input);
  finding_gradients(smooth_tensor);
  nms();
  return magnitude_;
}

Tensor CannyEdgeDetection(const Tensor &input) {
  Canny method;
  return method.detection(input);
}

} // namespace cv
} // namespace alu
