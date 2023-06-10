#ifndef ALU_CV_CORE_METHODS_H_
#define ALU_CV_CORE_METHODS_H_

#include "core/tensor.h"
#include "interface/common.h"

namespace alu {

namespace cv {

Tensor Get2DGaussianKernel(const int &kernel_size, double deviation = 1.0);

void GaussianBlur();

constexpr int sobelgx[]{-1, 0, 1, -2, 0, 2, -1, 0, 1};
constexpr int sobelgy[]{1, 2, 1, 0, 0, 0, -1, -2, -1};

Tensor CannyEdgeDetection(const Tensor &input);

class Canny {
public:
  Tensor smooth(const Tensor &input);
  void finding_gradients(const Tensor &input);
  Tensor detection(const Tensor &input);
  void nms();

private:
  Tensor magnitude_, theta_;
};

} // namespace cv
} // namespace alu

#endif // ALU_CV_CORE_METHODS_H_
