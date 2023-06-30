#ifndef ALU_CV_CORE_METHODS_H_
#define ALU_CV_CORE_METHODS_H_

#include <vector>

#include "core/tensor.h"
#include "interface/common.h"

namespace alu {

namespace cv {

Tensor Get2DGaussianKernel(const int &kernel_size, double deviation = 1.0);

void GaussianBlur();

constexpr int sobelgx[]{-1, 0, 1, -2, 0, 2, -1, 0, 1};
constexpr int sobelgy[]{1, 2, 1, 0, 0, 0, -1, -2, -1};

Tensor CannyEdgeDetection(const Tensor &input);

struct LineType {
  int cb, ce, r; // cb: begin pixel, cd: end pixel, r: rows
  LineType(int cbIn, int ceIn, int rIn) : cb(cbIn), ce(ceIn), r(rIn) {}
  LineType() : LineType(0, 0, 0) {}
};

typedef std::vector<LineType> LVec;

class Region {
public:
  Region() {}
  bool empty() const { return lines.empty(); }
  LVec lines;

private:
  int width_ = 0;
  int height_ = 0;
};

Tensor threshold(const Tensor &input, double lower = 0, double upper = 255.0);

Region Tensor2Region(const Tensor &input);

class Canny {
public:
  Tensor smooth(const Tensor &input);
  void finding_gradients(const Tensor &input);
  Tensor detection(const Tensor &input);
  void nms();
  void double_threshold();

private:
  Tensor magnitude_, theta_;
};

} // namespace cv
} // namespace alu

#endif // ALU_CV_CORE_METHODS_H_
