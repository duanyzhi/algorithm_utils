#include <cassert>

#include "arithmetic/convolution.h"
#include "arithmetic/dl_utils.h"
#include "interface/common.h"

namespace alu {

// TODO: optimation
Scalar Convolution2DWithRoi(const Tensor &input, const Tensor &weight,
                            alu::rect roi) {
  if (roi.empty()) {
    roi = alu::rect(0, 0, weight.info().width, weight.info().height);
  }
  assert(roi.w == weight.info().width);
  assert(roi.h == weight.info().height);
  Scalar output(double(0.0));
  for (int x = roi.x; x < roi.x + roi.w; x++) {
    for (int y = roi.y; y < roi.y + roi.h; y++) {
      int index = x * input.info().width + y;
      int w_index = (x - roi.x) * roi.w + (y - roi.y);
      output = output + input[index] * weight[w_index];
    }
  }
  return output;
}

Tensor Convolution2D(const Tensor &input, const Tensor &weight) {
  int ow = conv_output_size(input.info().width, weight.info().width);
  int oh = conv_output_size(input.info().height, weight.info().height);
  std::cout << ow << " " << oh << " " << input.dtype() << "\n";
  Tensor output(ow, oh, input.dtype());

  int sw = weight.info().width / 2;
  int ew = input.info().width - sw;
  int sh = weight.info().height / 2;
  int eh = input.info().height - sh;
  for (int iw = sw; iw < ew; iw++) {
    for (int ih = sh; ih < eh; ih++) {
      alu::rect roi(iw - sw, ih - sh, weight.info().width,
                    weight.info().height);
      auto scalar_out = Convolution2DWithRoi(input, weight, roi);
      int index = (iw - sw) * ow + (ih - sh);
      output.set(index, scalar_out);
    }
  }
  return output;
}

} // namespace alu
