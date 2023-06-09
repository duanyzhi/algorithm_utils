#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include "common/utils.h"

namespace alu {

Tensor load(const std::string &image) {
  auto im = cv::imread(image, cv::IMREAD_UNCHANGED);
  im.convertTo(im, CV_64F);
  auto im_data = im.data;
  Tensor output(im.cols, im.rows);
  memcpy(output.data(), (void *)(im.data),
         output.info().numel * sizeof(double));
  return output;
}

Tensor load(const char *image) { return load(std::string(image)); }

void save_image(const Tensor &tensor, const std::string &name) {
  cv::Mat im(tensor.info().height, tensor.info().width, CV_64F);
  memcpy((void *)(im.data), tensor.data(),
         tensor.info().numel * sizeof(double));
  cv::imwrite(name, im);
}

} // namespace alu
