#include "alu.h"
#include "arithmetic/convolution.h"
#include "common/utils.h"
#include "core/tensor.h"
#include "cv/core/methods.h"
#include <iostream>

namespace alu {

void test_alu() {
  int kernel_size = 3;
  auto kernel_tensor = alu::cv::Get2DGaussianKernel(kernel_size);

  { Tensor aa(3, 3); }
  double *data_ptr = static_cast<double *>(kernel_tensor.data());

  for (int i = 0; i < 9; i++) {
    std::cout << *data_ptr << ", ";
    std::cout << kernel_tensor[i] << " ;";
    data_ptr++;
  }
  std::cout << "\n";
  // int width = kernel_tensor.info().width;
  // int height = kernel_tensor.info().height;
  // for (int i = 0; i < width; i++) {
  //   for (int j = 0; j < height; j++) {
  //      std::cout << kernel_tensor[i, j] << " ";
  //   }
  // }
  std::cout << "\n";

  {
    auto input = alu::cv::Get2DGaussianKernel(10);
    alu::rect roi(0, 0, 3, 3);
    std::cout << "roi test " << input(roi);
    auto output = alu::Convolution2D(input, kernel_tensor);
    std::cout << input;
    std::cout << kernel_tensor;
    std::cout << output;
  }
  {
    auto input = alu::load("/workspace/algorithm_utils/data/im.png");
    std::cout << "input: " << input;
    auto output = alu::cv::CannyEdgeDetection(input);
    alu::save_image(output, "/workspace/algorithm_utils/data/output.png");
  }
}

} // namespace alu
