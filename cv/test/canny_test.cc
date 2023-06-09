#include "alu.h"
#include "arithmetic/convolution.h"
#include "core/tensor.h"
#include "cv/core/methods.h"
#include <iostream>

namespace alu {

void canny_test() {
  // int kernel_size = 3;
  // auto kernel_tensor = alu::cv::Get2DGaussianKernel(kernel_size);

  // { Tensor aa(3, 3); }
  // double *data_ptr = static_cast<double *>(kernel_tensor.data());

  // for (int i = 0; i < 9; i++) {
  //   std::cout << *data_ptr << ", ";
  //   std::cout << kernel_tensor[i] << " ;";
  //   data_ptr++;
  // }
  // std::cout << "\n";
  // // int width = kernel_tensor.info().width;
  // // int height = kernel_tensor.info().height;
  // // for (int i = 0; i < width; i++) {
  // //   for (int j = 0; j < height; j++) {
  // //      std::cout << kernel_tensor[i, j] << " ";
  // //   }
  // // }
  // std::cout << "\n";

  // {
  //   auto input = alu::cv::Get2DGaussianKernel(10);
  //   auto output = alu::Convolution2D(input, kernel_tensor);
  //   std::cout << input;
  //   std::cout << kernel_tensor;
  //   std::cout << output;
  // }
}

} // namespace alu
