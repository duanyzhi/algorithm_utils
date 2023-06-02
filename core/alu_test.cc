#include "alu.h"
#include "cv/core/methods.h"
#include "core/tensor.h"

#include <iostream>

namespace alu {

void test_alu() {
  int kernel_size = 3;
  auto kernel_tensor = alu::cv::Get2DGaussianKernel(kernel_size);

  double* data_ptr = static_cast<double*>(kernel_tensor.data()); 

  for (int i = 0; i < 9; i++) {
    std::cout << *data_ptr << " ";
    data_ptr++;
  }  
  std::cout << "\n";
}

}
