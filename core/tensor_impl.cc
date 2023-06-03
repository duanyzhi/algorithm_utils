#include "core/tensor.h"

namespace alu {

TensorImpl::TensorImpl(TensorBuffer* buffer, int width, int height, AluType type)
  : buffer_(buffer) {
  std::cout << "impl construct " << width << " " << height << "\n";
  info_.width = width;
  info_.height = height;
  info_.numel = width * height;
  info_.type = type;
}

TensorImpl::~TensorImpl() {
  delete buffer_;
}

void* TensorImpl::data_ptr() const {
  return buffer_->data();
}

}  // namespace alu
