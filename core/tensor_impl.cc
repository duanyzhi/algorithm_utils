#include "core/tensor.h"

namespace alu {

TensorImpl::TensorImpl(TensorBuffer* buffer, int width, int height)
  : buffer_(buffer) {
  info_.width = width;
  info_.height = height;
  info_.numel = width * height;
}

TensorImpl::~TensorImpl() {
  delete buffer_;
}

void* TensorImpl::data_ptr() const {
  return buffer_->data();
}

}  // namespace alu
