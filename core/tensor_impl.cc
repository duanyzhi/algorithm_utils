#include "core/tensor.h"
#include <cmath>

namespace alu {

TensorImpl::TensorImpl(TensorBuffer *buffer, int width, int height,
                       AluType type, size_t size_bytes)
    : buffer_(buffer) {
  info_.width = width;
  info_.height = height;
  info_.numel = width * height;
  info_.type = type;
  info_.size_bytes = size_bytes;
}

TensorImpl::~TensorImpl() { delete buffer_; }

void *TensorImpl::data_ptr() const { return buffer_->data(); }

Tensor arctan(const Tensor &a, const Tensor &b) {
  assert(a.info().width == b.info().width);
  assert(a.info().height == b.info().height);
  assert(a.info().type == AluType::ADOUBLE);
  assert(b.info().type == AluType::ADOUBLE);
  Tensor output(a.info().width, a.info().height);
  auto a_buffer = static_cast<double *>((void *)a.data());
  auto b_buffer = static_cast<double *>((void *)b.data());
  auto out_buffer = static_cast<double *>((void *)output.data());
  for (int i = 0; i < output.info().numel; i++) {
    auto y = static_cast<double>(*(a_buffer + i));
    auto x = static_cast<double>(*(b_buffer + i));
    *(out_buffer + i) = std::atan2(y, x);
  }
  return output;
}

// move to tensor base template
Tensor sqrt(const Tensor &input) {
  assert(input.info().type == AluType::ADOUBLE);
  Tensor output(input.info());
  auto input_buffer = static_cast<double *>((void *)input.data());
  auto out_buffer = static_cast<double *>((void *)output.data());
  for (int i = 0; i < output.info().numel; i++) {
    auto value = static_cast<double>(*(input_buffer + i));
    assert(value > 0 || value == 0);
    *(out_buffer + i) = std::sqrt(value);
  }
  return output;
}

} // namespace alu
