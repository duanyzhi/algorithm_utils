#include "core/tensor.h"

#include <memory>
#include <sstream>

namespace alu {

inline void *aligned_malloc(size_t size_bytes, size_t alignment) {
  void *memptr = nullptr;
  const bool state = posix_memalign(&memptr, alignment, size_bytes);
  const bool null_mem = state || (nullptr == memptr);
  assert(null_mem == false);
  return memptr;
}

void aligned_free(void *data_ptr) {
  if (data_ptr) {
    free(data_ptr);
  }
}

template <> struct alutype_traits<AluType::ABOOL> {
  // typedef AluType::ABOOL value;
  static constexpr size_t bytes = 1;
  using type = bool;
};

template <> struct alutype_traits<AluType::AINT> {
  // using value = AluType::AINT;
  static constexpr size_t bytes = 4;
  using type = int;
};

template <> struct alutype_traits<AluType::ALINT> {
  // using value = AluType::ALINT;
  static constexpr size_t bytes = 8;
  using type = int64_t;
};

template <> struct alutype_traits<AluType::AFLOAT> {
  // using value = AluType::AFLOAT;
  static constexpr size_t bytes = 4;
  using type = float;
};

template <> struct alutype_traits<AluType::ADOUBLE> {
  // using value = AluType::ADOUBLE;
  static constexpr size_t bytes = 8;
  using type = double;
};

Tensor::Tensor(const int &width, const int &height, AluType type) {
  const size_t capacity = width * height;
#define __ALU_ALLOCATOR_CASE(ALLOCATORTYPE, _TP, _)                            \
  case (ALLOCATORTYPE): {                                                      \
    impl_ = new TensorBase<ALLOCATORTYPE>(width, height);                      \
    break;                                                                     \
  }

  switch (type) {
    _ALUTYPE_MAP(__ALU_ALLOCATOR_CASE)
  default: {
    // LOG(ERROR) << "Not suppot input allocator type " << allocator_type;
    break;
  }
  }
#undef __ALU_ALLOCATOR_CASE
}

Tensor::~Tensor() { delete impl_; }

const TensorInfo Tensor::info() const { return impl_->info(); }

const Tensor &Tensor::fill(const Scalar &value) const {
  impl_->fill(value);
  return *this;
}

const Tensor &Tensor::set(const int &index, const Scalar &value) const {
  impl_->set(index, value);
  return *this;
}

const Tensor &Tensor::set(int row, int col, const Scalar &value) const {
  int index = row * info().width + col;
  impl_->set(index, value);
  return *this;
}

void *Tensor::data() const { return impl_->data_ptr(); }

const Tensor Tensor::mul(const Tensor &other) {
  assert(info().type == other.dtype());
  Tensor output(info().width, info().height, info().type);
  impl_->mul(other.data(), output.data());
  return output;
}

Scalar Tensor::operator[](int index) const { return impl_->data(index); }

Scalar Tensor::data(int row, int col) const {
  assert(row < info().height);
  assert(col < info().width);
  int index = row * info().width + col;
  return impl_->data(index);
}

Tensor Tensor::operator+(const Tensor &other) const {
  assert(info().type == other.dtype());
  Tensor output(info());
  impl_->add(other.data(), output.data());
  return output;
}

Tensor Tensor::operator/(const double value) const {
  Tensor output(info());
  impl_->div(value, output.data());
  return output;
}

Tensor Tensor::operator*(const double value) const {
  Tensor output(info());
  impl_->mul(value, output.data());
  return output;
}

Tensor Tensor::operator()(const alu::rect &roi) const {
  Tensor output(roi.w, roi.h, this->dtype());
  assert(roi.x >= 0 && roi.y >= 0 && roi.w >= 0 && roi.h >= 0);
  assert(roi.x + roi.w <= info().width);
  assert(roi.y + roi.h <= info().height);
  impl_->roi(roi, output.data());
  return output;
}

// Tensor &Tensor::operator=(const Tensor &other) {
//  this->impl_ = other.impl();
//  return *this;
//}

Tensor Tensor::abs() {
  Tensor output(info());
  impl_->abs(output.data());
  return output;
}

std::ostream &operator<<(std::ostream &os, const Scalar &s) {
  if (s.type() == AluType::ABOOL) {
    os << s.to<bool>();
  } else {
    os << s.to<double>();
  }
  return os;
}

std::ostream &operator<<(std::ostream &os, const AluType &type) {
  if (type == AluType::ABOOL)
    os << "bool";
  if (type == AluType::ADOUBLE)
    os << "double";
  return os;
}

std::ostream &operator<<(std::ostream &os, const Tensor &t) {
  std::stringstream ss;
  ss << "Tensor info: [" << t.info().width << ", " << t.info().height << ", "
     << t.info().type << "]\n";
  for (int h = 0; h < t.info().height; h++) {
    for (int w = 0; w < t.info().width; w++) {
      ss << t[h * t.info().width + w] << " ";
    }
    ss << "\n";
  }
  ss << "\n";
  os << ss.str();
  return os;
}

Scalar max(const Scalar &lhs, const Scalar &rhs) {
  return lhs > rhs ? lhs : rhs;
}

} // namespace alu
