#include "core/tensor.h"

#include <memory>

namespace alu {

inline void* aligned_malloc(size_t size_bytes, size_t alignment) {
  void* memptr = nullptr; 
  const bool state = posix_memalign(&memptr,  alignment, size_bytes);
  const bool null_mem = state || (nullptr == memptr);
  assert(null_mem == false);
  return memptr;
}

void aligned_free(void* data_ptr) {
  if (data_ptr) {
    free(data_ptr);
  }
}

template <>
struct alutype_traits<AluType::ABOOL> {
  // typedef AluType::ABOOL value;
  using type = bool; 
};

template <>
struct alutype_traits<AluType::AINT> {
  // using value = AluType::AINT;
  using type = int; 
};

template <>
struct alutype_traits<AluType::ALINT> {
  // using value = AluType::ALINT;
  using type = int64_t; 
};

template <>
struct alutype_traits<AluType::AFLOAT> {
  // using value = AluType::AFLOAT;
  using type = float; 
};

template <>
struct alutype_traits<AluType::ADOUBLE> {
  // using value = AluType::ADOUBLE;
  using type = double; 
};

Tensor::Tensor(const int& width, const int& height, AluType type) {
  const size_t capacity = width * height;
#define __ALU_ALLOCATOR_CASE(ALLOCATORTYPE, _TP, _) \
  case (ALLOCATORTYPE): {                        \
    impl_ = new TensorBase<ALLOCATORTYPE>(width, height);   \
    break;                                       \
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

Tensor::~Tensor() {
  delete impl_;
}

const TensorInfo Tensor::info() const {
  return impl_->info();
}

const Tensor& Tensor::fill(const Scalar& value) const {
  impl_->fill(value);
  return *this;
}

const Tensor& Tensor::set(const int& index, const Scalar &value) const {
  impl_->set(index, value);
  return *this;
}

void* Tensor::data() const {
  return impl_->data_ptr();
}

Scalar Tensor::operator[](int index) const {
  // int index = w * info().width + h; 
  return impl_->data(index);
}

std::ostream& operator<<(std::ostream& os, const Scalar& s) {
  if (s.type() == AluType::ABOOL) {
    os << s.to<bool>();
  } else {
    os << s.to<double>();
  }
  return os;
}

std::ostream& operator<<(std::ostream& os, const AluType& type) {
  if (type == AluType::ABOOL) os << "bool";
  if (type == AluType::ADOUBLE) os << "double";
  return os;
}

}  // namespace alu
