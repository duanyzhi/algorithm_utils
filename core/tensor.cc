#include "core/tensor.h"

#include <memory>

namespace alu {

inline void* aligned_malloc(size_t size, size_t alignment) {
  void* memptr = nullptr; 
  const bool state = posix_memalign(&memptr,  alignment, size);
  const bool null_mem = state || (nullptr == memptr);
  return memptr;
}


void aligned_free(void* data_ptr) {
  if (data_ptr) {
    free(data_ptr);
  }
}

template <typename _Tp>
DataStorage<_Tp>::DataStorage(size_t N)
    : TensorBuffer(TypedAllocator::Allocate<_Tp>(N)),
      elem_(N) {}

template <typename _Tp>
DataStorage<_Tp>::~DataStorage() {
  if (data()) {
    TypedAllocator::Deallocate<_Tp>(static_cast<_Tp*>(data()));
  }
}

Tensor::Tensor(const int& width, const int& height, AluType type) {
  const size_t capacity = width * height;
#define __ALU_ALLOCATOR_CASE(ALLOCATORTYPE, _TP) \
  case (ALLOCATORTYPE): {                        \
    buffer_ = new DataStorage<_TP>(capacity);   \
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
  info_.width = width;
  info_.height = height;
  info_.type = type;
}

Tensor::~Tensor() {
  delete buffer_;
}

}  // namespace alu
