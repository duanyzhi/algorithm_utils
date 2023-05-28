#ifndef ALU_CORE_TENSOR_H_
#define ALU_CORE_TENSOR_H_

#include <stdlib.h>

namespace alu {

void* aligned_malloc(size_t size, size_t alignment = 16);
void aligned_free(void* data_ptr);

struct TensorBuffer {
  TensorBuffer(void* data_ptr) : data_(data_ptr) {}
  void* data() const { return data_; }

 private:
  void* const data_;
};

template <typename _Tp, size_t N>
struct DataStorage : public TensorBuffer {
  DataStorage();
  ~DataStorage();
  typedef _Tp value_type;
  typedef _Tp* pointer;
  pointer data;
 private:
  size_t elem_;
};

class TypedAllocator {
 public:
  template <typename T>
  static T* Allocate(size_t num_elements) {
    void* p = aligned_malloc(num_elements * sizeof(T));
    T* data_ptr = reinterpret_cast<T*>(p);
    return data_ptr;
  }
  template <typename T>
  static void Deallocate(T* ptr) {
    if (ptr) {
      aligned_free(ptr);
    }
  }
};

enum AluType {
  AINT = 0,
  AFLOAT = 1
};

#define _ALUTYPE_MAP(_)         \
  _(alu::AluType::AINT, int)        \
  _(alu::AluType::AFLOAT, float)

class Tensor {
 public:
  Tensor(const int& width, const int& height, int type);
  
 private:
  TensorBuffer* buffer_;  
};

}

#endif  // ALU_CORE_TENSOR_H_
