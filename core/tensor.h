#ifndef ALU_CORE_TENSOR_H_
#define ALU_CORE_TENSOR_H_

#include <stdlib.h>
#include <type_traits>

namespace alu {

void* aligned_malloc(size_t size_bytes, size_t alignment = 16);
void aligned_free(void* data_ptr);

struct TensorBuffer {
  TensorBuffer(void* data_ptr) : data_(data_ptr) {}
  void* data() const { return data_; }

 private:
  void* const data_;
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

template <typename _Tp>
struct DataStorage : public TensorBuffer {
  DataStorage(size_t N);
  ~DataStorage();
  typedef _Tp value_type;
  typedef _Tp* pointer;
  pointer data;
 private:
  size_t elem_;
};

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

enum class AluType : int8_t {
  ABOOL,
  AINT,
  ALINT,
  AFLOAT,
  ADOUBLE
};

#define _ALUTYPE_MAP(_)                      \
  _(alu::AluType::ABOOL, bool, Bool)       \
  _(alu::AluType::AINT, int, Int)          \
  _(alu::AluType::ALINT, int64_t, Long)   \
  _(alu::AluType::AFLOAT, float, Float)    \
  _(alu::AluType::ADOUBLE, double, Double)

template <typename To, typename From>
To convert(From value) {
  return static_cast<To>(value);  // not safe
}

struct Scalar {
 public:
  Scalar() : Scalar(int64_t(0)) {}

#define DEFINE_IMPLICIT_CROR(ALUTYPE, TTYPE, NAME)  \
  Scalar(TTYPE vv) : Scalar(vv, true) {}

  _ALUTYPE_MAP(DEFINE_IMPLICIT_CROR)

#undef DEFINE_IMPLICIT_CROR

#define DEFINE_GET_DATA(ALUTYPE, TTYPE, NAME)  \
  TTYPE to##NAME() const {                     \
    if (AluType::ALINT == type_) {             \
      return convert<TTYPE, int64_t>(v.i);     \
    } else if (AluType::ADOUBLE == type_) {             \
      return convert<TTYPE, double>(v.d);      \
    } else if (AluType::ABOOL == type_) {               \
      return convert<TTYPE, bool>(v.b);        \
    } else {                                   \
      return convert<TTYPE, int64_t>(v.i);     \
    }                                          \
  }

  _ALUTYPE_MAP(DEFINE_GET_DATA)

  template <typename T>
  T to() const = delete;
#undef DEFINE_GET_DATA

 private:
  template <
       typename T,
       typename std::enable_if<
       std::is_integral<T>::value && !std::is_same<T, bool>::value,
       bool>::type* = nullptr>
  Scalar(T vv, bool) : type_(AluType::ALINT) {
    v.i = static_cast<decltype(v.i)>(vv);
  }

  template <
       typename T,
       typename std::enable_if<
       !std::is_integral<T>::value && !std::is_same<T, bool>::value,
       bool>::type* = nullptr>
  Scalar(T vv, bool) : type_(AluType::ADOUBLE) {
    v.d = static_cast<decltype(v.d)>(vv);
  }

  template <
       typename T,
       typename std::enable_if<std::is_same<T, bool>::value, bool>::type* = nullptr>
  Scalar(T vv, bool) : type_(AluType::ABOOL) {
    v.b = static_cast<decltype(v.b)>(vv);
  }

  AluType type_; 
  union value {
    bool b;
    double d;
    int64_t i;
  } v;
};

#define DEFINE_TO(ALUT, T, name)   \
  template <>                      \
  inline T Scalar::to<T>() const { \
    return to##name();             \
  }
_ALUTYPE_MAP(DEFINE_TO)
#undef DEFINE_TO

struct TensorInfo {
  int width;
  int height; 
  size_t numel;
  AluType type;
};


template <AluType _Tp>
struct alutype_traits {};

template <AluType _Tp>
struct scalar_traits {
  typedef alutype_traits<_Tp> traits;
  typedef typename traits::type value_type;
  typedef typename traits::type* pointer;
};

class TensorImpl {
 public:
  TensorImpl(TensorBuffer* buffer, int width, int height);
  ~TensorImpl();

  virtual void fill(const Scalar& value) = 0;

  void* data_ptr() const;

  size_t numel() const {
    return info_.numel;
  }

  const TensorInfo info() const {
    return info_;
  }

 private:
  TensorInfo  info_;
  TensorBuffer* buffer_;  
};

template <AluType _AluTp>
class TensorBase : public TensorImpl {
 public:
  typedef typename scalar_traits<_AluTp>::value_type scalar_type;
  typedef typename scalar_traits<_AluTp>::pointer pointer;

  TensorBase(int width, int height)
    : TensorImpl(new DataStorage<scalar_type>(width * height), width, height) {}

  ~TensorBase() {};
  void fill(const Scalar& value) override;
};

template <AluType _AluType>
void TensorBase<_AluType>::fill(const Scalar& value) {
  pointer buffer = static_cast<pointer>(data_ptr());
  scalar_type data = value.to<scalar_type>(); 
  for (auto size = 0; size < numel(); ++size) {
    *buffer = data;
    buffer++;
  }
}

class Tensor {
 public:
  Tensor(const int& width, const int& height, AluType type);
  ~Tensor();
  // const TensorBuffer* GetTensorBuffer() const;
  const TensorInfo info() const;
  const Tensor& fill(const Scalar& value) const;

  void* data() const;

 private:
  // TensorInfo  info_;
  // TensorBuffer* buffer_;  
  TensorImpl* impl_;
};

}

#endif  // ALU_CORE_TENSOR_H_
