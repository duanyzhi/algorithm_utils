#ifndef ALU_CORE_TENSOR_H_
#define ALU_CORE_TENSOR_H_

#include <cassert>
#include <stdlib.h>
#include <iostream>
#include <type_traits>

namespace alu {

void *aligned_malloc(size_t size_bytes, size_t alignment = 8);
void aligned_free(void *data_ptr);

struct TensorBuffer {
  TensorBuffer(void *data_ptr) : data_(data_ptr) {}
  // TensorBuffer deconstruct must be virtual for auto call DataStorage deconstruct when doing delete.
  virtual ~TensorBuffer() {}
  void *data() const { return data_; }

private:
  void *const data_;
};

class TypedAllocator {
public:
  template <typename T> static T *Allocate(size_t num_elements) {
    void *p = aligned_malloc(num_elements * sizeof(T));
    T *data_ptr = reinterpret_cast<T *>(p);
    return data_ptr;
  }
  template <typename T> static void Deallocate(T *ptr) {
    if (ptr) {
      aligned_free(ptr);
    }
  }
};

template <typename _Tp> struct DataStorage : public TensorBuffer {
  DataStorage(size_t N);
  ~DataStorage();
  typedef _Tp value_type;
  typedef _Tp *pointer;
  pointer data_pointer;

private:
  size_t elem_;
};

template <typename _Tp>
DataStorage<_Tp>::DataStorage(size_t N)
    : TensorBuffer(TypedAllocator::Allocate<_Tp>(N)), elem_(N) {
}

template <typename _Tp> DataStorage<_Tp>::~DataStorage() {
  if (data()) {
    TypedAllocator::Deallocate<_Tp>(static_cast<_Tp *>(data()));
  }
}

enum class AluType : int8_t {
  ABOOL = 0,
  AINT = 1,
  ALINT = 2,
  AFLOAT = 3,
  ADOUBLE = 4
};

std::ostream& operator<<(std::ostream& os, const AluType& type);

#define _ALUTYPE_MAP(_)                                                        \
  _(alu::AluType::ABOOL, bool, Bool)                                           \
  _(alu::AluType::AINT, int, Int)                                              \
  _(alu::AluType::ALINT, int64_t, Long)                                        \
  _(alu::AluType::AFLOAT, float, Float)                                        \
  _(alu::AluType::ADOUBLE, double, Double)

template <typename To, typename From> To convert(From value) {
  return static_cast<To>(value); // not safe
}

struct Scalar {
public:
  Scalar() : Scalar(int64_t(0)) {}

#define DEFINE_IMPLICIT_CROR(ALUTYPE, TTYPE, NAME)                             \
  Scalar(TTYPE vv) : Scalar(vv, true) {}

  _ALUTYPE_MAP(DEFINE_IMPLICIT_CROR)

#undef DEFINE_IMPLICIT_CROR

#define DEFINE_GET_DATA(ALUTYPE, TTYPE, NAME)                                  \
  TTYPE to##NAME() const {                                                     \
    if (AluType::ALINT == type_) {                                             \
      return convert<TTYPE, int64_t>(v.i);                                     \
    } else if (AluType::ADOUBLE == type_) {                                    \
      return convert<TTYPE, double>(v.d);                                      \
    } else if (AluType::ABOOL == type_) {                                      \
      return convert<TTYPE, bool>(v.b);                                        \
    } else {                                                                   \
      return convert<TTYPE, int64_t>(v.i);                                     \
    }                                                                          \
  }

  _ALUTYPE_MAP(DEFINE_GET_DATA)

  template <typename T> T to() const = delete;
#undef DEFINE_GET_DATA

  AluType type() const { return type_; }

private:
  template <typename T,
            typename std::enable_if<std::is_integral<T>::value &&
                                        !std::is_same<T, bool>::value,
                                    bool>::type * = nullptr>
  Scalar(T vv, bool) : type_(AluType::ALINT) {
    v.i = static_cast<decltype(v.i)>(vv);
  }

  template <typename T,
            typename std::enable_if<!std::is_integral<T>::value &&
                                        !std::is_same<T, bool>::value,
                                    bool>::type * = nullptr>
  Scalar(T vv, bool) : type_(AluType::ADOUBLE) {
    v.d = static_cast<decltype(v.d)>(vv);
  }

  template <typename T, typename std::enable_if<std::is_same<T, bool>::value,
                                                bool>::type * = nullptr>
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

#define DEFINE_TO(ALUT, T, name)                                               \
  template <> inline T Scalar::to<T>() const { return to##name(); }
_ALUTYPE_MAP(DEFINE_TO)
#undef DEFINE_TO

std::ostream& operator<<(std::ostream& os, const Scalar& s);

static inline Scalar operator+(const Scalar& lhs, const Scalar& rhs) {
  if (lhs.type() == AluType::ABOOL || rhs.type() == AluType::ABOOL) {
    return lhs;  
  }
  if (lhs.type() == AluType::ALINT) {
    auto v = lhs.to<int64_t>() + rhs.to<int64_t>();
    return Scalar(v);
  } else if (lhs.type() == AluType::ADOUBLE) {
    auto v = lhs.to<double>() + rhs.to<double>();
    return Scalar(v);
  }
  return Scalar();
}

static inline Scalar operator*(const Scalar& lhs, const Scalar& rhs) {
  assert(lhs.type() != AluType::ABOOL && rhs.type() != AluType::ABOOL);
  if (lhs.type() == AluType::ALINT) {
    auto v = lhs.to<int64_t>() * rhs.to<int64_t>();
    return Scalar(v);
  } else if (lhs.type() == AluType::ADOUBLE) {
    auto v = lhs.to<double>() * rhs.to<double>();
    return Scalar(v);
  }
  return Scalar();
}

struct TensorInfo {
  int width;
  int height;
  size_t numel;
  AluType type;
};

template <AluType _Tp> struct alutype_traits {};

template <AluType _Tp> struct scalar_traits {
  typedef alutype_traits<_Tp> traits;
  typedef typename traits::type value_type;
  typedef typename traits::type *pointer;
};

class TensorImpl {
public:
  TensorImpl(TensorBuffer *buffer, int width, int height, AluType type);
  ~TensorImpl();

  virtual void fill(const Scalar &value) = 0;

  virtual void set(const int& index, const Scalar &value) = 0;

  void *data_ptr() const;

  size_t numel() const { return info_.numel; }

  const TensorInfo info() const { return info_; }

  virtual Scalar data(int index) = 0;

private:
  TensorInfo info_;
  TensorBuffer *buffer_;
};

template <AluType _AluTp> class TensorBase : public TensorImpl {
public:
  typedef typename scalar_traits<_AluTp>::value_type scalar_type;
  typedef typename scalar_traits<_AluTp>::pointer pointer;

  TensorBase(int width, int height)
      : TensorImpl(new DataStorage<scalar_type>(width * height), width,
                   height, _AluTp) {
   }

  ~TensorBase(){};
  void fill(const Scalar &value) override;
  void set(const int& index, const Scalar &value) override;
  Scalar data(int index) override;
};

template <AluType _AluType>
void TensorBase<_AluType>::fill(const Scalar &value) {
  pointer buffer = static_cast<pointer>(data_ptr());
  scalar_type data = value.to<scalar_type>();
  for (auto size = 0; size < numel(); ++size) {
    *buffer = data;
    buffer++;
  }
}

template <AluType _AluType>
void TensorBase<_AluType>::set(const int& index, const Scalar &value) {
  pointer buffer = static_cast<pointer>(data_ptr());
  scalar_type data = value.to<scalar_type>();
  *(buffer + index) = data;
}

template <AluType _AluType>
Scalar TensorBase<_AluType>::data(int index) {
  pointer buffer = static_cast<pointer>(data_ptr());
  auto value = static_cast<scalar_type>(*(buffer + index));
  Scalar s(value);
  return s;
}

class Tensor {
public:
  Tensor(const int &width, const int &height, AluType type);
  Tensor(const int &width, const int &height) : Tensor(width, height, AluType::ADOUBLE) {};
  ~Tensor();
  // const TensorBuffer* GetTensorBuffer() const;
  const TensorInfo info() const;
  const Tensor &fill(const Scalar &value) const;
  const Tensor &set(const int& index, const Scalar &value) const;

  void *data() const;

  // common index = w_index * width + h_index;
  Scalar operator[](int index) const;

  AluType dtype() const {
    return info().type;
  }

private:
  TensorImpl *impl_;
};

std::ostream& operator<<(std::ostream& os, const Tensor& t);

} // namespace alu

#endif // ALU_CORE_TENSOR_H_
