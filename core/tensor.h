#ifndef ALU_CORE_TENSOR_H_
#define ALU_CORE_TENSOR_H_

#include <cassert>
#include <iostream>
#include <stdlib.h>
#include <type_traits>

#include "interface/common.h"

namespace alu {

void *aligned_malloc(size_t size_bytes, size_t alignment = 8);
void aligned_free(void *data_ptr);

struct TensorBuffer {
  TensorBuffer(void *data_ptr) : data_(data_ptr) {}
  // TensorBuffer deconstruct must be virtual for auto call DataStorage
  // deconstruct when doing delete.
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
    : TensorBuffer(TypedAllocator::Allocate<_Tp>(N)), elem_(N) {}

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

std::ostream &operator<<(std::ostream &os, const AluType &type);

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

std::ostream &operator<<(std::ostream &os, const Scalar &s);

static inline Scalar operator+(const Scalar &lhs, const Scalar &rhs) {
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

static inline Scalar operator*(const Scalar &lhs, const Scalar &rhs) {
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

static inline bool operator>(const Scalar &lhs, const Scalar &rhs) {
  assert(lhs.type() != AluType::ABOOL && rhs.type() != AluType::ABOOL);
  if (lhs.type() == AluType::ALINT) {
    return lhs.to<int64_t>() > rhs.to<int64_t>();
  } else if (lhs.type() == AluType::ADOUBLE) {
    return lhs.to<double>() > rhs.to<double>();
  } else {
    assert(false);
  }
  return false;
}

static inline bool operator>=(const Scalar &lhs, const Scalar &rhs) {
  assert(lhs.type() != AluType::ABOOL && rhs.type() != AluType::ABOOL);
  if (lhs.type() == AluType::ALINT) {
    return lhs.to<int64_t>() >= rhs.to<int64_t>();
  } else if (lhs.type() == AluType::ADOUBLE) {
    return lhs.to<double>() >= rhs.to<double>();
  } else {
    assert(false);
  }
  return false;
}

static inline bool operator<(const Scalar &lhs, const Scalar &rhs) {
  assert(lhs.type() != AluType::ABOOL && rhs.type() != AluType::ABOOL);
  if (lhs.type() == AluType::ALINT) {
    return lhs.to<int64_t>() < rhs.to<int64_t>();
  } else if (lhs.type() == AluType::ADOUBLE) {
    return lhs.to<double>() < rhs.to<double>();
  } else {
    assert(false);
  }
  return false;
}

static inline bool operator<=(const Scalar &lhs, const Scalar &rhs) {
  assert(lhs.type() != AluType::ABOOL && rhs.type() != AluType::ABOOL);
  if (lhs.type() == AluType::ALINT) {
    return lhs.to<int64_t>() <= rhs.to<int64_t>();
  } else if (lhs.type() == AluType::ADOUBLE) {
    return lhs.to<double>() <= rhs.to<double>();
  } else {
    assert(false);
  }
  return false;
}

Scalar max(const Scalar &lhs, const Scalar &rhs);

struct TensorInfo {
  int width = -1;
  int height = -1;
  size_t numel = -1;
  size_t size_bytes = -1;
  AluType type = AluType::ADOUBLE;
};

template <AluType _Tp> struct alutype_traits {};

template <AluType _Tp> struct scalar_traits {
  typedef alutype_traits<_Tp> traits;
  typedef typename traits::type value_type;
  typedef typename traits::type *pointer;
};

class TensorImpl {
public:
  TensorImpl(TensorBuffer *buffer, int width, int height, AluType type,
             size_t size_bytes);

  ~TensorImpl();
  virtual void fill(const Scalar &value) = 0;

  virtual void set(const int &index, const Scalar &value) noexcept = 0;
  virtual void add(const void *other, const void *output) = 0;
  virtual void mul(const void *other, const void *output) = 0;
  virtual void mul(const double value, const void *output) = 0;
  virtual void div(const double value, const void *output) = 0;
  virtual void roi(const alu::rect &roi, const void *output) = 0;
  virtual void abs(const void *output) = 0;

  void *data_ptr() const;

  size_t numel() const { return info_.numel; }
  size_t numel_bytes() const { return info_.numel * info_.size_bytes; }

  const TensorInfo info() const { return info_; }

  virtual Scalar data(int index) = 0;

private:
  TensorInfo info_;
  TensorBuffer *buffer_;
};

template <AluType _AluTp> class TensorBase : public TensorImpl {
public:
  typedef typename scalar_traits<_AluTp>::traits traits;
  typedef typename scalar_traits<_AluTp>::value_type scalar_type;
  typedef typename scalar_traits<_AluTp>::pointer pointer;

  TensorBase(int width, int height)
      : TensorImpl(new DataStorage<scalar_type>(width * height), width, height,
                   _AluTp, traits::bytes) {}

  ~TensorBase(){};
  void fill(const Scalar &value) override;
  void set(const int &index, const Scalar &value) noexcept override;
  void add(const void *other, const void *output) override;
  void mul(const void *other, const void *output) override;
  void mul(const double value, const void *output) override;
  void div(const double value, const void *output) override;
  void roi(const alu::rect &roi, const void *output) override;
  void abs(const void *output) override;
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
void TensorBase<_AluType>::roi(const alu::rect &roi, const void *output) {
  pointer buffer = static_cast<pointer>(data_ptr());
  pointer output_buffer = static_cast<pointer>((void *)output);
  int number = 0;
  for (auto y = 0; y < roi.h; ++y) {
    for (auto x = 0; x < roi.w; ++x) {
      auto index = y * info().width + x;
      *(output_buffer + number) = static_cast<scalar_type>(*(buffer + index));
      ++number;
    }
  }
}

template <AluType _AluType>
void TensorBase<_AluType>::set(const int &index, const Scalar &value) noexcept {
  pointer buffer = static_cast<pointer>(data_ptr());
  scalar_type data = value.to<scalar_type>();
  *(buffer + index) = data;
}

template <AluType _AluType> Scalar TensorBase<_AluType>::data(int index) {
  pointer buffer = static_cast<pointer>(data_ptr());
  auto value = static_cast<scalar_type>(*(buffer + index));
  Scalar s(value);
  return s;
}

template <AluType _Alutype>
void TensorBase<_Alutype>::mul(const void *other, const void *output) {
  pointer src_buffer = static_cast<pointer>(data_ptr());
  pointer other_buffer = static_cast<pointer>((void *)other);
  pointer output_buffer = static_cast<pointer>((void *)output);
  for (int i = 0; i < numel(); i++) {
    *(output_buffer + i) = static_cast<scalar_type>(*(src_buffer + i)) *
                           static_cast<scalar_type>(*(other_buffer + i));
  }
}

template <AluType _Alutype>
void TensorBase<_Alutype>::mul(const double value, const void *output) {
  pointer src_buffer = static_cast<pointer>(data_ptr());
  pointer output_buffer = static_cast<pointer>((void *)output);
  for (int i = 0; i < numel(); i++) {
    *(output_buffer + i) = static_cast<scalar_type>(*(src_buffer + i)) * value;
  }
}

template <AluType _AluType>
void TensorBase<_AluType>::div(const double value, const void *output) {
  pointer src_buffer = static_cast<pointer>(data_ptr());
  pointer output_buffer = static_cast<pointer>((void *)output);
  for (int i = 0; i < numel(); i++) {
    *(output_buffer + i) = static_cast<scalar_type>(*(src_buffer + i)) / value;
  }
}

template <AluType _AluType>
void TensorBase<_AluType>::add(const void *other, const void *output) {
  pointer src_buffer = static_cast<pointer>(data_ptr());
  pointer other_buffer = static_cast<pointer>((void *)other);
  pointer output_buffer = static_cast<pointer>((void *)output);
  for (int i = 0; i < numel(); i++) {
    *(output_buffer + i) = static_cast<scalar_type>(*(src_buffer + i)) +
                           static_cast<scalar_type>(*(other_buffer + i));
  }
}

template <AluType _AluType> void TensorBase<_AluType>::abs(const void *output) {
  pointer src_buffer = static_cast<pointer>(data_ptr());
  pointer output_buffer = static_cast<pointer>((void *)output);
  for (int i = 0; i < numel(); i++) {
    auto v = static_cast<scalar_type>(*(src_buffer + i));
    *(output_buffer + i) = v < 0 ? -1 * v : v;
  }
}

class Tensor {
public:
  Tensor() = default;
  Tensor(const int &width, const int &height, AluType type);
  Tensor(const TensorInfo &info) : Tensor(info.width, info.height, info.type){};
  Tensor(const int &width, const int &height)
      : Tensor(width, height, AluType::ADOUBLE){};
  ~Tensor();
  Tensor &operator=(const Tensor &rhs);
  // const TensorBuffer* GetTensorBuffer() const;
  const TensorInfo info() const;
  const Tensor &fill(const Scalar &value) const;
  const Tensor &set(const int &index, const Scalar &value) const;
  const Tensor &set(int row, int col, const Scalar &value) const;
  const Tensor mul(const Tensor &other);
  Tensor abs();
  void *data() const;

  Scalar data(int row, int col) const;

  Scalar operator[](int index) const;
  Tensor operator+(const Tensor &other) const;
  Tensor operator/(const double value) const;
  Tensor operator*(const double value) const;
  Tensor operator()(const alu::rect &roi) const;

  AluType dtype() const { return info().type; }
  bool empty() const { return nullptr == impl_; }
  TensorImpl *impl() const { return impl_; }

private:
  Tensor(TensorImpl *impl) : impl_(impl) {}

private:
  TensorImpl *impl_ = nullptr;
};

std::ostream &operator<<(std::ostream &os, const Tensor &t);

Tensor arctan(const Tensor &a, const Tensor &b);
Tensor sqrt(const Tensor &input);

} // namespace alu

#endif // ALU_CORE_TENSOR_H_
