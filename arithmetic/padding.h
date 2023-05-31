#ifndef ALU_ARITHMETIC_PADDING_H_
#define ALU_ARITHMETIC_PADDING_H_

#include "core/tensor.h"

namespace alu {

enum class AluPadType : int8_t {
  RawValue = 0;
  Const = 1;
};

Tensor pad(const Tensor& input);

}  //  namespace alu

#endif  // ALU_ARITHMETIC_PADDING_H_
