#ifndef ALU_COMMON_LOGGING_H_
#define ALU_COMMON_LOGGING_H_

#include <iostream>

namespace alu {

#define LOG(...)  std::cout << __VA_ARGS__ << std::endl;

}  // namespace alu

#endif  // ALU_COMMON_LOGGING_H_
