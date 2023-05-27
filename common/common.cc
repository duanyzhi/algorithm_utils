#include "interface/common.h"

namespace alu {

std::string Status::tostring() const {
  if (!this->state_) {
    if ("" != status_code_)
      return "code NG info: " + status_code_;
    return "NG";
  }
  return "OK";
}

}  // namespace Alu
