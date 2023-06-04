#include "interface/common.h"
#include <sstream>

namespace alu {

std::string Status::tostring() const {
  if (!this->state_) {
    if ("" != status_code_)
      return "code NG info: " + status_code_;
    return "NG";
  }
  return "OK";
}

std::ostream &operator<<(std::ostream &os, const rect &t) {
  std::stringstream ss;
  ss << "[x, y, w, h]: " << t.x << ", " << t.y << ", " << t.w << ", " << t.h
     << ".\n";
  os << ss.str();
  return os;
}

} // namespace alu
