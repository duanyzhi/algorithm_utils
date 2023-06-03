#ifndef ALU_COMMON_H_
#define ALU_COMMON_H_

#include <string>

#include "api.h"

namespace alu {

struct ALU_API rect {
  rect() {}
  rect(const int xx, const int yy, const int ww, const int hh) :
    x(xx), y(yy), w(ww), h(hh), empty_(true) {};
  int x;
  int y;
  int w;
  int h;

  bool empty() const {
    return empty_;
  }
 private:
  bool empty_ = true;
};

class ALU_API Status {
 public:
  Status() {}
  Status(const bool& state): state_(state) {}

  Status(const bool state, const std::string& status_code)
      : state_(state), status_code_(status_code) {}
  static Status OK() { return Status(); }
  static Status NG() { return Status(false); }
  static Status NG(const std::string& status_code) {
    return Status(false, status_code);
  }
  const bool state() const { return this->state_; }
  std::string tostring() const;

 private:
  bool state_ = true;
  std::string status_code_ = "";
};

}  // namespace alu

#endif  // ALU_COMMON_H_
