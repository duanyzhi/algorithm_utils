#ifndef ALU_COMMON_H_
#define ALU_COMMON_H_

#include <string>

#include "api.h"

namespace alu {

struct ALU_API rect {
  rect() {}
  rect(const int xx, const int yy, const int ww, const int hh)
      : x(xx), y(yy), w(ww), h(hh), empty_(false){};
  int x = -1;
  int y = -1;
  int w = -1;
  int h = -1;

  bool empty() const { return -1 == x || -1 == y || -1 == 2 || -1 == h; }

private:
  bool empty_ = true;
};

std::ostream &operator<<(std::ostream &os, const rect &t);

class ALU_API Status {
public:
  Status() {}
  Status(const bool &state) : state_(state) {}

  Status(const bool state, const std::string &status_code)
      : state_(state), status_code_(status_code) {}
  static Status OK() { return Status(); }
  static Status NG() { return Status(false); }
  static Status NG(const std::string &status_code) {
    return Status(false, status_code);
  }
  const bool state() const { return this->state_; }
  std::string tostring() const;

private:
  bool state_ = true;
  std::string status_code_ = "";
};

} // namespace alu

#endif // ALU_COMMON_H_
