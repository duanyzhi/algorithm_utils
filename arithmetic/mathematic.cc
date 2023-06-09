#include <cmath>

#include "arithmetic/mathematic.h"

namespace alu {
namespace math {

double GaussianDistribution(const double &x, const double &mean,
                            const double &std) {
  const double const_coefficient = 1.0 / std::sqrt(2 * ALU_PI * std * std);
  double gxy = const_coefficient *
               std::exp(-1 * (x - mean) * (x - mean) / (2 * std * std));
  return gxy;
}

double GaussianDistribution2D(const double &x, const double &y,
                              const double &std) {
  const double const_coefficient = 1.0 / (2 * ALU_PI * std * std);
  double gxy =
      const_coefficient * std::exp(-1 * (x * x + y * y) / (2 * std * std));
  return gxy;
}

constexpr double EuclideanDistance(const double &x, const double &y) {
  return std::sqrt(x * x + y * y);
}

constexpr double EuclideanDistance(const double &x1, const double &y1,
                                   const double &x2, const double &y2) {
  return std::sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2));
}

constexpr double Rad2Angle(const double &radian) {
  return radian * 180. / ALU_PI;
}

constexpr double Angle2Rad(const double &angle) {
  return angle * ALU_PI / 180.0;
}

} // namespace math
} // namespace alu
