#include <cmath>

#include "arithmetic/mathematic.h"

namespace alu {
namespace math {

double GaussianDistribution(const double& x, const double& mean, const double& std) {
  const double const_coefficient = 1.0 / std::sqrt(2 * ALU_PI * std * std);
  double gxy = const_coefficient * std::exp(-1 * (x - mean) * (x - mean) / (2 * std * std));
  return gxy;
}

double GaussianDistribution2D(const double& x, const double& y, const double& std) {
  const double const_coefficient = 1.0 / (2 * ALU_PI * std * std);
  double gxy = const_coefficient * std::exp(-1 * (x * x + y * y) / (2 * std * std));
  return gxy;
}

}  // namespace math
}  // namespace alu


