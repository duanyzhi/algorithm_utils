#ifndef ARITHMETIC_MATHEMATIC_H_
#define ARITHMETIC_MATHEMATIC_H_

namespace alu {
namespace math {

# define ALU_PI           3.14159265358979323846  /* pi */

double GaussianDistribution(const double& x, const double& mean, const double& std);


/***
 *  2 dim gaussian distribution, mean default is 0 and
 *  dx = dy
 ***/
double GaussianDistribution2D(const double& x, const double& y, const double& std);

}  // namespace math
}  // namespace alu

#endif  // ARITHMETIC_MATHEMATIC_H_
