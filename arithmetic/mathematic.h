#ifndef ARITHMETIC_MATHEMATIC_H_
#define ARITHMETIC_MATHEMATIC_H_

namespace alu {
namespace math {

#define ALU_PI 3.14159265358979323846 /* pi */

double GaussianDistribution(const double &x, const double &mean,
                            const double &std);

/***
 *  2 dim gaussian distribution, mean default is 0 and
 *  dx = dy
 ***/
double GaussianDistribution2D(const double &x, const double &y,
                              const double &std);

constexpr double EuclideanDistance(const double &x, const double &y);
constexpr double EuclideanDistance(const double &x1, const double &y1,
                                   const double &x2, const double &y2);

double ManhattanDistance(const double &x, const double &y);

constexpr double Rad2Angle(const double &radian);
constexpr double Angle2Rad(const double &angle);

} // namespace math
} // namespace alu

#endif // ARITHMETIC_MATHEMATIC_H_
