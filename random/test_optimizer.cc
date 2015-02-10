#include <mlpack/core.hpp>
#include <mlpack/core/optimizers/aug_lagrangian/aug_lagrangian.hpp>
using namespace std;
using namespace mlpack;
using namespace mlpack::optimization;

class OptimizerTestFunction {
 public:
  OptimizerTestFunction() {
    initialPoint.zeros(1, 1);
  }
  OptimizerTestFunction(const arma::mat& initial_point) {
    initialPoint = initial_point;
  }

  double Evaluate(const arma::mat& coordinates) {
    double x = coordinates(0, 0);
    return 5.0 - (x - 3.0) * (x - 3.0);
  }

  void Gradient(const arma::mat& coordinates, arma::mat& gradient) {
    double x = coordinates(0, 0);
    gradient.set_size(1, 1);
    gradient(0, 0) = - 2.0 * (x - 3.0);
  }

  size_t NumConstraints() const {
    return 1;
  }

  double EvaluateConstraint(const size_t index, const arma::mat& coordinates) {
    if (index != 0)
      return 0;

    double x = min(0.0, coordinates(0, 0) - 2.0);
    return x;
  }

  void GradientConstraint(const size_t index, const arma::mat& coordinates, arma::mat& gradient) {
    if (index == 0) {
      gradient.ones(1, 1);
    }
  }

  const arma::mat& GetInitialPoint() const {
    return initialPoint;
  }

  // convert the obkect into a string
  std::string ToString() const {
    return "testing optimizer";
  }

 private:
  arma::mat initialPoint;
};

int main() {

  OptimizerTestFunction func;
  AugLagrangian<OptimizerTestFunction> aug(func);
  arma::vec coords = func.GetInitialPoint();
  if (aug.Optimize(coords, 0)) {
    double finalValue = func.Evaluate(coords);
    coords.print();
    cout << finalValue << endl;
  } else {
    cout << "something went wrong" << endl;
  }

  return 0;
}
