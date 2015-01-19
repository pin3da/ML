#include <mlpack/core/dists/gaussian_distribution.hpp>

using namespace mlpack::distribution;
using namespace std;
using namespace arma;

double sigma, lambda;

inline double square(const double x) {
  return x * x;
}

double kernel(double x,double xp) {
  return square(x) * exp(- square(x - xp) / (2 * square(lambda)));
}

int main(int argc, char **argv) {

  if (argc < 5) {
    cout << "Usage " << argv[0] << " input_set output_set sigma lambda" << endl;
    exit(1);
  }
  arma_rng::set_seed_random();
  sigma = atof(argv[3]);
  lambda = atof(argv[4]);

  vec x, y;
  x.load(argv[1], raw_ascii);
  y.load(argv[2], raw_ascii);

  vec mean(x.size(), fill::zeros);
  mat cov(x.size(), x.size());

  for (int i = 0; i < x.size(); i++)
    for (int j = 0; j < x.size(); j++)
      cov(i, j) = kernel(x(i), x(j));


  GaussianDistribution dist(mean, cov);

  for (int i = 0; i < 25; ++i) {
    vec data = dist.Random();
    string name = "y" + to_string(i) + ".mio";
    data.save(name, raw_ascii);

  }


  return 0;
}
