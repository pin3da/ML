#include <mlpack/core/dists/gaussian_distribution.hpp>

using namespace mlpack::distribution;
using namespace std;
using namespace arma;

double sigma, lambda;

inline double square(const double x) {
  return x * x;
}

double kernel(double x,double xp) {
  return square(sigma) * exp(- square(x - xp) / (2 * square(lambda)));
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

  vec mean(x.size() - 1, fill::zeros);
  mat cov(x.size() - 1, x.size() - 1);

  for (int i = 0; i < x.size() - 1; i++)
    for (int j = 0; j < x.size() - 1; j++)
      cov(i, j) = kernel(x(i), x(j));


  GaussianDistribution prior(mean, cov);

  for (int i = 0; i < 3; ++i) {
    vec data = prior.Random();
    string name = "y" + to_string(i) + ".mio";
    data.save(name, raw_ascii);
  }


  mat A(1, x.size() - 1);
  mat B = cov;
  mat f(y);
  mat C(1, 1);

  for (int i = 0; i < x.size() - 1; ++i)
    A(0, i) = kernel(x(x.size() - 1), x(i));

  C(0, 0) = kernel(x(x.size() - 1), x(x.size() - 1));

  mat post_mean = A * B.i() * f;
  mat post_cov  = C - A * B.i() * A.t();

  post_mean.print();


  return 0;
}
