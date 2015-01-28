#include <mlpack/core/dists/gaussian_distribution.hpp>

using namespace mlpack::distribution;
using namespace std;
using namespace arma;


const double EPS = 1e-10;
double sigma, lambda, noise_sigma;

inline double square(const double x) {
  return x * x;
}

double kernel(double x,double xp) {
  return square(sigma) * exp(- square(x - xp) / (2 * square(lambda)));
}

int main(int argc, char **argv) {

  if (argc < 6) {
    cout << "Usage " << argv[0] << " input_set output_set sigma lambda noise_sigma" << endl;
    exit(1);
  }
  arma_rng::set_seed_random();
  sigma  = atof(argv[3]);
  lambda = atof(argv[4]);
  noise_sigma = atof(argv[5]);

  vec x, y;
  x.load(argv[1], raw_ascii);
  y.load(argv[2], raw_ascii);

  vec mean(x.size(), fill::zeros);
  mat cov(x.size(), x.size());

  for (int i = 0; i < x.size(); i++)
    for (int j = 0; j < x.size(); j++)
      cov(i, j) = kernel(x(i), x(j));


  GaussianDistribution prior(mean, cov + noise_sigma * eye<mat>(cov.size(),cov.size()));

  for (int i = 0; i < 3; ++i) {
    vec data = prior.Random();
    string name = "y" + to_string(i) + ".mio";
    data.save(name, raw_ascii);
  }

  mat new_y;

  new_y.load("new_y.mio", raw_ascii);

  mat A(new_y.n_cols , x.size());
  mat B = cov;
  mat f(y);
  mat C(new_y.n_cols, new_y.n_cols);

  for(int i = 0; i < new_y.n_cols; i ++){
    for(int j = 0; j < x.size(); j++){
      A(i, j) = kernel(new_y(0, i), x(j));
    }
  }

  for(int i = 0; i < new_y.n_cols; i++){
    for(int j = 0; j < new_y.n_cols; j++)
      C(i, j) = kernel(new_y(0, i), new_y(0, j));
  }

  mat estimate = A * B.i() * f;
  mat uncertainty  = C - A * B.i() * A.t();

  GaussianDistribution posterior (estimate, uncertainty);
  vec test = posterior.Mean();
  test.save("outval.mio", raw_ascii);

  return 0;
}
