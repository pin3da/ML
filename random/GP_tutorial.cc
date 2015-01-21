#include <mlpack/core/dists/gaussian_distribution.hpp>

using namespace mlpack::distribution;
using namespace std;
using namespace arma;


const double EPS = 1e-10;
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
  sigma  = atof(argv[3]);
  lambda = atof(argv[4]);

  vec x, y;
  x.load(argv[1], raw_ascii);
  y.load(argv[2], raw_ascii);

  vec mean(x.size(), fill::zeros);
  mat cov(x.size(), x.size());

  for (int i = 0; i < x.size(); i++)
    for (int j = 0; j < x.size(); j++)
      cov(i, j) = kernel(x(i), x(j));


  GaussianDistribution prior(mean, cov);

  for (int i = 0; i < 3; ++i) {
    vec data = prior.Random();
    string name = "y" + to_string(i) + ".mio";
    data.save(name, raw_ascii);
  }


  mat A(1, x.size());
  mat B = cov;
  mat f(y);
  mat C(1, 1);


  ofstream inval("inval.mio");
  double n_val = 0;
  for (; n_val < 8.0; n_val += 0.4) {
    for (int i = 0; i < x.size(); ++i)
      A(0, i) = kernel(n_val, x(i));

    C(0, 0) = kernel(n_val, n_val);

    mat estimate = A * B.i() * f;
    mat uncertainty  = C - A * B.i() * A.t();
    inval << n_val << endl;
    // estimate.print();
    uncertainty(0, 0) += EPS;
    GaussianDistribution posterior(estimate, uncertainty);
    vec data = posterior.Random();
    cout << setprecision(12) << data(0) << endl;
  }

  inval.close();

  return 0;
}
