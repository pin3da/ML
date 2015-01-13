#include <mlpack/core/dists/gaussian_distribution.hpp>

using namespace mlpack::distribution;
using namespace std;
using namespace arma;

int main(int argc, char **argv) {

  if (argc < 3) {
    cout << "Usage " << argv[0] << " n_points max_input [noise]" << endl;
    exit(1);
  }

  arma_rng::set_seed_random();
  vec mean({0});

  double tmp = 0.001;
  if (argc > 3)
    tmp = atof(argv[3]);

  mat cov({tmp});

  cov.print();
  GaussianDistribution dist(mean, cov);
  cout << dist.ToString() << endl;
  vec noise = dist.Random();

  ofstream x("x.mio");
  ofstream y("y.mio");

  const double delta = acos(0) / atoi(argv[1]);
  int max_value = atoi(argv[2]);
  for (double i = 0; i < max_value; i += delta) {
    noise = dist.Random();
    x << i << ' ';
    y << sin(i) + noise[0] << ' ';
  }

  x.close();
  y.close();

  noise.print();

  cout << dist.Probability(noise) << endl;

  return 0;
}
