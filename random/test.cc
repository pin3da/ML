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

  int max_value = atoi(argv[2]);
  int iter = atoi(argv[1]);
  const double delta = max_value / double(iter);

  double j = delta;
  vec x(iter), y(iter);
  for (int i = 0; i < iter; ++i, j += delta) {
    noise = dist.Random();
    x(i) = j;
    y(i) = sin(j);
  }

  x.save("x.mio", raw_ascii);
  y.save("y.mio", raw_ascii);

  noise.print();

  cout << dist.Probability(noise) << endl;

  return 0;
}
