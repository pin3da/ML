#include <mlpack/core/dists/gaussian_distribution.hpp>

using namespace mlpack::distribution;
using namespace std;
using namespace arma;

int main(int argc, char **argv) {

  arma_rng::set_seed_random();
  vec mean({0, 0});
  mat cov({1, 2, 2, 5});
  cov.reshape(2, 2);

  cov.print();
  GaussianDistribution dist(mean, cov);
  cout << dist.ToString() << endl;
  vec data = dist.Random();

  ofstream x("x.mio");
  ofstream y("y.mio");

  int iter = atoi(argv[1]);
  for (int i = 0; i < iter; ++i) {
    data = dist.Random();
    x << data[0] << ' ';
    y << data[1] << ' ';
  }

  x.close();
  y.close();

  data.print();

  cout << dist.Probability(data) << endl;

  return 0;
}
