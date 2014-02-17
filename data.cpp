#include <bits/stdc++.h>
#include <random>
#include <armadillo>

using namespace std;
using namespace arma;

#include "linear_regression.cpp"


void gen_data() {
  int n;
  cin>>n;
  mat data(n,2);
  unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
  std::default_random_engine generator (seed);

  std::normal_distribution<double> distribution (0.0, 0.7);

  int m = 5, b = 3;
  for (int i = 0; i < n; ++i) {
    data(i,0) = i*5;
    data(i,1) = m*i*5 + b + distribution(generator);
  }
  data.save("data.mio");
}



int main(int argc, char **argv) {

  //gen_data();
  mat data;
  data.load("data.mio");

  mat X = join_horiz(ones(data.n_rows, 1), data.col(0));
  mat W = gradient_descent(X, data.col(1), 10000, 0.01);
  cout<<W<<endl;
  double err = sum( pow((X * W) - data.col(1), 2) );
  cout<<"Error = "<<err<<endl;

  /*** Maximum likelihood bellow ***

  double sigma = (1.0/(double)data.n_rows) * sum( pow((X * W) - data.col(1), 2) );
  double input;
  while (cin>>input and input) {
    double mean = W(0,0) + W(1,0) * input;
    // Predictive distribution.
    normal_distribution<double> distribution(mean, sigma);
    cout<<"Mean : " <<mean<<endl<<"Sigma : "<<sigma<<endl;
  }
  *** End of ML ***/
  return 0;
}
