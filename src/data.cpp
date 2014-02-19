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
  unsigned seed = chrono::system_clock::now().time_since_epoch().count();
  default_random_engine generator (seed);

  normal_distribution<double> distribution (0.0, 15);

  int m = 5, b = 3;
  for (int i = 0; i < n; ++i) {
    data(i,0) = i*5;
    data(i,1) = m*i*5 + b + distribution(generator);
  }
  ((mat) data.col(0)).save("xo.mio", raw_ascii);
  ((mat) data.col(1)).save("yo.mio", raw_ascii);
  data.save("data.mio", raw_ascii);
}


int main(int argc, char **argv) {
  gen_data();
  mat data;
  data.load("data.mio", raw_ascii);
  mat X = join_horiz(ones(data.n_rows, 1), data.col(0));


  // Show indicators
  clock_t begin = clock();
  mat W = gradient_descent(X, data.col(1), 10000, 0.01);
  clock_t end = clock();
  double err = sum( pow((X * W) - data.col(1), 2) );

  cout<<"Linear regression: Maximum Likelihood."<<endl;
  cout<<"Params : "<<endl;
  cout<<W<<endl;
  cout<<"Error = "<<err<<endl;
  cout<<"Time elapsed: "<<(end - begin) / (double)CLOCKS_PER_SEC <<endl;


  begin = clock();
  mat W2 = gradient_descentMAP(X, data.col(1), 10000, 0.001, 0.0001);
  end = clock();
  err = sum( pow((X * W) - data.col(1), 2) );

  cout<<"Linear regression: MAP"<<endl;
  cout<<"Params : "<<endl;
  cout<<W2<<endl;
  cout<<"Error = "<<err<<endl;
  cout<<"Time elapsed: "<<(end - begin) / (double)CLOCKS_PER_SEC <<endl;

  // To plot

  mat Y = (X * W);
  ((mat) X.col(1)).save("x.mio", raw_ascii);
  Y.save("y.mio", raw_ascii);
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
