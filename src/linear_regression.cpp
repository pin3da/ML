
mat gradient_descent(mat X, vec T, int iter, double alpha, double EPS = 1e-9) {
  mat W(2, 1);
  W.randu();

  double err = sum( pow((X * W) - T, 2) );
  alpha = 0.01;
  while (err > EPS and iter--) {
    mat tmp(2,1);
    tmp(0,0) = sum((X * W) - T);
    tmp(1,0) = sum(((X * W) - T).t() * X);
    W = W - alpha * tmp;
    double ant = err;
    err = sum( pow((X * W) - T, 2) );
    if (err >= ant) alpha *= 0.5;
  }

  return W;
}
