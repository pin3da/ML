
mat gradient_descent(mat X, vec T, int iter, double step, double EPS = 1e-9) {
  mat W(2, 1);
  W.randu();

  double err = sum( pow((X * W) - T, 2) );
  step = 0.01;
  while (err > EPS and iter--) {
    mat tmp(2,1);
    tmp(0,0) = sum((X * W) - T);
    tmp(1,0) = sum(((X * W) - T).t() * X);
    mat Wn = W - step * tmp;
    double ant = err;
    err = sum( pow((X * Wn) - T, 2) );
    if (err >= ant) step *= 0.5;
    else W = Wn;
  }

  return W;
}



mat gradient_descentMAP(mat X, vec T, int iter, double step, double alpha, double EPS = 1e-9) {
  mat W(2, 1);
  W.randu();
  double err   = sum( pow((X * W) - T, 2) );
  double beta  = err/X.n_rows;

  while (err > EPS and iter--) {
    mat tmp(2,1);
    tmp(0,0) = (1.0 / beta) * sum((X * W) - T) + alpha * W(0,0);
    tmp(1,0) = (1.0 / beta) * sum(((X * W) - T).t() * X) + alpha * W(1,0);
    mat Wn = W - step * tmp;
    double ant = err;
    beta = err/X.n_rows;
    err = sum( pow((X * Wn) - T, 2) );
    if (err >= ant) step *= 0.5;
    else W = Wn;
  }

  return W;
}
