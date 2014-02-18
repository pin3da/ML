
mat gradient_descent(mat X, vec T, int iter, double alpha, double EPS = 1e-9) {
  mat W(2, 1);
  W.randu();

  double err = sum( pow((X * W) - T, 2) );
  alpha = 0.01;
  while (err > EPS and iter--) {
    mat tmp(2,1);
    tmp(0,0) = sum((X * W) - T);
    tmp(1,0) = sum(((X * W) - T).t() * X);
    mat Wn = W - alpha * tmp;
    double ant = err;
    err = sum( pow((X * Wn) - T, 2) );
    if (err >= ant) alpha *= 0.5;
    else W = Wn;
  }

  return W;
}



mat gradient_descentMAP(mat X, vec T, int iter, double lf, double EPS = 1e-9) { 
  mat W(2, 1);
  W.randu();
  double alpha = 0.0001;
  double err = sum( pow((X * W) - T, 2) );  
  double beta = err/X.n_rows;
  
  while (err > EPS and iter--) {
    mat tmp(2,1);   // nuevo W
    tmp(0,0) = (1/beta)* sum((X * W) - T) + alpha * W(0,0); // reajuste de w0
    tmp(1,0) = (1/beta) * sum(((X * W) - T).t() * X) + alpha * W(1,0);   // reajuste de w1
    W = W - lf * tmp;  // reajuste de los w
    double ant = err;  
    beta = err/X.n_rows; 
    err = sum( pow((X * W) - T, 2) );   
    if (err >= ant) lf *= 0.5;       // reajusta el factor de aprendizaje si el error no se reduce
  }

  return W;
}
