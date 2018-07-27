#include <RcppArmadillo.h>
#include <math.h>

using namespace Rcpp;
using namespace arma;

// [[Rcpp::plugins(cpp11)]]

// [[Rcpp::export]]
vec ElasticNetALO(vec beta, double intercept, 
                  mat X, vec y, 
                  double lambda, double alpha) {
  // find out the dimension of X
  double n = X.n_rows;
  double p = X.n_cols;
  // define full data set
  vec ones(n,fill::ones);
  mat X_full = X;
  X_full.insert_cols(0, ones);
  // define full beta (intercept & slope)
  vec beta_full(p + 1);
  beta_full(0) = intercept;
  beta_full(span(1, p)) = beta;
  // compute prediction
  vec y_hat = X_full * beta_full;
  // find active set
  uvec A = find(beta_full != 0);
  // compute matrix H
  mat H(n, n, fill::zeros);
  if(!A.is_empty()) {
    mat X_active = X_full.cols(A);
    mat R_diff2(A.n_elem, A.n_elem, fill::eye);
    R_diff2 = R_diff2 * n * lambda * (1-alpha);
    if(intercept == 0) {
      R_diff2(0,0) = 0;
    }
    H = X_active * inv_sympd(X_active.t() * X_active + R_diff2) * X_active.t();
  }
  // compute the ALO prediction
  vec y_alo = y_hat + H.diag() % (y_hat - y) / (1-H.diag());
  return y_alo;
}

// [[Rcpp::export]]
vec GenLASSOALO(vec beta, vec u, mat X, vec y, mat D, 
                double lambda, double tol) {
  // compute prediction
  vec y_hat = X * beta;
  // find complement of the active set
  uvec mE = find(abs(abs(u) - lambda) >= tol);
  // find the null space of D_{-E,*}
  mat VE = null(D.rows(mE));
  // find matrix H
  mat H = X * VE * pinv(X * VE);
  // compute the ALO prediction
  vec y_alo = y + (y_hat - y) / (1-H.diag());
  return y_alo;
}

// You can include R code blocks in C++ files processed with sourceCpp
// (useful for testing and development). The R code will be automatically 
// run after the compilation.
//

/*** R

*/
