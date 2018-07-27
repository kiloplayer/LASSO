GenLASSO_LOO = function(X, y, i, lambda, intercept = TRUE) {
  # adjust for the intercept
  if (intercept) {
    X = cbind(1, X)
  }
  # find out the dimension of X
  n = dim(X)[1]
  p = dim(X)[2]
  # fit the primal optimization problem
  beta = Variable(p)
  objective = Minimize(1 / 2 * sum_squares(y[-i] - X[-i, ] %*% beta) +
                         lambda * p_norm(D %*% beta, 1))
  problem = Problem(objective)
  result = solve(problem)
  # prediction
  beta = result$getValue(beta)
  y.loo = X[i, ] %*% beta
  return(y.loo)
}

GenLASSO_ALO = function(X, y, D, lambda, intercept = TRUE) {
  # adjust for the intercept
  if (intercept) {
    X = cbind(1, X)
  }
  # find out the dimension of X
  n = dim(X)[1]
  p = dim(X)[2]
  # fit the primal optimization problem to find beta
  beta = Variable(p)
  objective = Minimize(1 / 2 * sum_squares(y - X %*% beta) +
                         lambda * p_norm(D %*% beta, 1))
  problem = Problem(objective)
  result = solve(problem)
  beta = result$getValue(beta)
  # fit the dual optimization problem to find u
  theta = Variable(n)
  u = Variable(dim(D)[1])
  objective = Minimize(1 / 2 * sum_squares(theta - y))
  constraints = list(t(D) %*% u == t(X) %*% theta, abs(u) <= lambda)
  problem = Problem(objective, constraints)
  result = solve(problem)
  u = result$getValue(u)
  # compute alo prediction
  y.alo = GenLASSOALO(beta, u, X, y, D, lambda, 1E-4)
  return(y.alo)
}