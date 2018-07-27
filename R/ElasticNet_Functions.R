Elastic_Net_LOO = function(X, y, i, alpha, lambda, intercept = TRUE) {
  # find out the dimension of X
  n = dim(X)[1]
  p = dim(X)[2]
  # compute the scale parameter for y
  sd.y = sqrt(var(y[-i]) * (n - 1) / (n - 2))
  y.scaled = y / sd.y
  X.scaled = X / sd.y
  # build the model
  model = glmnet(
    x = X.scaled[-i, ],
    y = y.scaled[-i],
    family = "gaussian",
    alpha = alpha,
    lambda = lambda / sd.y ^ 2 / (n - 1) * n,
    thresh = 1E-14,
    intercept = intercept,
    standardize = FALSE,
    maxit = 1000000
  )
  # prediction
  beta.hat = as.matrix(model$beta)
  intercept.hat = model$a0 * sd.y
  y.loo = vector(mode = "double", length = length(lambda))
  for (k in 1:length(lambda))
    y.loo[k] = X[i, ] %*% beta.hat[, k] + intercept.hat[k]
  return(y.loo)
}