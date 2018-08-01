# Poisson Regression ------------------------------------------------------
setwd("E:\\Columbia_University\\Internship\\R_File\\LASSO\\")
library(glmnet)
library(ggplot2)
library(Rcpp)
sourceCpp("src/ALO_Primal.cpp")
source("R/ElasticNet_Functions.R")


# Poisson with Intercept --------------------------------------------------

# parameters
n = 300
p = 600
k = 60
lambda = 10 ^ seq(log10(5E-3), log10(1), length.out = 50)
lambda = sort(lambda, decreasing = TRUE)
alpha = seq(0, 1, 0.1)
param = data.frame(alpha = numeric(0),
                   lambda = numeric(0))
for (i in 1:length(alpha)) {
  for (j in 1:length(lambda)) {
    param[j + (i - 1) * length(lambda), c('alpha', 'lambda')] = c(alpha[i], lambda[j])
  }
}
set.seed(1234)

# simulation
beta = rnorm(p, mean = 0, sd = 1)
beta[(k + 1):p] = 0
intercept = 1
X = matrix(rnorm(n * p, mean = 0, sd = sqrt(1 / k)), ncol = p)
y.linear = intercept + X %*% beta
pois_lambda = exp(y.linear)
y = rpois(n, lambda = pois_lambda)

# true leave-one-out
y.loo = matrix(ncol = dim(param)[1], nrow = n)
starttime = proc.time() # count time
library(foreach)
library(doParallel)
no_cores = detectCores() - 1
cl = makeCluster(no_cores)
registerDoParallel(cl)
for (i in 1:n) {
  # do leave one out prediction
  y.temp <-
    foreach(
      k = 1:length(alpha),
      .combine = cbind,
      .packages = 'glmnet'
    ) %dopar%
    Poisson_LOO(X, y, i, alpha[k], lambda, intercept = TRUE)
  # save the prediction value
  y.loo[i, ] = y.temp
  # print middle result
  if (i %% 1 == 0)
    print(
      paste(
        i,
        " samples have beed calculated. ",
        "On average, every sample needs ",
        round((proc.time() - starttime)[3] / i, 2),
        " seconds."
      )
    )
}
stopCluster(cl)
# true leave-one-out risk estimate
risk.loo = 1 / n * colSums((y.loo -
                              matrix(rep(y, dim(
                                param
                              )[1]), ncol = dim(param)[1])) ^ 2)
# record the result
result = cbind(param, risk.loo)
# save the data
save(result, y.loo,
     file = "RData/Poisson_LOO.RData")

# approximate leave-one-out
load('RData/Poisson_LOO.RData')
# find the ALO prediction
y.alo = matrix(ncol = dim(param)[1], nrow = n)
starttime = proc.time() # count time
for (k in 1:length(alpha)) {
  # build the full data model
  model = glmnet(
    x = X,
    y = y,
    family = "poisson",
    alpha = alpha[k],
    lambda = lambda,
    thresh = 1E-14,
    intercept = TRUE,
    standardize = FALSE,
    maxit = 1000000
  )
  # find the prediction for each alpha value
  y.temp <- foreach(j = 1:length(lambda), .combine = cbind) %do% {
    PoissonALO(as.vector(model$beta[, j]),
               model$a0[j],
               X,
               y,
               lambda[j],
               alpha[k])
  }
  y.alo[, ((k - 1) * length(lambda) + 1):(k * length(lambda))] = y.temp
  # print middle result
  print(
    paste(
      k,
      " alphas have beed calculated. ",
      "On average, every alpha needs ",
      round((proc.time() - starttime)[3] / k, 2),
      " seconds."
    )
  )
}
# true leave-one-out risk estimate
risk.alo = 1 / n * colSums((y.alo -
                              matrix(rep(y, dim(
                                param
                              )[1]), ncol = dim(param)[1])) ^ 2)
# record the result
result = cbind(result, risk.alo)

# save the data
save(result, y.loo, y.alo,
     file = "RData/Poisson_ALO.RData")

# plot
load("RData/Poisson_ALO.RData")
result$alpha = factor(result$alpha)
p = ggplot(result) +
  geom_line(aes(x = log10(lambda), y = risk.loo), col = "black", lty = 2) +
  geom_line(aes(x = log10(lambda), y = risk.alo), col = "red", lty = 2) +
  ggtitle('Poisson Regression with Intercept & Elastic Net penalty') +
  xlab("Logarithm of Lambda") +
  ylab("Risk Estimate") +
  facet_wrap(~ alpha, nrow = 2)
bmp("figure/Poisson_with_Intercept.bmp",
    width = 1280,
    height = 720)
p
dev.off()