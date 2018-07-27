# Generalized_LASSO -------------------------------------------------------
setwd("E:\\Columbia_University\\Internship\\R_File\\LASSO\\")
library(glmnet)
library(ggplot2)
library(Rcpp)
library(CVXR)
library(MASS)
library(foreach)
library(doParallel)
sourceCpp("src/ALO_Primal.cpp")
source("R/GenLASSO_Functions.R")

# Generalized LASSO with Intercept ----------------------------------------

# parameters
n = 200
p = 400
k = 20
lambda = 10 ^ seq(log10(0.1), log10(10), length.out = 40)
D = matrix(0, ncol = p + 1, nrow = p)
for (i in 1:p) {
  if (i == 1) {
    next
  } else {
    D[i, i] = -1
    D[i, i + 1] = 1
  }
}
param = data.frame(lambda = lambda)
set.seed(1)

# simulation
beta0 = rnorm(p, mean = 0, sd = 1)
beta0[(k + 1):p] = 0
beta0 = sample(beta0) # shuffle
beta = cumsum(beta0)
beta = (beta - mean(beta)) / sd(beta)
rm(beta0)
X = matrix(rnorm(n * p, mean = 0, sd = sqrt(1 / k)), ncol = p)
intercept = 1
sigma = rnorm(n, mean = 0, sd = 0.5)
y = intercept + X %*% beta + sigma

# true leave-one-out
y.loo = matrix(ncol = dim(param)[1], nrow = n)
starttime = proc.time() # count time
no_cores = detectCores() - 1
cl = makeCluster(no_cores)
registerDoParallel(cl)
for (i in 1:n) {
  # do leave one out prediction
  y.temp <-
    foreach(
      k = 1:length(lambda),
      .combine = cbind,
      .packages = 'CVXR'
    ) %dopar%
    GenLASSO_LOO(X, y, i, lambda[k], intercept = TRUE)
  # save the prediction value
  y.loo[i,] = y.temp
  # print middle result
  if (i %% 10 == 0)
    print(
      paste(
        i,
        "samples have beed calculated.",
        "On average, every sample needs",
        round((proc.time() - starttime)[3] / i, 2),
        "seconds."
      )
    )
}
stopCluster(cl)
# true leave-one-out risk estimate
risk.loo = 1 / n *
  colSums((y.loo - matrix(rep(y, dim(
    param
  )[1]),
  ncol = dim(param)[1])) ^ 2)
# record the result
result = cbind(param, risk.loo)
# save the data
save(result, y.loo,
     file = "RData/GenLASSO_LOO.RData")

# approximate leave-one-out
load('RData/GenLASSO_LOO.RData')
y.alo = matrix(ncol = dim(param)[1], nrow = n)
starttime = proc.time() # count time
y.alo <- foreach(k = 1:length(lambda),
                 .combine = cbind,
                 .packages = 'CVXR') %do%
  GenLASSO_ALO(X, y, D, lambda[k], intercept = TRUE)
# true leave-one-out risk estimate
risk.alo = 1 / n * colSums((y.alo -
                              matrix(rep(y, dim(
                                param
                              )[1]), ncol = dim(param)[1])) ^ 2)
# record the result
result = cbind(result, risk.alo)

# save the data
save(result, y.loo, y.alo,
     file = "RData/GenLASSO_ALO.RData")

# plot
load("RData/GenLASSO_ALO.RData")
p = ggplot(result) +
  geom_line(aes(x = log10(lambda), y = risk.loo), lty = 2) +
  geom_line(aes(x = log10(lambda), y = risk.alo), col = "red", lty = 2)
bmp("figure/GenLASSO_with_Intercept.bmp",
    width = 1280,
    height = 720)
p
dev.off()