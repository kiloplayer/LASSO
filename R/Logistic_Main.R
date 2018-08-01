# Logistic Regression -----------------------------------------------------
setwd("E:\\Columbia_University\\Internship\\R_File\\LASSO\\")
library(glmnet)
library(ggplot2)
library(Rcpp)
sourceCpp("src/ALO_Primal.cpp")
source("R/Logistic_Functions.R")


# Logistic with Intercept -------------------------------------------------

# parameters
n = 300
p = 600
k = 60
lambda = 10 ^ seq(log10(5E-4), log10(1E-1), length.out = 50)
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
prob = exp(y.linear) / (1 + exp(y.linear))
y = rbinom(n, 1, prob = prob)
y.factor = factor(y)

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
    Logistic_LOO(X, y.factor, i, alpha[k], lambda, intercept = TRUE)
  # save the prediction value
  y.loo[i,] = y.temp
  # print middle result
  if (i %% 10 == 0)
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
     file = "RData/Logistic_LOO.RData")

# approximate leave-one-out
load('RData/Logistic_LOO.RData')
# find the ALO prediction
y.alo = matrix(ncol = dim(param)[1], nrow = n)
starttime = proc.time() # count time
for (k in 1:length(alpha)) {
  # build the full data model
  model = glmnet(
    x = X,
    y = y,
    family = "binomial",
    alpha = alpha[k],
    lambda = lambda,
    thresh = 1E-14,
    intercept = TRUE,
    standardize = FALSE,
    maxit = 1000000
  )
  # find the prediction for each alpha value
  y.temp <- foreach(j = 1:length(lambda), .combine = cbind) %do% {
    LogisticALO(as.vector(model$beta[, j]),
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
     file = "RData/Logistic_ALO")

# plot
load("RData/Logistic_ALO")
result$alpha = factor(result$alpha)
p = ggplot(result) +
  geom_line(aes(x = log10(lambda), y = risk.loo), col = "black", lty = 2) +
  geom_line(aes(x = log10(lambda), y = risk.alo), col = "red", lty = 2) +
  ggtitle('Logistic Regression with Intercept & Elastic Net penalty') +
  xlab("Logarithm of Lambda") +
  ylab("Risk Estimate") +
  facet_wrap( ~ alpha, nrow = 2)
bmp("figure/Logistic_with_Intercept.bmp",
    width = 1280,
    height = 720)
p
dev.off()


# Logistic without Intercept ----------------------------------------------

# parameters
n = 300
p = 600
k = 60
lambda = 10 ^ seq(log10(5E-4), log10(1E-1), length.out = 50)
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
intercept = 0
X = matrix(rnorm(n * p, mean = 0, sd = sqrt(1 / k)), ncol = p)
y.linear = intercept + X %*% beta
prob = exp(y.linear) / (1 + exp(y.linear))
y = rbinom(n, 1, prob = prob)
y.factor = factor(y)

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
    Logistic_LOO(X, y.factor, i, alpha[k], lambda, intercept = FALSE)
  # save the prediction value
  y.loo[i,] = y.temp
  # print middle result
  if (i %% 10 == 0)
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
     file = "RData/Logistic_without_InterceptLOO.RData")

# approximate leave-one-out
load('RData/Logistic_without_InterceptLOO.RData')
# find the ALO prediction
y.alo = matrix(ncol = dim(param)[1], nrow = n)
starttime = proc.time() # count time
for (k in 1:length(alpha)) {
  # build the full data model
  model = glmnet(
    x = X,
    y = y,
    family = "binomial",
    alpha = alpha[k],
    lambda = lambda,
    thresh = 1E-14,
    intercept = FALSE,
    standardize = FALSE,
    maxit = 1000000
  )
  # find the prediction for each alpha value
  y.temp <- foreach(j = 1:length(lambda), .combine = cbind) %do% {
    LogisticALO(as.vector(model$beta[, j]),
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
     file = "RData/Logistic_without_Intercept_ALO")

# plot
load("RData/Logistic_without_Intercept_ALO")
result$alpha = factor(result$alpha)
p = ggplot(result) +
  geom_line(aes(x = log10(lambda), y = risk.loo), col = "black", lty = 2) +
  geom_line(aes(x = log10(lambda), y = risk.alo), col = "red", lty = 2) +
  ggtitle('Logistic Regression with Elastic Net penalty & without Intercept') +
  xlab("Logarithm of Lambda") +
  ylab("Risk Estimate") +
  facet_wrap( ~ alpha, nrow = 2)
bmp("figure/Logistic_without_Intercept.bmp",
    width = 1280,
    height = 720)
p
dev.off()


# # Multinomial with Intercept ----------------------------------------------
# 
# # parameters
# n = 200
# p = 600
# k = 60
# num_class = 5
# lambda = 10 ^ seq(log10(5E-4), log10(1E-1), length.out = 50)
# lambda = sort(lambda, decreasing = TRUE)
# alpha = seq(0, 1, 0.1)
# param = data.frame(alpha = numeric(0),
#                    lambda = numeric(0))
# for (i in 1:length(alpha)) {
#   for (j in 1:length(lambda)) {
#     param[j + (i - 1) * length(lambda), c('alpha', 'lambda')] = c(alpha[i], lambda[j])
#   }
# }
# set.seed(1234)
# 
# # simulation
# data("MultinomialExample")
# beta = matrix(rnorm(num_class * p, mean = 0, sd = 1), ncol = num_class)
# beta[(k + 1):p,] = 0
# intercept = rnorm(num_class, mean = 0, sd = 1)
# X = matrix(rnorm(n * p, mean = 0, sd = sqrt(1 / k)), ncol = p)
# y.linear = matrix(rep(intercept, n), ncol = num_class, byrow = TRUE) +
#   X %*% beta
# prob = exp(y.linear) / matrix(rep(rowSums(exp(y.linear)), num_class), ncol =
#                                 num_class)
# y.mat = t(apply(prob, 1, function(x)
#   rmultinom(1, 1, prob = x))) # N * K matrix (N - #obs, K - #class)
# y.num = apply(y.mat == 1, 1, which) # vector
# y.num.factor = factor(y.num, levels = seq(1:num_class))
# 
# # true leave-one-out
# y.loo = array(numeric(0), dim = c(dim(y.mat)[1],
#                                   dim(y.mat)[2],
#                                   dim(param)[1])) # N * K * #param
# starttime = proc.time() # count time
# library(foreach)
# library(doParallel)
# no_cores = detectCores() - 1
# cl = makeCluster(no_cores)
# registerDoParallel(cl)
# for (i in 1:n) {
#   # do leave one out prediction
#   y.temp <-
#     foreach(
#       k = 1:length(alpha),
#       .combine = cbind,
#       .packages = 'glmnet'
#     ) %dopar%
#     Multinomial_LOO(X, y.num.factor, i, alpha[k], lambda, intercept = TRUE)
#   # save the prediction value
#   y.loo[i, , ] = y.temp
#   # print middle result
#   if (i %% 1 == 0)
#     print(
#       paste(
#         i,
#         " samples have beed calculated. ",
#         "On average, every sample needs ",
#         round((proc.time() - starttime)[3] / i, 2),
#         " seconds."
#       )
#     )
# }
# stopCluster(cl)
# # true leave-one-out risk estimate
# risk.loo = vector(mode = 'double', length = dim(param)[1])
# for (k in 1:dim(param)[1]) {
#   risk.loo[k] = 1 / n * sum(colSums((y.loo[, , k] - y.mat) ^ 2))
# }
# # record the result
# result = cbind(param, risk.loo)
# # save the data
# save(result, y.loo,
#      file = "RData/Multinomial_LOO.RData")
# 
# # approximate leave-one-out
# load('RData/Multinomial_LOO.RData')
# # find the ALO prediction
# y.alo = array(numeric(0), dim = c(dim(y.mat)[1],
#                                   dim(y.mat)[2],
#                                   dim(param)[1])) # N * K * #param
# starttime = proc.time() # count time
# for (k in 1:length(alpha)) {
#   # build the full data model
#   model = glmnet(
#     x = X,
#     y = y.num.factor,
#     family = "multinomial",
#     alpha = alpha[k],
#     lambda = lambda,
#     thresh = 1E-14,
#     intercept = TRUE,
#     standardize = FALSE,
#     maxit = 1000000
#   )
#   # find the prediction for each alpha value
#   foreach(j = 1:length(lambda), .combine = cbind) %do% {
#     # extract beta under all of the class
#     beta.temp = matrix(nrow = p, ncol = num_class)
#     for (i in 1:num_class) {
#       beta.temp[, i] = as.matrix(model$beta[[i]])[, j]
#     }
#     y.alo[, , (k - 1) * length(lambda) + j] =
#       MultinomialALO(beta.temp,
#                      as.vector(model$a0[, j]),
#                      X,
#                      y.mat,
#                      lambda[j],
#                      alpha[k])
#   }
#   # print middle result
#   print(
#     paste(
#       k,
#       " alphas have beed calculated. ",
#       "On average, every alpha needs ",
#       round((proc.time() - starttime)[3] / k, 2),
#       " seconds."
#     )
#   )
# }
# # true leave-one-out risk estimate
# risk.alo = vector(mode = 'double', length = dim(param)[1])
# for (k in 1:dim(param)[1]) {
#   risk.alo[k] = 1 / n * sum(colSums((y.alo[, , k] - y.mat) ^ 2))
# }
# # record the result
# result = cbind(result, risk.alo)
# 
# # save the data
# save(result, y.loo, y.alo,
#      file = "RData/Multinomial_ALO")
# 
# # plot
# load("RData/Multinomial_ALO")
# result$alpha = factor(result$alpha)
# p = ggplot(result) +
#   geom_line(aes(x = log10(lambda), y = risk.loo), col = "black", lty = 2) +
#   geom_line(aes(x = log10(lambda), y = risk.alo), col = "red", lty = 2) +
#   ggtitle('Logistic Regression with Intercept & Elastic Net penalty') +
#   xlab("Logarithm of Lambda") +
#   ylab("Risk Estimate") +
#   facet_wrap( ~ alpha, nrow = 2)
# bmp("figure/Logistic_with_Intercept.bmp",
#     width = 1280,
#     height = 720)
# p
# dev.off()
