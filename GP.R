# Theo AO Rashid -- February 2020

# ----- Machine Learning: Gaussian processes -----
# Task set by Seth Flaxman
library(MASS)
library(ggplot2)
library(dplyr)

set.seed(1)

# 1. Calculate the covariance matrix from a given kernel

# ----- DEFINE SOME KERNELS -----
# kernels act between points x_i and x_j
# Typically they have amplitude parameter (sigma)
# and length-scale (l) -- a smoothness parameter

# # Constant
# const_kernel <- function(x_i, x_j, C = 1) {
#     return(C)
# }

# # Linear
# lin_kernel <- function(x_i, x_j) {
#     return(t(x_i) * x_j)
# }

# Squared exponential 
se_kernel <- function(x_i, x_j, sigma = 1, l = 1) {
    return(sigma^2 * exp(- (x_i - x_j)^2 / (2 * l^2)))
}

# Periodic
period_kernel <- function(x_i, x_j, kappa = 1, sigma = 1, l = 1) {
    return(sigma^2 * exp(-2 * sin(kappa * abs(x_i - x_j) / 2 * pi)^2 / l^2))
}

# Matérn
matern_kernel <- function(x_i, x_j, nu = 1.5, sigma = 1, l = 1) {
    if (!(nu %in% c(0.5, 1.5, 2.5))) {
        stop("p must be equal to 0.5, 1.5 or 2.5")
    }
    p <- nu - 0.5
    d <- abs(x_i - x_j)
    if (p == 0) {
        return(sigma^2 * exp(- d / l))
    } else if (p == 1) {
        return(sigma^2 * (1 + sqrt(3)*d/l) * exp(- sqrt(3)*d/l))
    } else {
        return(sigma^2 * (1 + sqrt(5)*d/l + 5*d^2 / (3*l^2)) * exp(-sqrt(5)*d/l))
    }
}

# ----- CALCULATE THE COVARIANCE MATRIX -----
# generate covariance matrix for points in `x` using given kernel function
cov_matrix <- function(x, kernel_fn, ...) {
    return(outer(x, x, function(a, b) kernel_fn(a, b, ...)))
}

# Plot the covariance matrix in 2D
x <- seq(0, 5, length.out = 501)  # x-coordinates between 0 and 5
# image(x = x, y = x, z = cov_matrix(x, matern_kernel))

# Plot the covariance matrix in 1D
# data <- cov_matrix(x, se_kernel) 
# data <- data[sample(1:length(x), 1),] # Pick a random row
# plot_cov <- ggplot(data.frame(x, data), aes(x = x, y = data)) +
#     geom_line(size = 0.5) +
#     labs(title = "", x = "", y = "") +
#     theme_minimal()
# print(plot_cov)

# 2. Draw samples from the Gaussian process GP(0, K)
draw_GP_samples <- function(x, N, kernel_fn, ...) {
    # x -- coordinates
    # N -- number of draws
    Y <- matrix(NA, nrow = length(x), ncol = N)
    for (i in 1:N) {
        K <- cov_matrix(x, kernel_fn, ...)
        # use mvnorm from MASS with zero vector mean, one sample at a time
        Y[, i] <- mvrnorm(n = 1, mu = rep(0, times = length(x)), Sigma = K)
    }
    return(Y) # each column is a sample
}

N_samples <- 3
Y <- draw_GP_samples(x, N = N_samples, kernel_fn = period_kernel, l = 10)

# # Plotting works only for N_samples = 3
# plot_sample <- ggplot(data.frame(x, Y), aes(x = x)) +
#     geom_line(aes(y = Y[,1])) +
#     geom_line(aes(y = Y[,2])) +
#     geom_line(aes(y = Y[,3])) +
#     labs(title = "", x = "", y = "") +
#     theme_minimal()
# print(plot_sample)
# Varying length scale works as expected
# Shown best for periodic kernel function

# 3. Add data points
# Let's try for population data
pop_df <- read.csv("/Users/tar15/Dropbox/SPH/Data/Unit Test/pop_IMD_2004_17_hf.csv", row.names = 1)
# Add the populations over age groups and sexes
pop_LSOA_df <- pop_df %>%
    group_by(LSOA2011, YEAR) %>%
    summarise(population = sum(population))
# For now, choose only one LSOA
pop_samp_df <- pop_LSOA_df %>%
    filter(LSOA2011 == "E01001851")
# plot_pop <- ggplot(pop_samp_df, aes(x = YEAR, y = population)) +
#     geom_point()
# print(plot_pop)

X_train <- pop_samp_df$YEAR
y_train <- pop_samp_df$population

kernel <- matern_kernel # choose kernel
sigma_noise <- 0 # noise model

# ----- PRIORS -----
# prior is a sample from GP(0, K(., .))
# We have no knowledge to select other mean function value
prior_f <- draw_GP_samples(X_train, N = 1, kernel_fn = kernel)
# plot_prior <- ggplot(data.frame(X_train, prior_f), aes(x = X_train, y = prior_f)) +
#     geom_line() +
#     labs(title = "", x = "", y = "") +
#     theme_minimal()
# print(plot_prior)

K_ss <- kernel(X_train, X_train) # prior variance

# ----- POSTERIOR -----
# Test data
X_test <- seq(min(X_train) - 10, max(X_train) + 10, length.out = 10)
K <- kernel(X_test, X_test)
# m_post <- m_prior + solve(kernel(X_train, X_test) + sigma_noise^2 * diag(length(X_train))) %*% (y_train - m_prior(X_train))

# Posterior predictive distribution at test inputs X_s
# Obtained by Gaussian conditioning

