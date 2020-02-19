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

# Constant
const_kernel <- function(x_i, x_j, C = 1) {
   return(C + 0 * x_i*x_j) # <-- this works formysterious reasons
   # return(C)
}

# Linear
lin_kernel <- function(x_i, x_j) {
    return(x_i * x_j)
}

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
cov_matrix <- function(x1, x2, kernel_fn, ...) {
    return(outer(x1, x2, function(a, b) kernel_fn(a, b, ...)))
}

# Plot the covariance matrix in 2D
x <- seq(0, 5, length.out = 501)  # x-coordinates between 0 and 5
# image(x = x, y = x, z = cov_matrix(x, x, matern_kernel))

# Plot the covariance matrix in 1D
data <- cov_matrix(x, x, se_kernel) 
data <- data[sample(1:length(x), 1),] # Pick a random row
plot_cov <- ggplot(data.frame(x, data), aes(x = x, y = data)) +
    geom_line(size = 0.5) +
    labs(title = "", x = "", y = "") +
    theme_minimal()
# print(plot_cov)

# 2. Draw sample from the Gaussian process GP(m(.), K(.,.))
GP_sample <- function(m, K) {
    # m -- mean function vector
    # K -- covariance matrix
    # use mvnorm from MASS
    GP_samp <- mvrnorm(n = 1, mu = m, Sigma = K) # one sample
    return(GP_samp) # each column is a sample
}

# Plot 3 samples, vary l
N_samples <- 3
Y <- matrix(NA, nrow = length(x), ncol = N_samples)
K <- cov_matrix(x, x, matern_kernel, l = 10)
for (i in 1:N_samples) {
    # zero mean vector
    Y[, i] <- GP_sample(m = matrix(0, length(x), 1), K = K)
}
# Plotting works only for N_samples = 3
plot_sample <- ggplot(data.frame(x, Y), aes(x = x)) +
    geom_line(aes(y = Y[,1])) +
    geom_line(aes(y = Y[,2])) +
    geom_line(aes(y = Y[,3])) +
    labs(title = "", x = "", y = "") +
    theme_minimal()
# print(plot_sample)
# Varying length scale works as expected
# Shown best for periodic kernel function

# 3. Add data points
# ----- ONS POPULATION DATA -----
pop_df <- read.csv("/Users/tar15/Dropbox/SPH/Data/Unit Test/pop_IMD_2004_17_hf.csv", row.names = 1)
# Add the populations over age groups and sexes
pop_LSOA_df <- pop_df %>%
    group_by(LSOA2011, YEAR) %>%
    summarise(population = sum(population))
# For now, choose only one LSOA
pop_samp_df <- pop_LSOA_df %>%
    filter(LSOA2011 == "E01001851")
plot_pop <- ggplot(pop_samp_df, aes(x = YEAR, y = population)) +
    geom_point()
# print(plot_pop)

X_train <- pop_samp_df$YEAR
y_train <- scale(pop_samp_df$population) # scale so m = 0 is appropriate

kernel <- matern_kernel # choose kernel
sigma_noise <- 0.1 # noise model

# Test data -- around the range of years
X_test <- seq(min(X_train) - 5, max(X_train) + 5, length.out = 576)

# ----- PRIORS -----
# prior is a sample from GP(0, K(., .))
# We have no knowledge to select other mean function value
m_prior = matrix(0, length(X_test), 1) # prior mean = m(X_*) = 0
k_prior <- cov_matrix(X_test, X_test, kernel) # prior variance k(X_*, X_*)

prior_f <- GP_sample(m = m_prior, K = k_prior) # example draw
plot_prior <- ggplot() +
    geom_point(aes(x = X_train, y = y_train)) +
    geom_line(aes(x = X_test, y = prior_f)) +
    labs(title = "", x = "", y = "") +
    theme_minimal()
# print(plot_prior)

# ----- POSTERIOR -----
# X -- X_train; X_* -- X_test
# Posterior predictive distribution at test inputs X_*
# Obtained by Gaussian conditioning

K <- cov_matrix(X_train, X_train, kernel) # K = k(X, X)

# posterior mean
kalman_gain <- cov_matrix(X_test, X_train, kernel) %*% solve((K + sigma_noise^2 * diag(length(X_train))))
m_post <- m_prior + kalman_gain %*% (y_train - matrix(0, length(X_train), 1))

# posterior variance
k_post <- k_prior - kalman_gain %*% cov_matrix(X_train, X_test, kernel)
ci_95 <- sqrt(diag(k_post))*1.96 # confidence intervals

post_f <- GP_sample(m = m_post, K = k_post) # example draw
plot_post <- ggplot() +
    geom_point(aes(x = X_train, y = y_train)) +
    geom_line(aes(x = X_test, y = post_f)) +
    geom_ribbon(aes(x=X_test, ymin=post_f-ci_95, ymax=post_f+ci_95),fill="blue", alpha=0.2) +
    labs(title = "", x = "", y = "") +
    theme_minimal()
# print(plot_post)

# 4. Write a function to get the marginal likelihood
# and optimise the lengthscale of the kernel

# WORK OUT MARGINAL LIKELIHOOD OF THE DATA
# AND OPTIMISE USING optim