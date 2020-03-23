# Theo AO Rashid -- March 2020

# ----- Machine Learning: Gaussian processes -----
# Demo for group presentation

library(MASS)
library(ggplot2)
library(dplyr)

#set.seed(1)

# ----- DEFINE SOME KERNELS -----
# Squared exponential 
se_kernel <- function(x_i, x_j, sigma = 1, l = 1) {
    return(sigma^2 * exp(- (x_i - x_j)^2 / (2 * l^2)))
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
cov_matrix <- function(x1, x2, kernel_fn, ...) {
    return(outer(x1, x2, function(a, b) kernel_fn(a, b, ...)))
}

# ----- ONS POPULATION DATA -----
# Simulated deaths in Hammersmith and Fulham 2004-2017
mortality <- read.csv("/Users/tar15/Dropbox/SPH/Data/Unit Test/mortsim_hf.csv", row.names = 1)
mortality_m <- mortality %>%
    filter(sex == 1) %>%
    select(-sex) %>%
    filter(LSOA2011 == "E01001851") %>% # focus on one LSOA
    select(YEAR, deaths) %>% # focus only on year, deaths and populations
    group_by(YEAR) %>%
    summarise(deaths = sum(deaths))

plot_year_deaths <- ggplot(mortality_m, aes(x = YEAR, y = deaths)) +
    geom_point()

X_train <- mortality_m$YEAR
y_train <- scale(mortality_m$deaths) # Rescale the data so m = 0 is appropriate

X_test <- seq(min(X_train) - 5, max(X_train) + 5, length.out = 576) # data around the years

# ----- GP inference -----
# Hyperparameters
kernel <- matern_kernel # choose kernel
sigma_noise <- 0.1 # noise model
length_scale <- 1 # for SE/Matern kernel
amplitude <- 1 # for SE/Matern kernel

# Prior -- sample from GP(0, K(., .))
m_prior = matrix(0, length(X_test), 1)
k_prior <- cov_matrix(X_test, X_test, kernel, sigma = amplitude, l = length_scale)

prior_f <- mvrnorm(n = 1, mu = m_prior, Sigma = k_prior) # example draw from prior
plot_prior <- ggplot() +
    geom_point(aes(x = X_train, y = y_train)) +
    geom_line(aes(x = X_test, y = prior_f)) +
    labs(title = "", x = "", y = "") +
    theme_minimal()

# Posterior
K <- cov_matrix(X_train, X_train, kernel, sigma = amplitude, l = length_scale) # K = k(X, X)

# posterior mean
kalman_gain <- cov_matrix(X_test, X_train, kernel, sigma = amplitude, l = length_scale) %*% solve((K + sigma_noise^2 * diag(length(X_train))))
m_post <- m_prior + kalman_gain %*% (y_train - matrix(0, length(X_train), 1))

# posterior variance
k_post <- k_prior - kalman_gain %*% cov_matrix(X_train, X_test, kernel, sigma = amplitude, l = length_scale)
ci_95 <- sqrt(diag(k_post))*1.96 # confidence intervals
# mvrnorm(n = 1, mu = m, Sigma = K)
post_f <- mvrnorm(n = 1, mu = m_post, Sigma = k_post) # example draw
plot_post <- ggplot() +
    geom_point(aes(x = X_train, y = y_train)) +
    geom_line(aes(x = X_test, y = post_f)) +
    geom_ribbon(aes(x=X_test, ymin=post_f-ci_95, ymax=post_f+ci_95),fill="blue", alpha=0.2) +
    labs(title = "", x = "", y = "") +
    theme_minimal()