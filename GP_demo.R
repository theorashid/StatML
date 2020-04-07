# Theo AO Rashid -- March 2020

# ----- Machine Learning: Gaussian processes -----
# Demo for group presentation

library(MASS)
library(ggplot2)
library(dplyr)

set.seed(1)

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

# ----- SIMULATED DEATH DATA -----
# Simulated deaths in Hammersmith and Fulham 2004-2017
mortality <- read.csv("/Users/tar15/Dropbox/SPH/Data/Unit Test/mortsim_hf.csv", row.names = 1)
mortality_m <- mortality %>%
    filter(sex == 1) %>%
    select(-sex) %>%
    select(age_group, age_group.id, deaths, population) %>% # focus only on age, deaths and populations
    group_by(age_group.id) %>%
    summarise(death_rates = sum(deaths)/sum(population)) # emperical death rates
mortality_m$age = c(0.5, 3, 7.5, 12.5, 17.5, 22.5, 27.5, 32.5, 37.5, 42.5, 47.5, 52.5, 57.5,
                    62.5, 67.5, 72.5, 77.5, 82.5, 87.5) # bin midpoints (although last is 85+)
mortality_m <- mortality_m[c(3,2)] # age and death rates only

plot_year_deaths <- ggplot(mortality_m, aes(x = age, y = death_rates)) +
    geom_point()

X_train <- mortality_m$age
y_train <- mortality_m$death_rates

X_test <- seq(0, 100, length.out = 501) # data around from age 0 to 100

# ----- GP inference -----
# Hyperparameters
kernel <- se_kernel # choose kernel
sigma_noise <- 0.01 # noise model
length_scale <- 30 # for SE/Matern kernel
amplitude <- 0.1 # for SE/Matern kernel

# Prior -- sample from GP(0, K(., .))
m_prior = matrix(0, length(X_test), 1) # flat prior choice
k_prior <- cov_matrix(X_test, X_test, kernel, sigma = amplitude, l = length_scale)

prior_f <- mvrnorm(n = 1, mu = m_prior, Sigma = k_prior) # example draw from prior
plot_prior <- ggplot() +
    geom_point(aes(x = X_train, y = y_train)) +
    geom_line(aes(x = X_test, y = prior_f)) +
    labs(title = "", x = "age", y = "death rate") +
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
    labs(title = "", x = "age", y = "death rate") +
    theme_minimal()