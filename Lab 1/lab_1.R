# Theo AO Rashid -- January 2020

# ----- Machine Learning: Lab 1 -----

model <- function(nsamples) {
  for (i in 1:nsamples) {
      X_i = runif(1, min=0, max=100^2)
      epsilon = rnorm(1, 0, 100^2)
      Y_i = 5*X_i^3 - X_i^2 + X_i + epsilon
  }
  samples <- 
  return(samples)
}