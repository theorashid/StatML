CompLab1.Generate_Data <- function() {

# Standard deviation of the noise to be added
Noise_std <- 100
 
# Set up the input variable, 100 points between -5 and 5
x <- seq(-5, 5, by=10/(100-1))
 
# Calculate the true function and add some noise
y <- 5*x^3 - x^2 + x + Noise_std*rnorm(length(x), 0, 1)

# Concatenate x and y into a single matrix and return
dim(x) <- c(length(x), 1)
dim(y) <- c(length(y), 1)
Data <- cbind(x, y)

return(Data)
}