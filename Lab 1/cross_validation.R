# Theo AO Rashid -- January 2020

# ----- Machine Learning: Lab 1 -----

# 1. Generate Data from the model Y_i = 5X_i^3 - X_i^2 + X_i + noise

Generate_Data <- function() {
  # Standard deviation of the noise to be added
  noise_std <- 100
  # Set up the input variable, 100 points between -5 and 5
  x <- seq(-5, 5, by=10/(100-1))
  # Calculate the true function and add some noise
  y <- 5*x^3 - x^2 + x + noise_std*rnorm(length(x), 0, 1)
  # Concatenate x and y into a single matrix and return
  dim(x) <- c(length(x), 1)
  dim(y) <- c(length(y), 1)
  Data <- data.frame(x=x,y=y)

  return(Data)
}

set.seed(1)
data<-Generate_Data()
# plot(data$x,data$y)

# 2. Compute the MSE for a given order of polynomial

MSE <- function(x, y, PolyOrder) {
  # x and y are vectors containing the inputs and targets respectively
  # PolyOrder is the order of polynomial model used for fitting
 
  NumOfDataPairs <- length(x)

  # First construct design matrix of given order
  X      <- rep(1, NumOfDataPairs) # first row x^0 = 1
  dim(X) <- c(NumOfDataPairs, 1)
  # Design matrix of each element in x to the polynomial degree (row)
  for (n in 1:PolyOrder){
    X = cbind(X, x^n)
  }
  Train_X = X
  Train_y = y
  # Solve Paras_hat = (X^T*X)^(-1)*(X^T*y)
  # i.e.  (X^T*X)*Paras_hat = (X^T*y)
  # %*% is matrix multiplication
  Paras_hat <- solve( t(Train_X) %*% Train_X , t(Train_X) %*% Train_y)
  
  Test_X = X
  Test_y = y 

  # Pred_y = Test_X * Paras_hat
  Pred_y  <- Test_X %*% Paras_hat
  MSE <- mean((Pred_y - Test_y)^2)

  return(MSE)
}

# Plot for polynomials up to n=8
# plot(seq(8), sapply(seq(1:8), function(i) MSE(data$x,data$y, i)), ylab = "MSE", xlab="order", type="l")
# Big drop at n=3 (appropriate for the model given)

# 3. Edit the function for 80%/20% test/train split

MSE <- function(x, y, PolyOrder) {
  # x and y are vectors containing the inputs and targets respectively
  # PolyOrder is the order of polynomial model used for fitting
 
  NumOfDataPairs <- length(x)

  X      <- rep(1, NumOfDataPairs)
  dim(X) <- c(NumOfDataPairs, 1)
  
  for (n in 1:PolyOrder){
    X = cbind(X, x^n)
  }

  set.seed(1)
  s <- sample.int(NumOfDataPairs,floor(NumOfDataPairs*0.8)) # 80-20 split
  Train_X = X[s, ] # 80% training
  Train_y = y[s]
  
  Paras_hat <- solve( t(Train_X) %*% Train_X , t(Train_X) %*% Train_y)
  
  Test_X = X[-s, ] # 20% test
  Test_y = y [-s]

  Pred_y  <- Test_X %*% Paras_hat

  MSE_test <- mean((Pred_y - Test_y)^2)
  MSE_train <- mean((Train_X %*% Paras_hat - Train_y)^2)

  return(data.frame(MSE_train,MSE_test))
}

# plot(seq(1:8), sapply(seq(1:8), function(i) MSE(data$x,data$y, i)$MSE_train), ylab = "MSE", xlab="order", type="l", main ="On train set")
# plot(seq(1:8), sapply(seq(1:8), function(i) MSE(data$x,data$y, i)$MSE_test), ylab = "MSE", xlab="order", type="l", main="On test set")

# 4. MSE for LOOCV procedure

source("/Users/tar15/Documents/PhD/StatML/Lab 1/LOOCV.R") # get the LOOCV function, which was provided
# plot(seq(1:8), sapply(seq(1:8), function(i) LOOCV(data$x,data$y, i)$Mean_CV), ylab = "average MSE", xlab="order", type="l")
# Again, the function picks out n=3 as the optimum for the model

# 5. CV with regularisation (lambda)

LOOCV_lambda <- function(X,y,lambda){
  # X is the matrix of features (including polynomials)
  # y is a vector containig the targets
  # Lambda is the parameter for ridge regression
 
  NumOfDataPairs <- length(y)

  # Initialise CV variable for storing results
  CV = matrix(nrow=NumOfDataPairs, ncol=1)

  for (n in 1:NumOfDataPairs){
    # Create training design matrix and target data, leaving one out each time
	  Train_X <- X[-n, ]
   	Train_y <- y[-n]
   
	  # Create testing design matrix and target data
    Test_X <- X[n, ]
    Test_y <- y[n]

	  # Learn the optimal paramerers using MSE loss with regularisation parameter
    # Solve Paras_hat = (X^T*X + l*I)^(-1)*(X^T*y)
    # i.e.  (X^T*X + l*I)*Paras_hat = (X^T*y)
    Paras_hat <- solve( t(Train_X) %*% Train_X + lambda*diag(dim(X)[2]) , t(Train_X) %*% Train_y)
    Pred_y    <- Test_X %*% Paras_hat;
    
    # Calculate the MSE of prediction using training data
    CV[n]     <- (Pred_y - Test_y)^2
  }
  Mean_CV <- mean(CV)
  SD_CV   <- sd(CV)
  
  print(Mean_CV)
  print(SD_CV)

return(Mean_CV)
}

# Investigate the fitting as a function of lambda
x <-data$x
y <-data$y
PolyOrder <-5

NumOfDataPairs <- length(x)

# First construct design matrix of given order
X      <- rep(1, NumOfDataPairs)
dim(X) <- c(NumOfDataPairs, 1)

for (n in 1:PolyOrder){
    X = cbind(X, x^n)
}

s <- sample.int(NumOfDataPairs,floor(NumOfDataPairs*0.8)) # 80-20 split
Train_X = X[s,]
Train_y = y[s]

lambdalist <- 2^seq(0,15,by=0.1)

# Find lambda using LOOCV
MSE_lambda=sapply(lambdalist, function(lambda) LOOCV_lambda(Train_X,Train_y,lambda))
# plot(lambdalist,MSE_lambda, xlab="lambda", ylab="MSE", type="l") 
lambda <-lambdalist[which.min(MSE_lambda)] #lambda that minimises MSE error

Paras_hat <- solve( t(Train_X) %*% Train_X , t(Train_X) %*% Train_y)
Paras_hat_lambda <- solve( t(Train_X) %*% Train_X + lambda*diag(dim(Train_X)[2]) , t(Train_X) %*%Train_y)

Pred_y  <- X %*% Paras_hat
Pred_y_lambda  <- X %*% Paras_hat_lambda

# Both predictions seem to be a good fit
plot(data$x,data$y)
lines(data$x,Pred_y, col="red")
lines(data$x,Pred_y_lambda, col="green")

# Find lambda using 80-20 training split
Test_X = X[-s,]
Test_y = y[-s]
Pred_test_y  <- Test_X %*% Paras_hat
Pred_test_y_lambda  <- Test_X %*% Paras_hat_lambda # Test predictions are similar
MSE<- mean((Pred_test_y-Test_y)^2)
MSE_lambda<- mean((Pred_test_y_lambda-Test_y)^2)