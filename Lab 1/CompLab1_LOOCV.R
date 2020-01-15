CompLab1.LOOCV <- function(x, y, PolyOrder) {

# x and y are vectors containing the inputs and targets respectively
# PolyOrder is the order of polynomial model used for fitting
 
NumOfDataPairs <- length(x)

# First construct design matrix of given order
X      <- rep(1, NumOfDataPairs)
dim(X) <- c(NumOfDataPairs, 1)

for (n in 1:PolyOrder){
    X = cbind(X, x^n)
}
 
# Initialise CV variable for storing results
CV = matrix(nrow=NumOfDataPairs, ncol=1)

for (n in 1:NumOfDataPairs){
	
	 # Create training design matrix and target data, leaving one out each time
	Train_X <- X[-n, ]
   	Train_y <- y[-n]
   
	# Create testing design matrix and target data
    Test_X <- X[n, ]
    Test_y <- y[n]

	# Learn the optimal paramerers using MSE loss
    
    Paras_hat <- solve( t(Train_X) %*% Train_X , t(Train_X) %*% Train_y)
    Pred_y    <- Test_X %*% Paras_hat;
    
    # Calculate the MSE of prediction using training data
    CV[n]     <- (Pred_y - Test_y)^2

}

Mean_CV <- mean(CV)
SD_CV   <- sd(CV)

print(Mean_CV)
print(SD_CV)

# Concatenate x and y into a single matrix and return
Results <- cbind(Mean_CV, SD_CV)

return(Results)
}