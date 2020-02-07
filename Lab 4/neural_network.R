# Theo AO Rashid -- February 2020

# ----- Machine Learning: Lab 4 -----
# Neural Networks
set.seed(1)

# ----- Boston Housing Dataset -----
library(MASS)
data <- Boston
index <- sample(1:nrow(data),round(0.75*nrow(data)))
train <- data[index,] # split into 75-25 train-test
test <- data[-index,]

# Scale the data between [0,1] using min-max method
maxs <- apply(data, 2, max) 
mins <- apply(data, 2, min)

scaled <- as.data.frame(scale(data, center = mins, scale = maxs - mins))

train_ <- scaled[index,]
test_ <- scaled[-index,]

# Use the neural network package for analysis
library(neuralnet)

# Train a multilayer perception
n <- names(train_)
f <- as.formula(paste("medv ~", paste(n[!n %in% "medv"], collapse = " + ")))
# Start with fully connected network with two hidden layers (5 and 3 neurons)
# Output layer has one value for regression
nn <- neuralnet(f,data=train_,hidden=c(5,3),linear.output=T)
# nn <- neuralnet(f,data=train_,hidden=c(5,6,3),linear.output=T) # alternative architecture
# plot(nn)

# Make a prediction for the test set
pr.nn <- compute(nn,test_[,1:13])

# Rescale the normalised predictions
pr.nn_ <- pr.nn$net.result*(max(data$medv)-min(data$medv))+min(data$medv)
test.r <- (test_$medv)*(max(data$medv)-min(data$medv))+min(data$medv)

# Calculate the MSE
MSE.nn <- sum((test.r - pr.nn_)^2)/nrow(test_)
# print(MSE.nn)

# Plot the prediction values as a function of the real values
plot(test$medv,pr.nn_,col='red',main='Real vs predicted NN',pch=18,cex=0.7)
abline(0,1,lwd=2)
legend('bottomright',legend='NN',pch=18,col='red', bty='n')