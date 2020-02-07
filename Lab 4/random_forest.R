# Theo AO Rashid -- February 2020

# ----- Machine Learning: Lab 4 -----
# Random Forests

# ----- Boston Housing Dataset -----
library(MASS)
data(Boston)
y <- Boston[,14]
x <- Boston[,1:13]
# pairs(Boston)

# Use the random forest library for analysis
library(randomForest)

# fit the random forest to the data
# Default parameters: mtry = p/3, ntree = 500, nodesize = 5, maxnodes = NULL
# rf <- randomForest(x, y)
rf <- randomForest(x, y, mtry = 10, ntree = 1000, nodesize = 2, maxnodes = NULL)
# print(rf)

# plot(predict(rf), y) # plot predicted (out-of-bag estimation) vs true
# abline(c(0,1),col=2)

# plot(predict(rf,newdata=x), y)