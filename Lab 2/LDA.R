# Theo AO Rashid -- January 2020

# ----- Machine Learning: Lab 2 -----
# Linear Discriminant Analysis (LDA)
library(MASS)
set.seed(1)

# Iris dataset in the form:
# Sepal.Length | Sepal.Width | Petal.Length | Petal.Width | Species
iris.data=iris

# Classification task: Classify if flower is virginica (TRUE) or not (FALSE)
iris.data$Species <- iris.data$Species == "virginica"

# Pairs produces a matrix of scatterplots
# pairs(iris[,1:4],col=iris$Species) # plot for all species
# pairs(iris.data[,1:4],col=iris.data$Species+1) # plot for virginica or not

# Divide the data into test and training (here 60% train, 40% test)
training_sample <- sample(c(TRUE, FALSE), nrow(iris.data), replace = T, prob = c(0.6,0.4))
train <- iris.data[training_sample, ]
test <- iris.data[!training_sample, ]

# Fit LDA to training set
# lda function from MASS package
# "Species vs everything else, please fit the training data"
iris.lda <- lda(Species ~ ., train)

# Evaluate on the training set
lda.train <- predict(iris.lda)
train$lda <- lda.train$class
# table(train$lda,train$Species) # confusion matrix (predicted, columns)
# The model fits the training set fairly well (sum of diagonals correctly predicted)

# Generate the posterior probabilities (p(y=1|data) in the 2nd column)
train$lda_proba <- predict(iris.lda)$posterior[,2]

# Evaluate on the test set
lda.test <- predict(iris.lda, test)
test$lda <- lda.test$class
# table(test$lda,test$Species) # confusion matrix for test set
test$lda_proba <- predict(iris.lda, test)$posterior[,2]

# ROC curve
# Define a grid of thresholds
cvec <- seq(0.001,0.999,length=1000)
specif.lda <- numeric(length(cvec))
sensit.lda <- numeric(length(cvec))

# Compute the specificity and 1-sensitivity for all these thresholds
for (cc in 1:length(cvec)){
  sensit.lda[cc] <- sum( test$lda_proba> cvec[cc] & test$Species==T)/sum(test$Species==T)
  specif.lda[cc] <- sum( test$lda_proba<=cvec[cc] & test$Species==F)/sum(test$Species==F)
}
# plot(1-specif.lda,sensit.lda,xlab="Specificity",ylab="Sensitivity",type="l",lwd=2)