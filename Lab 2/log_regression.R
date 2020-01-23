# Theo AO Rashid -- January 2020

# ----- Machine Learning: Lab 2 -----
# Logistic Regression
library(MASS)
set.seed(1)

# Set up the iris data as in the LDA exercise
iris.data=iris
iris.data$Species <- iris.data$Species == "virginica"

training_sample <- sample(c(TRUE, FALSE), nrow(iris.data), replace = T, prob = c(0.6,0.4))
train <- iris.data[training_sample, ]
test <- iris.data[!training_sample, ]

# Fit logistic regression to training set
iris.logistic <- glm(Species ~ ., family = binomial(logit), data=train)

# summary(iris.logistic) # see the inferred parameters of the model

# Evaluate on the training data to see if it fits reasonably
train$logistic <- predict(iris.logistic, type="response")
# table(actual= train$Species, predicted=train$logistic>=0.5) # threshold of 0.5

# Evaluate on the test data
test$logistic <- predict(iris.logistic, test, type="response")
# table(actual= test$Species, predicted=test$logistic>=0.5) # only 3 false negatives

# ROC curve
cvec <- seq(0.001,0.999,length=1000)
specif.lr <- numeric(length(cvec))
sensit.lr <- numeric(length(cvec))

for (cc in 1:length(cvec)){
  sensit.lr[cc] <- sum( test$logistic> cvec[cc] & test$Species==T)/sum(test$Species==T)
  specif.lr[cc] <- sum( test$logistic<=cvec[cc] & test$Species==F)/sum(test$Species==F)
}
# lines(1-specif.lr,sensit.lr,col="red",lwd=2)