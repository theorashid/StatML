# Theo AO Rashid -- January 2020

# ----- Machine Learning: Lab 2 -----
# Classification on the MNIST handwritten digits dataset

# 28x28 greyscale (value 0 to 255) images
mnist <- read.csv("/Users/tar15/Documents/PhD/StatML/Lab 2/mnist_train.csv")

# function for graphically displaying a digit
visualise = function(vec, ...){
    image(matrix(as.numeric(vec),nrow=28)[,28:1], col=gray((255:0)/255), ...)
}

# plot some digits
# old_par <- par(mfrow=c(2,2))
# for (i in 1:4) visualise(mnist[i,-1])
# par(old_par)

# Separate into test and training data
# convert first column to integer
idx = which(mnist[,1] <= 1) # only classify between 0 and 1 digits
mnist <- mnist[idx,]
s <- sample.int(floor(2*length(idx)/3))
train <- mnist[s,]
test <- mnist[-s,]
identical <- apply(train, 2, function(v){all(v==v[1])})
train <- train[,!identical]
test <- test[,!identical]

# ----- LDA ------
mnist.lda <- lda(label ~ ., train)

# Evaluate on the training set
lda.train <- predict(mnist.lda)
train$lda <- lda.train$class
# table(train$lda,train$label) # confusion matrix (predicted, columns)
# 8398/8443 correctly predicted

# Evaluate on the test set
lda.test <- predict(mnist.lda, test)
test$lda <- lda.test$class
# table(test$lda,test$label) # confusion matrix for test set
# 4202/4222 correctly predicted

# ----- Logistic regression ------
# Fit logistic regression to training set
mnist.logistic <- glm(label ~ ., family = binomial(logit), data=train)

# summary(iris.logistic) # see the inferred parameters of the model

# Evaluate on the training data to see if it fits reasonably
train$logistic <- predict(mnist.logistic, type="response")
# table(actual= train$label, predicted=train$logistic>=0.5) # threshold of 0.5

# Evaluate on the test data
test$logistic <- predict(mnist.logistic, test, type="response")
# table(actual= test$label, predicted=test$logistic>=0.5)
# 4203/4222 correctly predicted