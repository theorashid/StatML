# Theo AO Rashid -- January 2020

# ----- Machine Learning: Lab 3 -----
# Principal Component Analysis (PCA)

# ----- US Arrests Dataset -----
USArrests <- read.csv("/Users/tar15/Documents/PhD/StatML/Lab 3/USArrests.csv")

USArrests <- USArrests[,2:5] # state column is non-numeric
states <- USArrests[,1]

# pairs(USArrests)

# Perform PCA
df <- scale(USArrests) # rescale the data
usarrests.pca <- princomp(df)
# summary(usarrests.pca) # relative contribution of each component
# plot(usarrests.pca) # plot of contribution of each component
# biplot(usarrests.pca, scale =1) # observations in the space of the two first principal components

# Perform PCA "by hand"
# Need to find the eigenvalues of S = X^T X
eval <- (eigen(t(df) %*% df)$value)
# sqrt(eval) # standard deviations of each component
evec <- eigen(t(df) %*% df)$vector
# evec

# ----- MNIST Dataset -----
mnist <- read.csv("http://wwwf.imperial.ac.uk/~sfilippi/Data/mnist_train.csv", header = F)

m <- colMeans(mnist[1:1000,-1])
mdata <- scale(mnist[1:1000,-1], center=T, scale=F)

# Perform PCA "by hand"
S <- t(mdata)%*% mdata
eval <- eigen(S)$value
evec <- eigen(S)$vector

# Visualise the amount of variance explained by each component
# plot(eval/sum(eval)*100, ylab="% of variance explained", xlab="principal components")

# function for graphically displaying a digit
visualise = function(vec, ...){
    image(matrix(as.numeric(vec),nrow=28)[,28:1], col=gray((255:0)/255), ...)
}

x <- mdata[1,] # Consider the first image

old_par <- par(mfrow=c(3,2))

# Digit looks better when using more components
visualise(x+m)
visualise(evec[,1:10]%*%t(evec[,1:10])%*%x+m) # using first 10 components
visualise(evec[,1:20]%*%t(evec[,1:20])%*%x+m)
visualise(evec[,1:50]%*%t(evec[,1:50])%*%x+m)
visualise(evec[,1:100]%*%t(evec[,1:100])%*%x+m)
visualise(evec[,1:200]%*%t(evec[,1:200])%*%x+m) # using first 200 components
# par(old_par)