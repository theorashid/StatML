# Theo AO Rashid -- January 2020

# ----- Machine Learning: Lab 3 -----
# Clustering

# ----- K-means -----
USArrests <- read.csv("/Users/tar15/Documents/PhD/StatML/Lab 3/USArrests.csv")

USArrests <- USArrests[,2:5] # state column is non-numeric
states <- USArrests[,1]

df <- scale(USArrests) # rescale the data

# K = 4 clusters
set.seed(1)
km.res <- kmeans(df, centers=4)
# km.res

# library(ggfortify)
# autoplot(km.res, data = df) # Plot on first two principal components (to visualise)

# ----- Hierarchical clustering -----
dist_data <- dist(df, method = 'euclidean') # distance between each observation

hdata <- hclust(dist_data)
# plot(hdata) # plot the dendogram

alloc <- cutree(hdata, h=3.75) # cut the dendogram at 3.75 (chosen) to obtain clusters