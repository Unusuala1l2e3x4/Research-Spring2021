data("iris")
iris.dist <- dist(iris[, -5])
iris.mds <- cmdscale(iris.dist)
# iris$Species is the 5th column
c.chars <- c("*", "o", "+")[as.integer(iris$Species)]
# iris$Species is the 5th column
# KMEANSRESULT is the variable you used in your kmeans lab assignment for the return variable.
a.cols <- rainbow(3)[KMEANSRESULT$cluster]

plot(iris.mds, col = a.cols, pch = c.chars, xlab = "X", ylab = "Y")
plot of chunk unnamed-chunk-5

corr <- KMEANSRESULT$cluster == 4 - as.integer(iris$Species)
correct <- c("o", "x")[2 - corr]

plot(iris.mds, col = a.cols, pch = correct, xlab = "X", ylab = "Y")
