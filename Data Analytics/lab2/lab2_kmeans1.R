#fragment
set.seed(123)
sim.xy <- function(n, mean, sd) cbind(rnorm(n, mean[1], sd[1]),rnorm(n, mean[2],sd[2]))
# generate three clouds of points, well separated in the 2D plane
xy <- rbind(sim.xy(100, c(0,0), c(.2,.2)),sim.xy(100, c(2.5,0), c(.4,.2)),sim.xy(100, c(1.25,.5), c(.3,.2)))
xy[1,] <- c(0,2)     # convert 1st obs. to an outlying value
#
km3 <- kmeans(xy, 3) # ask for three clusters
plot(xy, col=km3$cluster)
cex=2.0
points(km3$centers, pch=3)
#
km4 <- kmeans(xy, 4) # ask for four clusters
cex=1.0
plot(xy, col=km4$cluster)
cex=2.0
points(km4$centers, pch=3)
