# Copyright (c) 2010 - 2013  Jonas Peters  [peters@stat.math.ethz.ch]
# All rights reserved.  See the file COPYING for license terms. 
n <- 500

x <- rnorm(n)
y <- x + rnorm(n)
z <- y + rnorm(n)

X <- cbind(x,y,z)

truth <- cbind(c(0,0,0),c(1,0,0),c(0,1,0))
cat("true DAG:\n")
show(truth)


resGDS <- GDS(X)
cat("estimated DAG by GDS with equal error variances:\n")
show(resGDS$Adj)
cat("BIC score of the estimated DAG:")
show(resGDS$Score)


