standardizeMat <- function(X)
# Copyright (c) 2012-2012  Jonas Peters [peters@stat.math.ethz.ch]
# All rights reserved.  See the file COPYING for license terms.
{
    for(i in 1:dim(X)[2])
    {
        X[,i] <- (X[,i] - mean(X[,i])) / sd(X[,i])        
    }
    return(X)
}
