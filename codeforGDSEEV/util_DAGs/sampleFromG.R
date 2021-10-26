sampleFromG <- function(n,G,B,noiseVariances)
# requires MASS
# Copyright (c) 2010 - 2012  Jonas Peters  [peters@stat.math.ethz.ch]
# All rights reserved.  See the file COPYING for license terms. 
{
    p <- length(noiseVariances)
    SigmaN <- diag(noiseVariances)
    Id <- diag(rep(1,p))
    SigmaX <- solve(Id-B) %*% SigmaN %*% solve(t(Id - B))
    samples <- mvrnorm(n, rep(0,p), SigmaX)
    return(samples)
}
