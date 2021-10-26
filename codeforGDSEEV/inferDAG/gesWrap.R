gesWrap <- function(X)
# Copyright (c) 2012-2012  Jonas Peters [peters@stat.math.ethz.ch]
# All rights reserved.  See the file COPYING for license terms.
{
    result <- list()
#    score <- new("gauss.l0pen.obs.score", X)
    score <- new("GaussL0penObsScore", X)
    G <- ges(ncol(X), score)
    result$Adj <- as(G$essgraph, "matrix")
   # result$Score <- -(G$essgraph$score$global.score(G$repr) - log((2*pi)^dim(X)[2])* dim(X)[1]/2)
    result$Score <- computeScoreSEMGauss(X,as(G$repr,"matrix"))
    return(result)
}

