dag2cpdagAdj <- function(Adj)
# Copyright (c) 2010 - 2012  Jonas Peters  [peters@stat.math.ethz.ch]
# All rights reserved.  See the file COPYING for license terms. 
{
    d <- as(Adj, "graphNEL")
    cpd <- dag2cpdag(d)
    result <- as(cpd, "matrix")
    return(result)
}
