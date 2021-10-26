oneStepCheckingUntilFirst <- function(State1, scoreName,pars,X,penFactor,checkDAGs)
# Copyright (c) 2010 - 2012  Jonas Peters  [peters@stat.math.ethz.ch]
# All rights reserved.  See the file COPYING for license terms. 
{
    p <- State1$p
    # collect all different neighbors
    index <- array(dim = c(p*(p-1),2))
    indexCount <- 0
    for(i in 1:p)
    {
        for(j in 1:(p-1))
        {
            # do not add sth on the diagonal
            j <- j + (j>(i-1))
            indexCount <- indexCount + 1
            index[indexCount,] <- c(i,j)
        }
    }

    # permute this list
    index <- index[sample(p*(p-1), replace = FALSE),]

    madeStep1 <- FALSE
    indexCount <- 0
    while( (madeStep1 == FALSE) & (indexCount < (p*(p-1))) )
    {    
        indexCount <- indexCount + 1
        i <- index[indexCount,1]
        j <- index[indexCount,2]

        candidateAdj <- State1$Adj
        candidateAdj[i,j] <- (candidateAdj[i,j] + 1) %% 2
        candidateAdj[j,i] <- 0
                    
        if(!containsCycle(candidateAdj))
        {
            checkDAGs <- checkDAGs + 1
            computeNewState <- get(paste("computeNewState", scoreName, sep = ""))
            newState <- computeNewState(State1,c(i,j),X,pars,penFactor)
            if(newState$Score < State1$Score)
            {
                State1 <- newState
                madeStep1 <- TRUE
            }
        }
    }
    return(list(State = State1, madeStep = madeStep1, checkDAGs = checkDAGs))
}


