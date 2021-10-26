# Copyright (c) 2010 - 2013  Jonas Peters  [peters@stat.math.ethz.ch]
# All rights reserved.  See the file COPYING for license terms. 

n <- 500
numExp <- 100

X <- matrix(0,n,3)
trueG <- rbind(c(0,1,1),c(0,0,1),c(0,0,0))
trueGCPDAG <- dag2cpdagAdj(trueG)

hdpcDAG <- rep(0,numExp)
hdpcCPDAG <- rep(0,numExp)
hdgdsDAG <- rep(0,numExp)
hdgdsCPDAG <- rep(0,numExp)
hdgesDAG <- rep(0,numExp)
hdgesCPDAG <- rep(0,numExp)

pars <- list(regr.pars = list())

for(expCounter in 1:numExp)
{   
    X[,1] <- rnorm(n) 
    X[,2] <- -1*X[,1] + rnorm(n)
    X[,3] <- 1*X[,2] + 1*X[,1] + rnorm(n)
    SigmaHat <- cov(X)
        
    #GDS
    resGDS <- GDS(X, scoreName = "SEMSEV", pars, "checkUntilFirstMinK")$Adj
    hdgdsDAG[expCounter] <- hammingDistance(trueG, resGDS)
    hdgdsCPDAG[expCounter] <- hammingDistance(trueGCPDAG, dag2cpdagAdj(resGDS))
            
    #PC
    resPC <- pcWrap(X, 0.01, mmax = Inf)
    hdpcDAG[expCounter] <- hammingDistance(trueG, resPC)
    hdpcCPDAG[expCounter] <- hammingDistance(trueGCPDAG, resPC)    

    #GES
    resGES <- trueG
    cat("Until today (28.08.2013) Chickering's GES is not in the pcalg package yet. Therefore, the GES estimate is set to the truth. Uncomment the next line to change that.\n")  
    # resGES <- gesWrap(X)
    hdgesDAG[expCounter] <- hammingDistance(trueG, resGES)
    hdgesCPDAG[expCounter] <- hammingDistance(trueGCPDAG, resGES)    

    #TRUTH
#    stateTruth <- initialize(X,trueG,SigmaHat)
#    cat("Score of True DAG: ", stateTruth$score, "\n")
} 

cat("GDS:", mean(hdgdsDAG), "+-", sd(hdgdsDAG),"\n")
cat("PC:", mean(hdpcDAG), "+-", sd(hdpcDAG),"\n")
cat("GES:", mean(hdgesDAG), "+-", sd(hdgesDAG),"\n")


save.image("./results/experiment2NonFaithful.RData")
