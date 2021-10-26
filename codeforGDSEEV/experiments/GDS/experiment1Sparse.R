# Copyright (c) 2010 - 2013  Jonas Peters  [peters@stat.math.ethz.ch]
# All rights reserved.  See the file COPYING for license terms. 

nVec <- c(100,500,1000)
pVec <- c(5,20,40)
numExp <- 100

timeGDS <- 0
timeGES <- 0
timePC <- 0
timeRand <- 0

ptmTotal <- proc.time()


#pars <- list(regr.method = train_linear, regr.pars = list(), indtest.method = indtestHsic, indtest.pars = list())
pars <- list(regr.pars = list())


hdgdsDAGtmp <- rep(-1, length(numExp))
hdgdsDAG <- matrix(-1, length(nVec), length(pVec))
hdgdsDAGsd <- matrix(-1, length(nVec), length(pVec))
hdpcDAGtmp <- rep(-1, length(numExp))
hdpcDAG <- matrix(-1, length(nVec), length(pVec))
hdpcDAGsd <- matrix(-1, length(nVec), length(pVec))
hdgesDAGtmp <- rep(-1, length(numExp))
hdgesDAG <- matrix(-1, length(nVec), length(pVec))
hdgesDAGsd <- matrix(-1, length(nVec), length(pVec))
hdrandDAGtmp <- rep(-1, length(numExp))
hdrandDAG <- matrix(-1, length(nVec), length(pVec))
hdrandDAGsd <- matrix(-1, length(nVec), length(pVec))

hdgdsCPDAGtmp <- rep(-1, length(numExp))
hdgdsCPDAG <- matrix(-1, length(nVec), length(pVec))
hdgdsCPDAGsd <- matrix(-1, length(nVec), length(pVec))
hdpcCPDAGtmp <- rep(-1, length(numExp))
hdpcCPDAG <- matrix(-1, length(nVec), length(pVec))
hdpcCPDAGsd <- matrix(-1, length(nVec), length(pVec))
hdgesCPDAGtmp <- rep(-1, length(numExp))
hdgesCPDAG <- matrix(-1, length(nVec), length(pVec))
hdgesCPDAGsd <- matrix(-1, length(nVec), length(pVec))
hdrandCPDAGtmp <- rep(-1, length(numExp))
hdrandCPDAG <- matrix(-1, length(nVec), length(pVec))
hdrandCPDAGsd <- matrix(-1, length(nVec), length(pVec))

for(nCounter in 1:length(nVec))
{
    n <- nVec[nCounter]
    for(pCounter in 1:length(pVec))
    {
        p <- pVec[pCounter]
        pCon <- 1.5/(p-1) #sparse => 0.75p edges
        
        noiseVar <- rep(1,p)
        
        for(expCounter in 1:numExp)
        {   
            cat("p =",p,", n =",n, ", exp = ", expCounter, "\r")
            trueG <- randomDAG(p,pCon)
            trueGCPDAG <- dag2cpdagAdj(trueG)
            trueB <- randomB(trueG,0.1,1,TRUE)
            X <- sampleFromG(n,trueG,trueB,noiseVar)     
            SigmaHat <- cov(X)
            
            
            #GDS
            ptm <- proc.time()
            resGDS <- GDS(X, scoreName = "SEMSEV", pars, check = "checkUntilFirstMinK", output = FALSE)$Adj
            timeGDS <- timeGDS + proc.time() - ptm
            hdgdsDAGtmp[expCounter] <- hammingDistance(trueG, resGDS)
            hdgdsCPDAGtmp[expCounter] <- hammingDistance(trueGCPDAG, dag2cpdagAdj(resGDS))
            
            #PC
            ptm <- proc.time()
            resPC <- pcWrap(X, 0.01, mmax = Inf)
            timePC <- timePC + proc.time() - ptm
            hdpcDAGtmp[expCounter] <- hammingDistance(trueG, resPC)
            hdpcCPDAGtmp[expCounter] <- hammingDistance(trueGCPDAG, resPC)

            #GES
            ptm <- proc.time()
            resGES <- trueG
            cat("Until today (28.08.2013) Chickering's GES is not in the pcalg package yet. Therefore, the GES estimate is set to the truth. Uncomment the next line to change that.\n")  
            #resGES <- gesWrap(X)$Adj
            timeGES <- timeGES + proc.time() - ptm
            hdgesDAGtmp[expCounter] <- hammingDistance(trueG, resGES)
            hdgesCPDAGtmp[expCounter] <- hammingDistance(trueGCPDAG, resGES)
            
            #RANDOM
            ptm <- proc.time()
            resRand <- randomDAG(p,runif(1))
            timeRand <- timeRand + proc.time() - ptm
            hdrandDAGtmp[expCounter] <- hammingDistance(trueG, resRand)      
            hdrandCPDAGtmp[expCounter] <- hammingDistance(trueGCPDAG, dag2cpdagAdj(resRand))      
        } 
        
        hdgdsDAG[nCounter,pCounter] <- mean(hdgdsDAGtmp)
        hdgdsDAGsd[nCounter,pCounter] <- sd(hdgdsDAGtmp)
        hdpcDAG[nCounter,pCounter] <- mean(hdpcDAGtmp)
        hdpcDAGsd[nCounter,pCounter] <- sd(hdpcDAGtmp)
        hdgesDAG[nCounter,pCounter] <- mean(hdgesDAGtmp)
        hdgesDAGsd[nCounter,pCounter] <- sd(hdgesDAGtmp)
        hdrandDAG[nCounter,pCounter] <- mean(hdrandDAGtmp)
        hdrandDAGsd[nCounter,pCounter] <- sd(hdrandDAGtmp)

        hdgdsCPDAG[nCounter,pCounter] <- mean(hdgdsDAGtmp)
        hdgdsCPDAGsd[nCounter,pCounter] <- sd(hdgdsDAGtmp)
        hdpcCPDAG[nCounter,pCounter] <- mean(hdpcDAGtmp)
        hdpcCPDAGsd[nCounter,pCounter] <- sd(hdpcDAGtmp)
        hdgesCPDAG[nCounter,pCounter] <- mean(hdgesDAGtmp)
        hdgesCPDAGsd[nCounter,pCounter] <- sd(hdgesDAGtmp)
        hdrandCPDAG[nCounter,pCounter] <- mean(hdrandDAGtmp)
        hdrandCPDAGsd[nCounter,pCounter] <- sd(hdrandDAGtmp)
    }
}

cat("======== \n")
cat("DAG results \n")
cat("======== \n")
cat("GDS\n")
cat(hdgdsDAG, " +- ", hdgdsDAGsd, "\n \n")
cat("PC\n")
cat(hdpcDAG, " +- ", hdpcDAGsd, "\n \n")
cat("GES\n")
cat(hdgesDAG, " +- ", hdgesDAGsd, "\n \n")
cat("random\n")
cat(hdrandDAG, " +- ", hdrandDAGsd, "\n\n")

cat("======== \n")
cat("CPDAG results \n")
cat("======== \n")
cat("GDS\n")
cat(hdgdsCPDAG, " +- ", hdgdsCPDAGsd, "\n \n")
cat("PC\n")
cat(hdpcCPDAG, " +- ", hdpcCPDAGsd, "\n \n")
cat("GES\n")
cat(hdgesCPDAG, " +- ", hdgesCPDAGsd, "\n \n")
cat("random\n")
cat(hdrandCPDAG, " +- ", hdrandCPDAGsd, "\n\n")


cat("======== \n")
cat("Time results \n")
cat("======== \n")
cat("GDS:\n")
cat(timeGDS[1],"\n \n")
cat("PC:\n")
cat(timePC[1],"\n \n")
cat("GES:\n")
cat(timeGES[1],"\n \n")
cat("random:\n")
cat(timeRand[1],"\n \n")
cat("------\n")
cat("Computing Time Total:\n")
cat((proc.time() - ptmTotal)[1],"\n \n")

save.image("./results/experiment1Sparse.RData")
cat("\n")
