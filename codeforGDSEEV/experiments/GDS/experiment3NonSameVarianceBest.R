# Copyright (c) 2010 - 2013  Jonas Peters  [peters@stat.math.ethz.ch]
# All rights reserved.  See the file COPYING for license terms. 

n <- 500
p <- 10
#pCon <- 0.3 #non-sparse
#pCon <- 1.5/(p-1) #sparse => 0.75p edges
pCon <- 2/(p-1) #sparse => p edges
numExp <- 100
aVec <- c(0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9)
#aVec <- c(0.1,0.3,0.5,0.8)


hdsemDAG <- list(NULL)
hdsemScore <- list(NULL)
hdsemCPDAG <- list(NULL)
hdpcDAG <- list(NULL)
hdpcCPDAG <- list(NULL)
hdgesDAG <- list(NULL)
hdgesCPDAG <- list(NULL)
hdbestDAG <- list(NULL)
hdbestCPDAG <- list(NULL)
hdScoreDiff <- list(NULL)
hdgesScore <- list(NULL)

tmp1 <- rep(0,numExp)
tmp2 <- rep(0,numExp)
tmp2b <- rep(0,numExp)
tmp3 <- rep(0,numExp)
tmp4 <- rep(0,numExp)
tmp5 <- rep(0,numExp)
tmp6 <- rep(0,numExp)
tmp6b <- rep(0,numExp)
tmp7 <- rep(0,numExp)
tmp8 <- rep(0,numExp)
tmp9 <- rep(0,numExp)

meansemDAG <- rep(0,length(aVec))
sdsemDAG <- rep(0,length(aVec))
meanpcDAG <- rep(0,length(aVec))
sdpcDAG <- rep(0,length(aVec))
meangesDAG <- rep(0,length(aVec))
sdgesDAG <- rep(0,length(aVec))
meanbestDAG <- rep(0,length(aVec))
sdbestDAG <- rep(0,length(aVec))
meangesCPDAG <- rep(0,length(aVec))
sdgesCPDAG <- rep(0,length(aVec))
meansemCPDAG <- rep(0,length(aVec))
sdsemCPDAG <- rep(0,length(aVec))
meanpcCPDAG <- rep(0,length(aVec))
sdpcCPDAG <- rep(0,length(aVec))
meanbestCPDAG <- rep(0,length(aVec))
sdbestCPDAG <- rep(0,length(aVec))
SEMbetter <- rep(0,length(aVec))

for(aCounter in 1:length(aVec))
{
    a <- aVec[aCounter]  
    cat("a =",a,"\n")
    
    for(expCounter in 1:numExp)
    {
        noiseVar <- runif(p, min = 1-a, max = 1+a)
        
        trueG <- randomDAG(p,pCon)
        trueGCPDAG <- dag2cpdagAdj(trueG)
        trueB <- randomB(trueG,0.1,1,TRUE)
        X <- sampleFromG(n,trueG,trueB,noiseVar)     
        SigmaHat <- cov(X)
        pars <- list(SigmaHat = SigmaHat)
        
        #TRUTH
        stateTruth <- initializeSEMSEV(X,trueG,pars)
        #cat("Score of True DAG: ", stateTruth$Score, "\n")
            
        #GDS
        resGDS <- GDS(X, scoreName = "SEMSEV", pars, check = "checkUntilFirstMinK", output = FALSE)
        tmp1[expCounter] <- hammingDistance(trueG, resGDS$Adj)
        tmp2[expCounter] <- hammingDistance(trueGCPDAG, dag2cpdagAdj(resGDS$Adj))
        tmp2b[expCounter] <- resGDS$Score 
        #cat("Score of GDSSEV DAG: ", resSEM$Score, "\n")
        
        #PC
        resPC <- pcWrap(X, 0.01, mmax = Inf)
        tmp3[expCounter] <- hammingDistance(trueG, resPC)
        tmp4[expCounter] <- hammingDistance(trueGCPDAG, resPC) 
        
        #GES
        resGES <- resGDS
        cat("Until today (28.08.2013) Chickering's GES is not in the pcalg package yet. Therefore, the GES estimate is set to the truth. Uncomment the next line to change that.\n")  
        #resGES <- gesWrap(X)
        tmp5[expCounter] <- hammingDistance(trueG, resGES$Adj)
        tmp6[expCounter] <- hammingDistance(trueGCPDAG, resGES$Adj) 
        tmp6b[expCounter] <- resGES$Score 
        #cat("Score of GES DAG: ", resGES$Score, "\n")
        
        #BEST OF GDS AND GES
        tmp9[expCounter] <- resGDS$Score - resGES$Score
        if(resGES$Score < resGDS$Score)
        {
            tmp7[expCounter] <- hammingDistance(trueG, resGES$Adj)
            tmp8[expCounter] <- hammingDistance(trueGCPDAG, resGES$Adj)
        }
        if(resGES$Score >= resGDS$Score)
        {
            tmp7[expCounter] <- hammingDistance(trueG, resGDS$Adj)
            tmp8[expCounter] <- hammingDistance(trueGCPDAG, dag2cpdagAdj(resGDS$Adj))
        }
        
    }
    hdsemDAG[[aCounter]] <- tmp1
    hdsemCPDAG[[aCounter]] <- tmp2
    hdsemScore[[aCounter]] <- tmp2b
    hdpcDAG[[aCounter]] <- tmp3
    hdpcCPDAG[[aCounter]] <- tmp4
    hdgesDAG[[aCounter]] <- tmp5
    hdgesCPDAG[[aCounter]] <- tmp6
    hdgesScore[[aCounter]] <- tmp6b 
    hdbestDAG[[aCounter]] <- tmp7 
    hdbestCPDAG[[aCounter]] <- tmp8
    hdScoreDiff[[aCounter]] <- tmp9 
}

for(aCounter in 1:length(aVec))
{
    
    meansemDAG[aCounter] <- mean(hdsemDAG[[aCounter]])
    sdsemDAG[aCounter] <- sd(hdsemDAG[[aCounter]])
    meanpcDAG[aCounter] <- mean(hdpcDAG[[aCounter]])
    sdpcDAG[aCounter] <- sd(hdpcDAG[[aCounter]])
    meangesDAG[aCounter] <- mean(hdgesDAG[[aCounter]])
    sdgesDAG[aCounter] <- sd(hdgesDAG[[aCounter]])
    meanbestDAG[aCounter] <- mean(hdbestDAG[[aCounter]])
    sdbestDAG[aCounter] <- sd(hdbestDAG[[aCounter]])
    
    meansemCPDAG[aCounter] <- mean(hdsemCPDAG[[aCounter]])
    sdsemCPDAG[aCounter] <- sd(hdsemCPDAG[[aCounter]])
    meanpcCPDAG[aCounter] <- mean(hdpcCPDAG[[aCounter]])
    sdpcCPDAG[aCounter] <- sd(hdpcCPDAG[[aCounter]])
    meangesCPDAG[aCounter] <- mean(hdgesCPDAG[[aCounter]])
    sdgesCPDAG[aCounter] <- sd(hdgesCPDAG[[aCounter]])
    meanbestCPDAG[aCounter] <- mean(hdbestCPDAG[[aCounter]])
    sdbestCPDAG[aCounter] <- sd(hdbestCPDAG[[aCounter]])
    
    SEMbetter[aCounter] <- sum(hdScoreDiff[[aCounter]]<0)
    
    cat("\n \n alpha = ",aVec[aCounter]," \n")
    cat("\n DAG\n")
    cat("PC:", mean(hdpcDAG[[aCounter]]), "+-", sd(hdpcDAG[[aCounter]]),"\n")
    cat("GDS:", mean(hdsemDAG[[aCounter]]), "+-", sd(hdsemDAG[[aCounter]]),"\n")
    cat("GES:", mean(hdgesDAG[[aCounter]]), "+-", sd(hdgesDAG[[aCounter]]),"\n")
    cat("BEST:", mean(hdbestDAG[[aCounter]]), "+-", sd(hdbestDAG[[aCounter]]),"\n")
    
    cat("CPDAG\n")    
    cat("PC:", mean(hdpcCPDAG[[aCounter]]), "+-", sd(hdpcCPDAG[[aCounter]]),"\n")
    cat("GDS:", mean(hdsemCPDAG[[aCounter]]), "+-", sd(hdsemCPDAG[[aCounter]]),"\n")
    cat("GES:", mean(hdgesCPDAG[[aCounter]]), "+-", sd(hdgesCPDAG[[aCounter]]),"\n")
    cat("BEST:", mean(hdbestCPDAG[[aCounter]]), "+-", sd(hdbestCPDAG[[aCounter]]),"\n")

    cat("GDS better:", SEMbetter[aCounter],"\n")
    
}


save.image("./results/experiment3NonSameVarianceBest.RData")

