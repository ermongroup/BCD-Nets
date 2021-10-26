# Copyright (c) 2010 - 2013  Jonas Peters  [peters@stat.math.ethz.ch]
# All rights reserved.  See the file COPYING for license terms. 


load("./results/experiment3NonSameVarianceBest.RData")



pdf(file  = "./results/experimentNonSameVarianceBestDAG.pdf", 
    width = 13, height = 4)
par(cex.lab=1, las = 1)
par(cex.axis=0.8)
boxplot(hdsemDAG[[1]],hdgesDAG[[1]],hdpcDAG[[1]],hdbestDAG[[1]], 
        hdsemDAG[[2]],hdgesDAG[[2]],hdpcDAG[[2]],hdbestDAG[[2]],
        hdsemDAG[[3]],hdgesDAG[[3]],hdpcDAG[[3]],hdbestDAG[[3]],
        hdsemDAG[[4]],hdgesDAG[[4]],hdpcDAG[[4]],hdbestDAG[[4]],
        hdsemDAG[[5]],hdgesDAG[[5]],hdpcDAG[[5]],hdbestDAG[[5]],
        hdsemDAG[[6]],hdgesDAG[[6]],hdpcDAG[[6]],hdbestDAG[[6]],
        hdsemDAG[[7]],hdgesDAG[[7]],hdpcDAG[[7]],hdbestDAG[[7]],
        hdsemDAG[[8]],hdgesDAG[[8]],hdpcDAG[[8]],hdbestDAG[[8]],
        hdsemDAG[[9]],hdgesDAG[[9]],hdpcDAG[[9]],hdbestDAG[[9]],
        hdsemDAG[[10]],hdgesDAG[[10]],hdpcDAG[[10]],hdbestDAG[[10]],
        at = c(1,2,3,4, 6,7,8,9, 11,12,13,14, 16,17,18,19, 21,22,23,24, 26,27,28,29, 31,32,33,34, 36,37,38,39, 41,42,43,44, 46,47,48,49),
        xlab="a",
        ylab="Structural Hamming distance (to DAG)",
        col = c(0,gray(0.8),gray(0.55),gray(0.3)),
        names = c("","0","","",  "","0.1","","",  "","0.2","","",  "","0.3","","",  "","0.4","","",  "","0.5","","",  "","0.6","","",
                  "","0.7","","",  "","0.8","","",  "","0.9","",""))
        
#legend(40,30, # places a legend at the appropriate place 
#       c("GDS_SEV","GES", "PC", "BEST_SCORE"), # puts text in the legend
#       lty=c(0,0,0,0), # gives the legend appropriate symbols (lines)
#       #pch=c(22,22,22), # gives the legend appropriate symbols (lines)
#       #lwd=c(2.5,2.5,2.5),
#       #pt.bg=c(0,gray(0.7),gray(0.3)),
#       #col=c(1,1,1))
#       bg = c(gray(1)),
#       fill = c(0,gray(0.8),gray(0.55),gray(0.3)))
dev.off()   


pdf(file  = "./results/experimentNonSameVarianceDAG.pdf", 
    width = 13, height = 4)

boxplot(hdsemDAG[[1]],hdgesDAG[[1]],hdpcDAG[[1]], 
        hdsemDAG[[2]],hdgesDAG[[2]],hdpcDAG[[2]],
        hdsemDAG[[3]],hdgesDAG[[3]],hdpcDAG[[3]],
        hdsemDAG[[4]],hdgesDAG[[4]],hdpcDAG[[4]],
        hdsemDAG[[5]],hdgesDAG[[5]],hdpcDAG[[5]],
        hdsemDAG[[6]],hdgesDAG[[6]],hdpcDAG[[6]],
        hdsemDAG[[7]],hdgesDAG[[7]],hdpcDAG[[7]],
        hdsemDAG[[8]],hdgesDAG[[8]],hdpcDAG[[8]],
        hdsemDAG[[9]],hdgesDAG[[9]],hdpcDAG[[9]],
        hdsemDAG[[10]],hdgesDAG[[10]],hdpcDAG[[10]],
        at = c(1,2,3, 5,6,7, 9,10,11, 13,14,15, 17,18,19, 21,22,23, 25,26,27, 29,30,31, 33,34,35, 37,38,39),
        col = c(0,gray(0.7),gray(0.3)),
        xlab="a",
        ylab="Structural Hamming distance (to DAG)",
        names = c("","0","","","0.1","","","0.2","","","0.3","","","0.4","","","0.5","","","0.6","",
                  "","0.7","","","0.8","","","0.9",""))

#legend(35,22, # places a legend at the appropriate place 
#       c("GDS_SEV","GES", "PC"), # puts text in the legend
#       lty=c(0,0,0), # gives the legend appropriate symbols (lines)
#       #pch=c(22,22,22), # gives the legend appropriate symbols (lines)
#       #lwd=c(2.5,2.5,2.5),
#       #pt.bg=c(0,gray(0.7),gray(0.3)),
#       #col=c(1,1,1))
#       bg = c(gray(1)),
#       fill = c(0,gray(0.7),gray(0.3)))
dev.off()   


pdf(file  = "./results/experimentNonSameVarianceBestCPDAG.pdf", 
    width = 13, height = 4)
par(cex.axis=0.8, las = 1)
boxplot(hdsemCPDAG[[1]],hdgesCPDAG[[1]],hdpcCPDAG[[1]],hdbestCPDAG[[1]], 
        hdsemCPDAG[[2]],hdgesCPDAG[[2]],hdpcCPDAG[[2]],hdbestCPDAG[[2]],
        hdsemCPDAG[[3]],hdgesCPDAG[[3]],hdpcCPDAG[[3]],hdbestCPDAG[[3]],
        hdsemCPDAG[[4]],hdgesCPDAG[[4]],hdpcCPDAG[[4]],hdbestCPDAG[[4]],
        hdsemCPDAG[[5]],hdgesCPDAG[[5]],hdpcCPDAG[[5]],hdbestCPDAG[[5]],
        hdsemCPDAG[[6]],hdgesCPDAG[[6]],hdpcCPDAG[[6]],hdbestCPDAG[[6]],
        hdsemCPDAG[[7]],hdgesCPDAG[[7]],hdpcCPDAG[[7]],hdbestCPDAG[[7]],
        hdsemCPDAG[[8]],hdgesCPDAG[[8]],hdpcCPDAG[[8]],hdbestCPDAG[[8]],
        hdsemCPDAG[[9]],hdgesCPDAG[[9]],hdpcCPDAG[[9]],hdbestCPDAG[[9]],
        hdsemCPDAG[[10]],hdgesCPDAG[[10]],hdpcCPDAG[[10]],hdbestCPDAG[[10]],
        at = c(1,2,3,4, 6,7,8,9, 11,12,13,14, 16,17,18,19, 21,22,23,24, 26,27,28,29, 31,32,33,34, 36,37,38,39, 41,42,43,44, 46,47,48,49),
        xlab="a",
        ylab="Structural Hamming distance (to CPDAG)",
        col = c(0,gray(0.8),gray(0.55),gray(0.3)),
        names = c("","0","","",  "","0.1","","",  "","0.2","","",  "","0.3","","",  "","0.4","","",  "","0.5","","",  "","0.6","","",
                  "","0.7","","",  "","0.8","","",  "","0.9","",""))

#legend(40,30, # places a legend at the appropriate place 
#       c("GDS_SEV","GES", "PC", "BEST_SCORE"), # puts text in the legend
#       lty=c(0,0,0,0), # gives the legend appropriate symbols (lines)
#       #pch=c(22,22,22), # gives the legend appropriate symbols (lines)
#       #lwd=c(2.5,2.5,2.5),
#       #pt.bg=c(0,gray(0.7),gray(0.3)),
#       #col=c(1,1,1))
#       bg = c(gray(1)),
#       fill = c(0,gray(0.8),gray(0.55),gray(0.3)))
dev.off()   



pdf(file  = "./results/experimentNonSameVarianceCPDAG.pdf", 
    width = 13, height = 4)
boxplot(hdsemCPDAG[[1]],hdgesCPDAG[[1]],hdpcCPDAG[[1]], 
        hdsemCPDAG[[2]],hdgesCPDAG[[2]],hdpcCPDAG[[2]],
        hdsemCPDAG[[3]],hdgesCPDAG[[3]],hdpcCPDAG[[3]],
        hdsemCPDAG[[4]],hdgesCPDAG[[4]],hdpcCPDAG[[4]],
        hdsemCPDAG[[5]],hdgesCPDAG[[5]],hdpcCPDAG[[5]],
        hdsemCPDAG[[6]],hdgesCPDAG[[6]],hdpcCPDAG[[6]],
        hdsemCPDAG[[7]],hdgesCPDAG[[7]],hdpcCPDAG[[7]],
        hdsemCPDAG[[8]],hdgesCPDAG[[8]],hdpcCPDAG[[8]],
        hdsemCPDAG[[9]],hdgesCPDAG[[9]],hdpcCPDAG[[9]],
        hdsemCPDAG[[10]],hdgesCPDAG[[10]],hdpcCPDAG[[10]],
        at = c(1,2,3, 5,6,7, 9,10,11, 13,14,15, 17,18,19, 21,22,23, 25,26,27, 29,30,31, 33,34,35, 37,38,39),
        xlab="a",
        ylab="Structural Hamming distance (to CPDAG)",
        col = c(0,gray(0.7),gray(0.3)),
        names = c("","0","","","0.1","","","0.2","","","0.3","","","0.4","","","0.5","","","0.6","",
                  "","0.7","","","0.8","","","0.9",""))

#legend(35,22, # places a legend at the appropriate place 
#       c("GDS_SEV","GES", "PC"), # puts text in the legend
#       lty=c(0,0,0), # gives the legend appropriate symbols (lines)
#       #pch=c(22,22,22), # gives the legend appropriate symbols (lines)
#       #lwd=c(2.5,2.5,2.5),
#       #pt.bg=c(0,gray(0.7),gray(0.3)),
#       #col=c(1,1,1))
#       bg = c(gray(1)),
#       fill = c(0,gray(0.7),gray(0.3)))
dev.off()   

