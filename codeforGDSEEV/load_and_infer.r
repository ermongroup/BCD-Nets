source("util_DAGs/randomDAG.R")
source("util_DAGs/randomB.R")
source("util_DAGs/sampleFromG.R")
source("util_DAGs/dag2cpdagAdj.R")
library(MASS)

cat("Greedy Equivalent Search is not loaded since it is not included in the standart pcalg package yet (28.08.2013).\n")
#source("startups/startupGES.R", chdir = TRUE)
source("startups/startupGDS.R", chdir = TRUE)
source("startups/startupPC.R", chdir = TRUE)
source("startups/startupSHD.R", chdir = TRUE)

library(gdata)
args = commandArgs(trailingOnly=TRUE)

mydata <- as.matrix(read.csv(args[1], header=TRUE))


resGDS <- GDS(mydata)
cat("estimated DAG by GDS with equal error variances:\n")
show(resGDS$Adj)
cat("BIC score of the estimated DAG:")
show(resGDS$Score)

write.table(resGDS$Adj, args[2], sep=",", row.names=FALSE, col.names=FALSE)
