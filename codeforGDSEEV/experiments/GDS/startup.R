# Copyright (c) 2010 - 2012  Jonas Peters  [peters@stat.math.ethz.ch]
# All rights reserved.  See the file COPYING for license terms. 
#library(igraph)
#library(RBGL) 
#library(pcalg)
#library(pcalg, lib.loc = "~/R/library") 
#library(ggm)
#library(gplots)
#library(fields)

source("../../util_DAGs/randomDAG.R")
source("../../util_DAGs/randomB.R")
source("../../util_DAGs/sampleFromG.R")
source("../../util_DAGs/dag2cpdagAdj.R")
library(MASS)

cat("Greedy Equivalent Search is not loaded since it is not included in the standart pcalg package yet (28.08.2013).\n")
#source("../../startups/startupGES.R", chdir = TRUE)
source("../../startups/startupGDS.R", chdir = TRUE)
source("../../startups/startupPC.R", chdir = TRUE)
source("../../startups/startupSHD.R", chdir = TRUE)



