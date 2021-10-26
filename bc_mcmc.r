# Calculating graph undirected graph posteriors using the
# method from A. Mohammadi∗ and E. C. Wit 2015

library(gdata)
library(BDgraph)
args = commandArgs(trailingOnly=TRUE)
mydata <- read.csv(args[1])
bdgraph.obj <- bdgraph(data = mydata, iter = 10000, save=TRUE )
samples = bdgraph.obj$sample_graphs
write.table(samples, args[2], sep=",", row.names=FALSE, col.names=FALSE)



