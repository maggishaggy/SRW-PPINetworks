# Second version for calculating partial wang similarities (problem with memory)

# source("https://bioconductor.org/biocLite.R")
# biocLite("GOSemSim")
# install.packages("RSQLite")
library(GOSemSim)
source("wangMethod.R")
source("resnikLinMethods.R")


##' Saves a semantic similarity matrix of the GO terms in the given dataset of GO protein annotations.
##' @param inputFile Path to the tab separated file containing GO protein annotations
##' @param ont Ontology for the mapping with values in BP, CC or MF. Only annotations of the listed GO namespaces BP (biological process), MF (molecular function) or CC (cellular component) are returned.
##' @param outputFile Path to save the similarity matrix
calcGOSimMatrix <- function(inputFile, ont, outputFile){
  
  if(ont == "BP")
    go_data <- read.table("../data/go/BPGOfull.txt", header = FALSE, sep = ' ', col.names = c('child', 'relationship', 'parent'), stringsAsFactors=FALSE)
  else if(ont == "MF")
    go_data <- read.table("../data/go/MFGOfull.txt", header = FALSE, sep = ' ', col.names = c('child', 'relationship', 'parent'), stringsAsFactors=FALSE)
  else
    go_data <- read.table("../data/go/CCGOfull.txt", header = FALSE, sep = ' ', col.names = c('child', 'relationship', 'parent'), stringsAsFactors=FALSE)
  
  message('Loading mappings ...')
  go.id.col <- 2
  go <- unique(read.table(inputFile, header = TRUE, sep = '\t', col.names = c('PID', 'GO'), stringsAsFactors=FALSE)[, go.id.col])
  go <- sort(go)
  message('Computing GO terms similarities ...')
  startTime <- Sys.time()
  
  go1 <- go[1:(length(go)/3)]
  go2 <- go[(length(go)/3 + 1):(length(go)/3*2)]
  go3 <- go[(length(go)/3*2 + 1):length(go)]
	
  scores <- wangMethodSim(go1, go, go_data)
  write.table(scores, paste0(outputFile, "1.txt"), sep = '\t')
  scores <- wangMethodSim(go2, go, go_data)
  write.table(scores, paste0(outputFile, "2.txt"), sep = '\t')
  scores <- wangMethodSim(go3, go, go_data)
  write.table(scores, paste0(outputFile, "3.txt"), sep = '\t')
    
  endTime <- Sys.time()
  print(endTime - startTime)
  message("Done.")
}


calculateSim700 <- function(){
  calcGOSimMatrix("../data/human_ppi_700/HumanPPI_GO_BP.txt", "BP", "../data/sim/human_ppi_700/GO_BP_wang")
  calcGOSimMatrix("../data/human_ppi_700/HumanPPI_GO_MF.txt", "MF", "../data/sim/human_ppi_700/GO_MF_wang")
  calcGOSimMatrix("../data/human_ppi_700/HumanPPI_GO_CC.txt", "CC", "../data/sim/human_ppi_700/GO_CC_wang")
  calcGOSimMatrix("../data/human_ppi_700/HumanPPI_GO_BP_no_bias.txt", "BP", "../data/sim/human_ppi_700/GO_BP_no_bias_wang")
  calcGOSimMatrix("../data/human_ppi_700/HumanPPI_GO_MF_no_bias.txt", "MF", "../data/sim/human_ppi_700/GO_MF_no_bias_wang")
  calcGOSimMatrix("../data/human_ppi_700/HumanPPI_GO_CC_no_bias.txt", "CC", "../data/sim/human_ppi_700/GO_CC_no_bias_wang")
}


calculateSim900 <- function(){
  calcGOSimMatrix("../data/human_ppi_900/HumanPPI_GO_BP.txt", "BP", "../data/sim/human_ppi_900/GO_BP_wang")
  calcGOSimMatrix("../data/human_ppi_900/HumanPPI_GO_MF.txt", "MF", "../data/sim/human_ppi_900/GO_MF_wang")
  calcGOSimMatrix("../data/human_ppi_900/HumanPPI_GO_CC.txt", "CC", "../data/sim/human_ppi_900/GO_CC_wang")
  calcGOSimMatrix("../data/human_ppi_900/HumanPPI_GO_BP_no_bias.txt", "BP", "../data/sim/human_ppi_900/GO_BP_no_bias_wang")
  calcGOSimMatrix("../data/human_ppi_900/HumanPPI_GO_MF_no_bias.txt", "MF", "../data/sim/human_ppi_900/GO_MF_no_bias_wang")
  calcGOSimMatrix("../data/human_ppi_900/HumanPPI_GO_CC_no_bias.txt", "CC", "../data/sim/human_ppi_900/GO_CC_no_bias_wang")
}


calculateSim700()
calculateSim900()