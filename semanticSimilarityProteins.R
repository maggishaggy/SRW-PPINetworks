# source("https://bioconductor.org/biocLite.R")
# biocLite("GOSemSim")
# biocLite("GO.db")
library(GO.db)
library(GOSemSim)


##' Saves a semantic similarity matrix of the GO terms in the given dataset of GO protein annotations and GO terms semantic similarity
##' @param inputFile1 Path to the tab separated file containing GO protein annotations
##' @param inputFile2 Path to the tab separated file containing the similarity matrix of GO terms
##' @param method combine method, one of max, avg, rcmax, BMA
##' @param outputFile Path to save the similarity matrix
calcProteinSimMatrix <- function(inputFile1, inputFile2, method, outputFile){
  goa <- read.table(inputFile1, header = FALSE, sep='\t', col.names = c('PID', 'PID2', 'GO', ''))
  proteins <- unique(goa$PID)
  simMatrix <- as.matrix(read.table(inputFile2, sep = '\t'))
  
  numProteins <- length(proteins)
  
  scores <- matrix(1, numProteins, numProteins)
  rownames(scores) <- levels(proteins)
  colnames(scores) <- levels(proteins)
  
  message('Computing protein similarities ...')
  pb = txtProgressBar(min = 0, max = numProteins - 1, style = 3)
  startTime <- Sys.time()
  
  for(i in 1:(numProteins - 1)){
    for(j in (i+1):numProteins){
      protein1 <- toString(proteins[i])
      protein2 <- toString(proteins[j])
      simMat <- simMatrix[goa[goa$PID == protein1, ]$GO, goa[goa$PID == protein2, ]$GO]
      sim <- combineScores(simMat, method)
      scores[protein1, protein2] <- sim
      scores[protein2, protein1] <- sim
    }
    setTxtProgressBar(pb, i)
  }
  
  endTime <- Sys.time()
  message('')
  print(endTime - startTime)
  message("Done.")
  
  write.table(scores, outputFile, sep = '\t')
}

calcProteinSimMatrix("../../Data/Go annotations/HumanPPI700_GO_BP_modified.txt", 
                     "../../Data/Semantic similarity/700modified_BP_resnik.txt", 
                     "rcmax", 
                     "../../Data/Protein semantic similarity/HumanPPI700_GO_BP_modified_resnik_rcmax.txt")

calcProteinSimMatrix("../../Data/Go annotations/HumanPPI700_GO_BP_modified.txt", 
                     "../../Data/Semantic similarity/700modified_BP_resnik.txt", 
                     "BMA", 
                     "../../Data/Protein semantic similarity/HumanPPI700_GO_BP_modified_resnik_BMA.txt")


calcProteinSimMatrix("../../Data/Go annotations/HumanPPI700_GO_BP_modified.txt", 
                     "../../Data/Semantic similarity/700modified_BP_lin.txt", 
                     "rcmax", 
                     "../../Data/Protein semantic similarity/HumanPPI700_GO_BP_modified_lin_rcmax.txt")

calcProteinSimMatrix("../../Data/Go annotations/HumanPPI700_GO_BP_modified.txt", 
                     "../../Data/Semantic similarity/700modified_BP_lin.txt", 
                     "BMA", 
                     "../../Data/Protein semantic similarity/HumanPPI700_GO_BP_modified_lin_BMA.txt")


calcProteinSimMatrix("../../Data/Go annotations/HumanPPI700_GO_BP_modified.txt", 
                     "../../Data/Semantic similarity/700modified_BP_wang.txt", 
                     "rcmax", 
                     "../../Data/Protein semantic similarity/HumanPPI700_GO_BP_modified_wang_rcmax.txt")

calcProteinSimMatrix("../../Data/Go annotations/HumanPPI700_GO_BP_modified.txt", 
                     "../../Data/Semantic similarity/700modified_BP_wang.txt", 
                     "BMA", 
                     "../../Data/Protein semantic similarity/HumanPPI700_GO_BP_modified_wang_BMA.txt")
