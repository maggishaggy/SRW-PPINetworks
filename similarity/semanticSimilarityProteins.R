# source("https://bioconductor.org/biocLite.R")
# biocLite("GOSemSim")
library(GOSemSim)


##' Saves a semantic similarity matrix of the GO terms in the given dataset of GO protein annotations and GO terms semantic similarity
##' @param inputFile1 Path to the tab separated file containing GO protein annotations
##' @param inputFile2 Path to the tab separated file containing the similarity matrix of GO terms
##' @param method Method for calculating the semantic similarity. It can be Resnik, Lin, Jiang, Wang
##' @param combineMethod combine method, one of max, avg, rcmax, BMA
##' @param ont Ontology for the mapping with values in BP, CC or MF. Only annotations of the listed GO namespaces BP (biological process), MF (molecular function) or CC (cellular component) are returned.
##' @param outputFile Path to save the similarity matrix
calcProteinSimMatrix <- function(inputFile1, inputFile2, method, combineMethod, ont, outputFile){
  if(ont == "BP")
    go_data <- read.table("../data/go/BPGOfull.txt", header = FALSE, sep = ' ', col.names = c('child', 'relationship', 'parent'), stringsAsFactors=FALSE)
  else if(ont == "MF")
    go_data <- read.table("../data/go/MFGOfull.txt", header = FALSE, sep = ' ', col.names = c('child', 'relationship', 'parent'), stringsAsFactors=FALSE)
  else
    go_data <- read.table("../data/go/CCGOfull.txt", header = FALSE, sep = ' ', col.names = c('child', 'relationship', 'parent'), stringsAsFactors=FALSE)
  
  if(method != "Wang"){
    stringGO <- getGoAnnoForOnt(inputFile, ont, go_data)
    terms <- unique(stringGO@geneAnno$GO)
    go <- levels(terms)
    anc <- getAncestors(go_data, go)
  }
  
  goa <- read.table(inputFile1, header = TRUE, sep='\t', col.names = c('PID', 'GO'))
  proteins <- unique(goa$PID)
  
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
      gos1 <- goa[goa$PID == protein1, ]$GO
      gos2 <- goa[goa$PID == protein2, ]$GO
      if(method == "Wang"){
        simMat <- wangMethodSim(gos1, gos2, go_data)
      }
      else{
        simMat <- infoContentMethod_cpp(go, go, anc, stringGO@IC, method, stringGO@ont)
      }
      sim <- combineScores(simMat, combineMethod)
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

calcProteinSimMatrix("../data/human_ppi_700/HumanPPI_GO_BP_no_bias.txt", 
                     "../data/sim/human_ppi_700/GO_BP_no_bias_resnik.txt", 
                     "Resnik",
                     "rcmax", 
                     "BP",
                     "../data/sim/human_ppi_700/GO_BP_no_bias_resnik_rcmax.txt")
