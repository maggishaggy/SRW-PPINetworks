# source("https://bioconductor.org/biocLite.R")
# biocLite("GOSemSim")
# install.packages("RSQLite")
library(GOSemSim)
source("wangMethod.R")
source("resnikLinMethods.R")


##' Saves a semantic similarity matrix of the GO terms in the given dataset of GO protein annotations.
##' @param inputFile Path to the tab separated file containing GO protein annotations
##' @param ont Ontology for the mapping with values in BP, CC or MF. Only annotations of the listed GO namespaces BP (biological process), MF (molecular function) or CC (cellular component) are returned.
##' @param method Method for calculating the semantic similarity. It can be Resnik, Lin, Jiang, Wang
##' @param verbose flag to print the progress bar
##' @param outputFile Path to save the similarity matrix
calcGOSimMatrix <- function(inputFile, ont, method, outputFile, verbose = TRUE){
  
  if(ont == "BP")
    go_data <- read.table("../data/go/BPGOfull.txt", header = FALSE, sep = ' ', col.names = c('child', 'relationship', 'parent'), stringsAsFactors=FALSE)
  else if(ont == "MF")
    go_data <- read.table("../data/go/MFGOfull.txt", header = FALSE, sep = ' ', col.names = c('child', 'relationship', 'parent'), stringsAsFactors=FALSE)
  else
    go_data <- read.table("../data/go/CCGOfull.txt", header = FALSE, sep = ' ', col.names = c('child', 'relationship', 'parent'), stringsAsFactors=FALSE)
  
  if(method == "Wang"){
    message('Loading mappings ...')
    go.id.col <- 2
    go <- unique(read.table(inputFile, header = TRUE, sep = '\t', col.names = c('PID', 'GO'), stringsAsFactors=FALSE)[, go.id.col])
    go <- sort(go)
    message('Computing GO terms similarities ...')
    startTime <- Sys.time()
    
    scores <- wangMethodSim(go, go, go_data)
    diag(scores) <- 1
    
    endTime <- Sys.time()
    print(endTime - startTime)
    
    write.table(scores, outputFile, sep = '\t')
    message("Done.")
  }
  else{
    stringGO <- getGoAnnoForOnt(inputFile, ont, go_data)
    terms <- unique(stringGO@geneAnno$GO)
    go <- levels(terms)
    
    message('Computing GO terms similarities ...')
    startTime <- Sys.time()
    
    anc <- getAncestors(go_data, go)
    
    scores <- infoContentMethod_cpp(go, go, anc, stringGO@IC, method, stringGO@ont)
    
    diag(scores) <- 1
    
    endTime <- Sys.time()
    
    print(endTime - startTime)
    
    write.table(scores, outputFile, sep = '\t')
    message("Done.")
  }
}


calculateSim700 <- function(){
  calcGOSimMatrix("../data/human_ppi_700/HumanPPI_GO_BP.txt", "BP", "Resnik", "../data/sim/human_ppi_700/GO_BP_resnik.txt")
  calcGOSimMatrix("../data/human_ppi_700/HumanPPI_GO_BP.txt", "BP", "Lin", "../data/sim/human_ppi_700/GO_BP_lin.txt")
  calcGOSimMatrix("../data/human_ppi_700/HumanPPI_GO_BP.txt", "BP", "Wang", "../data/sim/human_ppi_700/GO_BP_wang.txt")
  
  calcGOSimMatrix("../data/human_ppi_700/HumanPPI_GO_MF.txt", "MF", "Resnik", "../data/sim/human_ppi_700/GO_MF_resnik.txt")
  calcGOSimMatrix("../data/human_ppi_700/HumanPPI_GO_MF.txt", "MF", "Lin", "../data/sim/human_ppi_700/GO_MF_lin.txt")
  calcGOSimMatrix("../data/human_ppi_700/HumanPPI_GO_MF.txt", "MF", "Wang", "../data/sim/human_ppi_700/GO_MF_wang.txt")
  
  calcGOSimMatrix("../data/human_ppi_700/HumanPPI_GO_CC.txt", "CC", "Resnik", "../data/sim/human_ppi_700/GO_CC_resnik.txt")
  calcGOSimMatrix("../data/human_ppi_700/HumanPPI_GO_CC.txt", "CC", "Lin", "../data/sim/human_ppi_700/GO_CC_lin.txt")
  calcGOSimMatrix("../data/human_ppi_700/HumanPPI_GO_CC.txt", "CC", "Wang", "../data/sim/human_ppi_700/GO_CC_wang.txt")
  
  calcGOSimMatrix("../data/human_ppi_700/HumanPPI_GO_BP_no_bias.txt", "BP", "Resnik", "../data/sim/human_ppi_700/GO_BP_no_bias_resnik.txt")
  calcGOSimMatrix("../data/human_ppi_700/HumanPPI_GO_BP_no_bias.txt", "BP", "Lin", "../data/sim/human_ppi_700/GO_BP_no_bias_lin.txt")
  calcGOSimMatrix("../data/human_ppi_700/HumanPPI_GO_BP_no_bias.txt", "BP", "Wang", "../data/sim/human_ppi_700/GO_BP_no_bias_wang.txt")
  
  calcGOSimMatrix("../data/human_ppi_700/HumanPPI_GO_MF_no_bias.txt", "MF", "Resnik", "../data/sim/human_ppi_700/GO_MF_no_bias_resnik.txt")
  calcGOSimMatrix("../data/human_ppi_700/HumanPPI_GO_MF_no_bias.txt", "MF", "Lin", "../data/sim/human_ppi_700/GO_MF_no_bias_lin.txt")
  calcGOSimMatrix("../data/human_ppi_700/HumanPPI_GO_MF_no_bias.txt", "MF", "Wang", "../data/sim/human_ppi_700/GO_MF_no_bias_wang.txt")
  
  calcGOSimMatrix("../data/human_ppi_700/HumanPPI_GO_CC_no_bias.txt", "CC", "Resnik", "../data/sim/human_ppi_700/GO_CC_no_bias_resnik.txt")
  calcGOSimMatrix("../data/human_ppi_700/HumanPPI_GO_CC_no_bias.txt", "CC", "Lin", "../data/sim/human_ppi_700/GO_CC_no_bias_lin.txt")
  calcGOSimMatrix("../data/human_ppi_700/HumanPPI_GO_CC_no_bias.txt", "CC", "Wang", "../data/sim/human_ppi_700/GO_CC_no_bias_wang.txt")
}


calculateSim900 <- function(){
  calcGOSimMatrix("../data/human_ppi_900/HumanPPI_GO_BP.txt", "BP", "Resnik", "../data/sim/human_ppi_900/GO_BP_resnik.txt")
  calcGOSimMatrix("../data/human_ppi_900/HumanPPI_GO_BP.txt", "BP", "Lin", "../data/sim/human_ppi_900/GO_BP_lin.txt")
  calcGOSimMatrix("../data/human_ppi_900/HumanPPI_GO_BP.txt", "BP", "Wang", "../data/sim/human_ppi_900/GO_BP_wang.txt")
  
  calcGOSimMatrix("../data/human_ppi_900/HumanPPI_GO_MF.txt", "MF", "Resnik", "../data/sim/human_ppi_900/GO_MF_resnik.txt")
  calcGOSimMatrix("../data/human_ppi_900/HumanPPI_GO_MF.txt", "MF", "Lin", "../data/sim/human_ppi_900/GO_MF_lin.txt")
  calcGOSimMatrix("../data/human_ppi_900/HumanPPI_GO_MF.txt", "MF", "Wang", "../data/sim/human_ppi_900/GO_MF_wang.txt")
  
  calcGOSimMatrix("../data/human_ppi_900/HumanPPI_GO_CC.txt", "CC", "Resnik", "../data/sim/human_ppi_900/GO_CC_resnik.txt")
  calcGOSimMatrix("../data/human_ppi_900/HumanPPI_GO_CC.txt", "CC", "Lin", "../data/sim/human_ppi_900/GO_CC_lin.txt")
  calcGOSimMatrix("../data/human_ppi_900/HumanPPI_GO_CC.txt", "CC", "Wang", "../data/sim/human_ppi_900/GO_CC_wang.txt")
  
  calcGOSimMatrix("../data/human_ppi_900/HumanPPI_GO_BP_no_bias.txt", "BP", "Resnik", "../data/sim/human_ppi_900/GO_BP_no_bias_resnik.txt")
  calcGOSimMatrix("../data/human_ppi_900/HumanPPI_GO_BP_no_bias.txt", "BP", "Lin", "../data/sim/human_ppi_900/GO_BP_no_bias_lin.txt")
  calcGOSimMatrix("../data/human_ppi_900/HumanPPI_GO_BP_no_bias.txt", "BP", "Wang", "../data/sim/human_ppi_900/GO_BP_no_bias_wang.txt")
  
  calcGOSimMatrix("../data/human_ppi_900/HumanPPI_GO_MF_no_bias.txt", "MF", "Resnik", "../data/sim/human_ppi_900/GO_MF_no_bias_resnik.txt")
  calcGOSimMatrix("../data/human_ppi_900/HumanPPI_GO_MF_no_bias.txt", "MF", "Lin", "../data/sim/human_ppi_900/GO_MF_no_bias_lin.txt")
  calcGOSimMatrix("../data/human_ppi_900/HumanPPI_GO_MF_no_bias.txt", "MF", "Wang", "../data/sim/human_ppi_900/GO_MF_no_bias_wang.txt")
  
  calcGOSimMatrix("../data/human_ppi_900/HumanPPI_GO_CC_no_bias.txt", "CC", "Resnik", "../data/sim/human_ppi_900/GO_CC_no_bias_resnik.txt")
  calcGOSimMatrix("../data/human_ppi_900/HumanPPI_GO_CC_no_bias.txt", "CC", "Lin", "../data/sim/human_ppi_900/GO_CC_no_bias_lin.txt")
  calcGOSimMatrix("../data/human_ppi_900/HumanPPI_GO_CC_no_bias.txt", "CC", "Wang", "../data/sim/human_ppi_900/GO_CC_no_bias_wang.txt")
}

calculateSim700()
calculateSim900()