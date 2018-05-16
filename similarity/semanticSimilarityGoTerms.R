# source("https://bioconductor.org/biocLite.R")
# biocLite("GOSemSim")
# biocLite("GO.db")
library(GO.db)
library(GOSemSim)

##' @importFrom GO.db GOMFPARENTS
##' @importFrom GO.db GOBPPARENTS
##' @importFrom GO.db GOCCPARENTS
getParents <- function(ont) {
  Parents <- switch(ont,
                    MF = "GOMFPARENTS",
                    BP = "GOBPPARENTS",
                    CC = "GOCCPARENTS",
                    DO = "DO.db::DOPARENTS"
  )
  if (ont == "DO") {
    db <- "DO.db"
    requireNamespace(db)
  }
  Parents <- eval(parse(text=Parents))
  return(Parents)
}

##' @importFrom GO.db GOTERM
##' @importFrom AnnotationDbi toTable
prepare_relation_df <- function() {
  gtb <- toTable(GOTERM)
  gtb <- gtb[,c(2:4)]
  gtb <- unique(gtb)
  
  ptb <- lapply(c("BP", "MF", "CC"), function(ont) {
    id <- with(gtb, go_id[Ontology == ont])
    pid <- mget(id, getParents(ont))
    
    n <- sapply(pid, length)
    cid <- rep(names(pid), times=n)
    relationship <- unlist(lapply(pid, names))
    
    data.frame(id=cid,
               relationship=relationship,
               parent=unlist(pid),
               stringsAsFactors = FALSE)
  }) 
  ptb <- do.call('rbind', ptb)
  
  gotbl <- merge(gtb, ptb, by.x="go_id", by.y="id")
  return(gotbl)
}

#' Computes IC for GO terms. Code taken from https://github.com/GuangchuangYu/GOSemSim/blob/72caecd9d5fc9d3a4b2fa484ea7b39c1e4d38266/R/computeIC.R
#' @param goAnno gene ontology annotations
#' @param ont ontology, with values in BP, MF or CC. 
##' @importFrom AnnotationDbi as.list
##' @importFrom GO.db GOBPOFFSPRING
##' @importFrom GO.db GOCCOFFSPRING
##' @importFrom GO.db GOMFOFFSPRING
computeIC <- function(goAnno, ont) {
  godata <- prepare_relation_df()        
  
  goids <- unique(godata[godata$Ontology == ont, "go_id"])
  ## all GO terms appearing in an given ontology ###########
  goterms=goAnno$GO
  gocount <- table(goterms)
  ## goid of specific organism and selected category.
  goname  <- names(gocount) 
  
  ## ensure goterms not appearing in the specific annotation have 0 frequency..
  go.diff        <- setdiff(goids, goname)
  m              <- double(length(go.diff))
  names(m)       <- go.diff
  gocount        <- as.vector(gocount)
  names(gocount) <- goname
  gocount        <- c(gocount, m)
  
  Offsprings <- switch(ont,
                       MF = AnnotationDbi::as.list(GOMFOFFSPRING),
                       BP = AnnotationDbi::as.list(GOBPOFFSPRING),
                       CC = AnnotationDbi::as.list(GOCCOFFSPRING))
  
  cnt <- gocount[goids] + sapply(goids, function(i) sum(gocount[Offsprings[[i]]], na.rm=TRUE))
  names(cnt) <- goids
  
  ## the probabilities of occurrence of GO terms in a specific corpus.
  p <- cnt/sum(gocount)
  ## IC of GO terms was quantified as the negative log likelihood.
  IC <- -log(p)
  return(IC)
}


#' Generates GO annotations used for semantic similarity
#' @param filePath Path to the tab separated file containing GO protein annotations
#' @param ont Ontology for the mapping with values in BP, CC or MF. Only annotations of the listed GO namespaces BP (biological process), MF (molecular function) or CC (cellular component) are returned.
#' @return GOSemSimData object
getGoAnnoForOnt <- function(filePath, ont){
  message("Loading mappings ...")
  goa <- read.table(filePath, header = FALSE, sep='\t', col.names = c('PID', 'PID2', 'GO', ''))
  protein.id.col <- 1
  go.id.col <- 3
  mappings <- data.frame( 
    PROTEINID = goa[, protein.id.col],
    GO = goa[,go.id.col])
  d <- godata(ont=ont)
  d@geneAnno <- mappings
  message("Computing IC from protein GO mapping database...")
  d@IC <- computeIC(d@geneAnno, ont=d@ont)
  return(d)
}


##' Saves a semantic similarity matrix of the GO terms in the given dataset of GO protein annotations.
##' @param inputFile Path to the tab separated file containing GO protein annotations
##' @param ont Ontology for the mapping with values in BP, CC or MF. Only annotations of the listed GO namespaces BP (biological process), MF (molecular function) or CC (cellular component) are returned.
##' @param method Method for calculating the semantic similarity. It can be Resnik, Lin, Jiang, Wang
##' @param verbose flag to print the progress bar
##' @param outputFile Path to save the similarity matrix
calcGOSimMatrix <- function(inputFile, ont, method, outputFile, verbose = TRUE){
  stringGO <- getGoAnnoForOnt(inputFile, ont)
  terms <- unique(stringGO@geneAnno$GO)
  numTerms <- length(terms)
  
  scores <- matrix(1, numTerms, numTerms)
  rownames(scores) <- levels(terms)
  colnames(scores) <- levels(terms)
  
  message('Computing GO terms similarities ...')
  pb = txtProgressBar(min = 0, max = numTerms - 1, style = 3)
  startTime <- Sys.time()
  
  for(i in 1:(numTerms - 1)){
    for(j in (i+1):numTerms){
      GO1 <- toString(terms[i])
      GO2 <- toString(terms[j])
      sim <- termSim(GO1, GO2, stringGO, method = method)
      scores[GO1, GO2] <- sim
      scores[GO2, GO1] <- sim
    }
    setTxtProgressBar(pb, i)
  }
  
  endTime <- Sys.time()
  message('')
  print(endTime - startTime)
  message("Done.")
  
  write.table(scores, outputFile, sep = '\t')
}

calcGOSimMatrix("../../Data/Go annotations/HumanPPI700_GO_BP_modified.txt", "BP", "Resnik", "../../Data/Semantic similarity/700modified_BP_resnik.txt")
calcGOSimMatrix("../../Data/Go annotations/HumanPPI700_GO_BP_modified.txt", "BP", "Lin", "../../Data/Semantic similarity/700modified_BP_lin.txt")
calcGOSimMatrix("../../Data/Go annotations/HumanPPI700_GO_BP_modified.txt", "BP", "Wang", "../../Data/Semantic similarity/700modified_BP_wang.txt")

calcGOSimMatrix("../../Data/Go annotations/HumanPPI700_GO_MF_modified.txt", "MF", "Resnik", "../../Data/Semantic similarity/700modified_MF_resnik.txt")
calcGOSimMatrix("../../Data/Go annotations/HumanPPI700_GO_MF_modified.txt", "MF", "Lin", "../../Data/Semantic similarity/700modified_MF_lin.txt")
calcGOSimMatrix("../../Data/Go annotations/HumanPPI700_GO_MF_modified.txt", "MF", "Wang", "../../Data/Semantic similarity/700modified_MF_wang.txt")

calcGOSimMatrix("../../Data/Go annotations/HumanPPI700_GO_CC_modified.txt", "CC", "Resnik", "../../Data/Semantic similarity/700modified_CC_resnik.txt")
calcGOSimMatrix("../../Data/Go annotations/HumanPPI700_GO_CC_modified.txt", "CC", "Lin", "../../Data/Semantic similarity/700modified_CC_lin.txt")
calcGOSimMatrix("../../Data/Go annotations/HumanPPI700_GO_CC_modified.txt", "CC", "Wang", "../../Data/Semantic similarity/700modified_CC_wang.txt")

calcGOSimMatrix("../../Data/Go annotations/HumanPPI900_GO_BP_modified.txt", "BP", "Resnik", "../../Data/Semantic similarity/900modified_BP_resnik.txt")
calcGOSimMatrix("../../Data/Go annotations/HumanPPI900_GO_BP_modified.txt", "BP", "Lin", "../../Data/Semantic similarity/900modified_BP_lin.txt")
calcGOSimMatrix("../../Data/Go annotations/HumanPPI900_GO_BP_modified.txt", "BP", "Wang", "../../Data/Semantic similarity/900modified_BP_wang.txt")

calcGOSimMatrix("../../Data/Go annotations/HumanPPI900_GO_MF_modified.txt", "MF", "Resnik", "../../Data/Semantic similarity/900modified_MF_resnik.txt")
calcGOSimMatrix("../../Data/Go annotations/HumanPPI900_GO_MF_modified.txt", "MF", "Lin", "../../Data/Semantic similarity/900modified_MF_lin.txt")
calcGOSimMatrix("../../Data/Go annotations/HumanPPI900_GO_MF_modified.txt", "MF", "Wang", "../../Data/Semantic similarity/900modified_MF_wang.txt")

calcGOSimMatrix("../../Data/Go annotations/HumanPPI900_GO_CC_modified.txt", "CC", "Resnik", "../../Data/Semantic similarity/900modified_CC_resnik.txt")
calcGOSimMatrix("../../Data/Go annotations/HumanPPI900_GO_CC_modified.txt", "CC", "Lin", "../../Data/Semantic similarity/900modified_CC_lin.txt")
calcGOSimMatrix("../../Data/Go annotations/HumanPPI900_GO_CC_modified.txt", "CC", "Wang", "../../Data/Semantic similarity/900modified_CC_wang.txt")