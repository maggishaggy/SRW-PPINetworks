# Resnik and Lin semantic similarities of GO terms, code taken from package GOSemSim and modified
# to work with user defined dataset of protein-go annotations and defined go terms relatonships data

# biocLite("GOSemSim")
# library(GOSemSim)

infoContentMethod_cpp <- function(id1_, id2_, anc_, ic_, method_, ont_) {
  .Call('GOSemSim_infoContentMethod_cpp', PACKAGE = 'GOSemSim', id1_, id2_, anc_, ic_, method_, ont_)
}


#' Finds offsprings from given relation able for given go term
#' @param go_data table with go term child parent relationships
#' @param go_id id of go term
getOffsprings <- function(go_data, go_id){
  all_offsprings <- c()
  children_go <- c(go_id)
  while(length(children_go) != 0){
    parent <- children_go[1]
    children_go <- setdiff(children_go, parent)
    
    children <- go_data[go_data$parent == parent, "child"]
    children_go <- unique(c(children_go, children))
    all_offsprings <- unique(c(all_offsprings, children))
  }
  return(all_offsprings)
}


#' Finds ancestors from given relation able for given go term
#' @param go_data table with go term child parent relationships
#' @param go_ids ids of go terms
getAncestors <- function(go_data, go_ids){
  ancestors <- list()
  for(go_id in go_ids){
    all_ancestors <- c()
    parents_go <- c(go_id)
    while(length(parents_go) != 0){
      child <- parents_go[1]
      parents_go <- setdiff(parents_go, child)
      
      parents <- as.character(go_data[go_data$child == child, "parent"])
      parents_go <- unique(c(parents_go, parents))
      all_ancestors <- unique(c(all_ancestors, parents))
    }
    ancestors[[go_id]] <- all_ancestors
  }
  return(ancestors)
}

#' Computes IC for GO terms. Code taken from https://github.com/GuangchuangYu/GOSemSim/blob/72caecd9d5fc9d3a4b2fa484ea7b39c1e4d38266/R/computeIC.R
#' @param goAnno gene ontology annotations
#' @param ont ontology, with values in BP, MF or CC
#' @param go_data gene term child-parent relationship data
computeIC <- function(goAnno, ont, go_data) {
  go_data <- go_data[go_data$relationship %in% c("is_a", "part_of"), ]
  goids <- unique(c(go_data$child, go_data$parent))
  
  ## all GO terms appearing in an given ontology ###########
  goterms <- goAnno$GO
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
  
  cnt <- gocount[goids] + sapply(goids, function(i) sum(gocount[getOffsprings(go_data, i)], na.rm=TRUE))
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
getGoAnnoForOnt <- function(filePath, ont, go_data){
  message("Loading mappings ...")
  goa <- read.table(filePath, header = TRUE, sep = '\t', col.names = c('PID', 'GO'))
  protein.id.col <- 1
  go.id.col <- 2
  mappings <- data.frame( 
    PROTEINID = goa[, protein.id.col],
    GO = goa[,go.id.col])
  d <- godata(ont=ont)
  d@geneAnno <- mappings
  message("Computing IC from protein GO mapping database...")
  d@IC <- computeIC(d@geneAnno, ont = d@ont, go_data = go_data)
  return(d)
}