# Wang semantic similarity measure of GO terms, code taken from package GOSemSim and modified
# to work with user defined go terms relatonships data

wangMethodSim <- function(t1, t2, rel_df) {
  message("Calculate sv for go terms ...")
  sv <- list()
  for(t in union(t1, t2)){
    sv[[t]] <- getSV(t, rel_df)
  }
  rm(rel_df)
  message("Wang method internal ...")
  matrix( mapply(wangMethod_internal,
                 rep(t1, length(t2)),
                 rep(t2, each = length(t1)),
                 MoreArgs = list(sv = sv)),
          dimnames = list(t1, t2), ncol = length(t2) ) 
}

##' Method Wang for semantic similarity measuring
##'
##'
##' @title wangMethod
##' @param ID1 Ontology Term
##' @param ID2 Ontology Term
##' @param sv calculated values with getSV function including terms ID1 and ID2
##' @return semantic similarity score
##' @author Guangchuang Yu \url{http://ygc.name}
wangMethod_internal <- function(ID1, ID2, sv) {
  if (ID1 == ID2)
    return (sim=1)
  
  sv.a <- sv[[ID1]]
  sv.b <- sv[[ID2]]
  
  if(all(is.na(sv.a)) || all(is.na(sv.b)))
    return (NA)
  
  idx         <- intersect(names(sv.a), names(sv.b))
  inter.sva   <- sv.a[idx]
  inter.svb   <- sv.b[idx]
  if (is.null(inter.sva) ||
      is.null(inter.svb) ||
      length(inter.sva) == 0 ||
      length(inter.svb) == 0) {
    return (NA)
  } 
  
  sim <- sum(inter.sva,inter.svb) / sum(sv.a, sv.b)
  return(sim)
}

getSV <- function(ID, rel_df, weight=NULL) {
  topNode <- "all"
  
  if (ID == topNode) {
    sv <- 1
    names(sv) <- topNode
    return (sv)
  }
  if (is.null(weight)) {
    weight <- c(0.8, 0.6, 0.7)
    names(weight) <- c("is_a", "part_of", "other")
  }
  
  if (! 'relationship' %in% colnames(rel_df))
    rel_df$relationship <- "other"
  
  rel_df$relationship[!rel_df$relationship %in% c("is_a", "part_of")] <- "other"
  
  sv <- 1
  names(sv) <- ID
  allid <- ID
  
  idx <- which(rel_df[,1] %in% ID)
  while (length(idx) != 0) {
    p <- rel_df[idx,]
    pid <- p$parent
    allid <- c(allid, pid)
    
    sv <- c(sv, weight[p$relationship]*sv[p[,1]])
    names(sv) <- allid
    idx <- which(rel_df[,1] %in% pid)
  }
  
  sv <- sv[!is.na(names(sv))]
  sv <- sv[!duplicated(names(sv))]
  
  return(sv)
}

