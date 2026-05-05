library(spatstat.data)

#' Locations and sizes of Longleaf pine trees.
#' A marked point pattern.
#' The data record the locations and diameters of 
#' 584 Longleaf pine (\emph{Pinus palustris}) trees 
#' in a 200 x 200 metre region in southern Georgia (USA).
#' They were collected and analysed by Platt, Evans and Rathbun (1988).
#' This is a marked point pattern; the mark associated with a tree is its
#' diameter at breast height (\code{dbh}), a convenient measure of its size. 
#' Several analyses have considered only the ``adult'' trees which
#' are conventionally defined as those trees with \code{dbh}
#' greater than or equal to 30 cm.
#' The pattern is regarded as spatially inhomogeneous.

ll <- data.frame(
  x=longleaf$x, 
  y=longleaf$y, 
  marks=longleaf$marks
)

write.csv(ll, file="data/longleaf.csv", row.names=FALSE)
