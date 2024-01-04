library(bioconductor)

gdcprojects <- getGDCprojects()
getProjectSummary('TCGA-LUAD')






# build a query to retrieve gene expression data ------------
query_TCGA <- GDCquery(     project = "TCGA-COAD", 
                            data.category = "Biospecimen", 
                            data.type = "Slide Image",
                            experimental.strategy = "Diagnostic Slide",
                            barcode = c("TCGA-RU-A8FL","TCGA-AA-3972"))

getResults(query_TCGA)

# download data - GDCdownload
GDCdownload(query_TCGA)