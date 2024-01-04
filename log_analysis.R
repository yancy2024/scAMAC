setwd("D:/Code/analysis/P")

rm(list = ls())
library(gridExtra)
library(ggplot2)
library(scales)
library(TSCAN)
library(R.matlab)
library(monocle3)
library(Matrix)

data <- Matrix::readMM("matrix_log.mtx")
#data<-t(data)
cell<-read.table("barcodes_log.csv",header=F,row.names=1,sep=",",check.names=F)
gene<-read.table("genes_log.csv",header=F,row.names=1,sep=",",check.names=F)


#rownames(cell)<-colnames(data)
#rownames(gene)<-rownames(data)
cds <- new_cell_data_set(data,
                         cell_metadata = cell,
                         gene_metadata = gene)


cds <- preprocess_cds(cds, num_dim = 50)
cds <- reduce_dimension(cds)

class.label<-read.table("Petropoulos_lable.csv",header=T,row.names=1,sep=",",check.names=F)
class.label<-as.matrix(class.label)
class.label<- class.label[,1]
label=class.label
label<-as.factor(label)
pData(cds)$Time_Point<-label


cds <- cluster_cells(cds)
cds <- learn_graph(cds)

plot_cells(cds, 
           label_groups_by_cluster=FALSE,  
           color_cells_by = "Time_Point", 
           cell_size = 2,
           graph_label_size = 0,
           label_cell_groups=FALSE,
           label_branch_points=FALSE,
           trajectory_graph_segment_size = 1.2)

cds <- order_cells(cds)

plot_cells(cds,
           color_cells_by = "pseudotime",
           label_cell_groups=FALSE,
           label_leaves=FALSE,
           label_branch_points=FALSE,
           graph_label_size=0,cell_size = 2)

cor.kendall = cor(cds@principal_graph_aux@listData$UMAP$pseudotime, as.numeric(cds@colData@listData$Time), 
                  method = "kendall", use = "complete.obs")

lpsorder2 = data.frame(sample_name = label, State= cds@colData@listData$Time_Point,
                       Pseudotime = cds@principal_graph_aux@listData$UMAP$pseudotime, rank = rank(cds@principal_graph_aux@listData$UMAP$pseudotime))

lpsorder_rank = dplyr::arrange(lpsorder2, rank)

lpsorder_rank$Pseudotime = lpsorder_rank$rank

lpsorder_rank = lpsorder_rank[-4]

lpsorder_rank[1] <- lapply(lpsorder_rank[1], as.character)

subpopulation <- data.frame(cell = label, sub = as.numeric(label)-1)

POS <- TSCAN::orderscore(subpopulation, lpsorder_rank['sample_name'])[[1]]

re = list()
re[[1]] = cor.kendall
re[[2]] = POS

plot_cells(cds, color_cells_by="Time_Point", cell_size = 2,
           label_cell_groups=F,
           label_leaves=F,
           label_branch_points=F,
           graph_label_size=0)+ggtitle(paste0("Cor = ", round(abs(re[[1]]),2), "  POS = ", round(abs(re[[2]]),2)))+theme(text = element_text(size=8))

# "Raw",subtitle = 




