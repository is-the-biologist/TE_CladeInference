#This file contains the scripts that are used to infer subfamily structure via hierarchical clustering of correlated minor alleles
#The main function used to generate these matrices is the pheatmap function.

library('pheatmap')
library('plyr')
library('ggplot2')
library('ggdendro')
library('matrixStats')
library("RColorBrewer")
library('reticulate')
library("viridis")
library("igraph")
library("Polychrome")


construct_linkage <- function(df, linkage, TE_name, names=FALSE, fontsize=8){
  
  #This function will use pheatmap with the no outlier to construct the linkage of the 
  

  
  out<- pheatmap(df, 
                       clustering_distance_cols = 'correlation',
                       clustering_method = "average",
                       cluster_rows = FALSE,
                       color=inferno(50),
                       border_color = FALSE,
                       main = TE_name,
                       show_rownames = FALSE,
                       show_colnames = FALSE,
                       fontsize = 10,
                       silent = TRUE
  )
  
  return(out)
  
}

draw_heatmaps<-function(df, outlier_df, TE_name, dendrogram, names=FALSE, fontsize=10, sample="/Users/iskander/Documents/Barbash_lab/TE_diversity_data/GDL_sample_sheet.csv", color_pal) {
  
  #This function will draw out the final heatmaps that use the dendrogram that we constructed in the get linkage function
  cmat <- cor(outlier_df) #get the correlation matrix

  df <- log10(df + 1) #log transform
  
  sample_sheet <- read.csv(sample, header=F)
  pop_labels <- sample_sheet$V1
  strain_IDs <- sample_sheet$V2
  rownames(df) <- strain_IDs
  
  #User can input color_pal in the function call to match number of populations
  
  
  POPS<- color_pal
  
  names(POPS) <- unique(pop_labels)
  
  assertthat::assert_that(length(color_pal) == length(unique(pop_labels)), msg = "Color palette and number of populations are not equal. Re-assign color_pal or check populations.")
  
  anno_colors <- list(POPULATION= POPS)
  
  annotations <- matrix(data = pop_labels, nrow = dim(sample_sheet)[1], ncol = 1)
  rownames(annotations) <- row.names(df)
  colnames(annotations) <- c('POPULATION')
  annotations <- as.data.frame(annotations)
  
  pop_hmap<-pheatmap(df, 
                       cluster_cols = dendrogram$tree_col,
                       cluster_rows = FALSE,
                       color=inferno(50),
                       border_color = FALSE,
                       annotation_row = annotations,
                       annotation_names_row = FALSE,
                       annotation_colors = anno_colors,
                       main = TE_name,
                       show_rownames = FALSE,
                       show_colnames = FALSE,
                       fontsize = fontsize,
                      silent = TRUE
  )
  
  #construct the correlation matrix and seriate it based on the linkage
  breaksList = seq(-1, 1, by = .05)
  
  haplo_hmap<-pheatmap(cmat, 
                
                cluster_rows = dendrogram$tree_col, cluster_cols = dendrogram$tree_col,
                breaks = breaksList,
                color = colorRampPalette(rev(brewer.pal(n = 7, name = "Spectral")))(length(breaksList)),
                border_color = FALSE,
                main = TE_name,
                show_rownames = names,
                show_colnames = FALSE,
                fontsize = fontsize,
                silent=TRUE
  )
  return(list(pop_hmap, haplo_hmap))
}

cut_dendrogram <- function(len, dendrogram, labs=TRUE, fontsize=5){
  #cut the dendogram at a given length.
  p <- ggdendrogram(dendrogram$tree_col, labels = labs, size=fontsize) + geom_hline(yintercept = len, col='red', linetype = 'dashed')
  
  labels <- sort(cutree(dendrogram$tree_col, h=len))
  
  return(list(labels, p))
  
}

cluster_CN <- function(labels, CN_df, sample='/Users/iskander/Documents/Barbash_lab/TE_diversity_data/GDL_sample_sheet.csv', minSize=T){
  
  #This function will take in our cluster labels from our dendogram trimming function and then will output two data frames one is the the average copy number of each haplotype and the other is a key that will tell us what combination of alleles make up each haplotype. 
  
  nt_dict <- c(1, 2, 3, 4)
  names(nt_dict) <- c("A", "T", "C", "G")
  
  sample_sheet <- read.csv(sample, header=F)
  population_labels <- sample_sheet$V1
  strain_IDs <- sample_sheet$V2
  
  #removes size 1 clusters
  if (minSize) {
    mask<-duplicated(labels, fromLast = TRUE) | duplicated(labels)
    masked_clusters <- labels[mask]
    
  }else{
    masked_clusters <- labels
  }
  
  clustIDs <- unique(masked_clusters)
  
  if (length(clustIDs) == 0){
    
    clustIDs <- c("NULL")
    no_clusters <- TRUE
    
  }
  else{
    no_clusters <- FALSE
  }
  
  named_clustIDs <- paste0('Cluster_', clustIDs)
  
  #Create a table that records the clusterIDs and the alleles that they correspond to them 
  cluster_Matrix <- matrix(nrow = length(clustIDs), ncol = 2)
  cluster_Matrix[,1] <- named_clustIDs
  colnames(cluster_Matrix)<-c('CLUST_ID', 'ALLELES')
  

    #create the data frame where these will be stored
  
  CN_matrix <- matrix(nrow = length(population_labels), ncol = length(clustIDs)+2)

  CN_matrix[,1] <- population_labels
  CN_matrix[,2] <- as.character(strain_IDs)

  colnames(CN_matrix) <- c('POP', 'SAMPLE_ID', named_clustIDs)

  #variance data frame:
  CN_var <- matrix(nrow = length(population_labels), ncol = length(clustIDs)+2)
  CN_var[,1] <- population_labels
  CN_var[,2] <- as.character(strain_IDs)
  
  colnames(CN_var) <- c('POP', 'SAMPLE_ID', named_clustIDs)
  if (no_clusters == TRUE){
    
    #when we find no cluster we should just skip trying to compute the copy number and variance for the clusters and leave NaNs
  }
  else {
  cindex = 1

  for (nt in clustIDs){
    
    #Get the copy number information for the clusters
    
    c_alleles <- names(which(masked_clusters == nt))
    c_desig <- paste(c_alleles, collapse=',')
    cluster_Matrix[cindex, 2] <- c_desig
    clust_mean <- rowMeans(CN_df[c_alleles])
    
    clust_var <- rowVars(as.matrix(CN_df[c_alleles]))
    
    CN_matrix[,cindex+2] <- clust_mean
    CN_var[,cindex+2] <- clust_var
    
    cindex = cindex + 1
  }
  }
  
  cluster_ID.df <- as.data.frame(cluster_Matrix)
  mean_CN.df <- as.data.frame(CN_matrix)
  var_CN.df <- as.data.frame(CN_var)

  for (rw in 3:(length(clustIDs)+2)){
    mean_CN.df[,rw] <- as.numeric(as.character(mean_CN.df[,rw]))
    var_CN.df[,rw] <- as.numeric(as.character(var_CN.df[,rw]))
    } 
  
  return(list(mean_CN.df, cluster_ID.df, var_CN.df))
}

getHaploclusterStats <- function(copyNumber, cluster_labels, TE, global_name="GDL", CN_dir='/Users/iskander/Documents/Barbash_lab/TE_diversity_data/allele_CN/GDL_RUN-11-15-19', sample='/Users/iskander/Documents/Barbash_lab/TE_diversity_data/GDL_sample_sheet.csv'){
  
  #Function to compute some relevant summary statistics 
  
  sample_sheet <- read.csv(sample, header=F)
  population_labels <- sample_sheet$V1
  strain_IDs <- sample_sheet$V2
  
  populations <- unique(population_labels)
  pop_indexes <- list()
  for (p in 1:length(populations)){
    
    #This list of vectors contains the indices of each individual that correspond to each population.
    pop_indexes[[p]] <- which(population_labels == populations[p])
    
  }
  
  
  np <- import('numpy')
  CN_npy <- np$load(file.path(CN_dir, paste0(TE, '_CN.npy')))
  
  #have loaded in the CN numpy now lets calculate the population frequency and global frequency of the haplotypes
  
  nt_dict <- c(1, 2, 3, 4)
  names(nt_dict) <- c("A", "T", "C", "G")
  
 
  pos_CN <- np$sum(CN_npy, axis=as.integer(1) )
  
  global_CN <- np$sum(pos_CN, axis=as.integer(0) )
  
  popfreq_table <- matrix(nrow = dim(cluster_labels)[1], ncol = length(populations)+1)
  colnames(popfreq_table) <- sapply(c(global_name, populations), paste, "FREQ")
  
  mean_CN_table <- matrix(nrow = dim(cluster_labels)[1], ncol = length(populations)+1)
  colnames(mean_CN_table) <- sapply(c(global_name, populations), paste, "MEAN CN")
  
  
  c = 1
  for (clust in cluster_labels$CLUST_ID){
    
    #Get CN of the haplotype
    haploCN <- get(clust, copyNumber)
    
    #Get the total CN of the positions in the haplotype
    pos <- as.character(cluster_labels$ALLELES[c])
    
    nt <- lapply(strsplit(pos, ','), strsplit, '_')
    
    #haplotype copy number matrix
    haploCN_vector <- vector()
    haploCN_vector <- c(haploCN_vector, sum(haploCN))
    CN_matrix <- matrix(nrow = dim(sample_sheet)[1], ncol = length(nt[[1]])) #add the total CN for global, B, I, N, T, Z pops for the alleles in haplotype 
    for (p in 1:length(nt[[1]])){
      
      haploP <- as.numeric(nt[[1]][[p]][2])
      
      CN_matrix[, p] <- pos_CN[, haploP + 1]
      
    }
    
    
    #Gets means of haplotype cluster copy numbers
    meanCN_vector <- vector()
    meanCN_vector <- c(meanCN_vector, mean(haploCN))
  
    #Calculate the population frequency of the clusters
    pop_freq <- c()
    totalCN <- rowMeans(CN_matrix)
    globalFreq <- sum(haploCN) / sum(totalCN)
    pop_freq <- c(pop_freq, globalFreq)
    
    for (s in 1:length(populations)){ #calculate the frequency for each population
      freq <- sum(haploCN[pop_indexes[[s]]]) / sum(totalCN[pop_indexes[[s]]]) #gets pop frequency
      
      
      if (is.na(freq) ){# check for NaNs
        freq <- 0
      }
      
      pop_freq <- c(pop_freq, freq)
      
      meanCN_vector <- c(meanCN_vector, mean(haploCN[pop_indexes[[s]]])) #gets mean CN for population
    }
    #we sum the copy number across all strains and divide by the copy number of these positions to get a population frequency
    popfreq_table[c,] <- pop_freq
    mean_CN_table[c, ] <- meanCN_vector
    c = c + 1
    
  }
  
  stat_table <- cbind(popfreq_table, mean_CN_table)
  
  return(stat_table)
}

formatTable <- function(CN, clabels, stats_table, sample = '/Users/iskander/Documents/Barbash_lab/TE_diversity_data/GDL_sample_sheet.csv'){
  
  #This function is just going to take the tables that I produced and unite them into one larger table with relevant summary statistics
  #nRows = Cluster length
  #nColumns = alleles(1) + stats_table (pop_size+1 x 2) + size (1) + blank (1) + avg CN per strains (num_strains) 
  sample_sheet <- read.csv(sample, header=F)
  population_labels <- sample_sheet$V1
  
  total_cols = dim(stats_table)[2] + dim(sample_sheet)[1] + 3
  
  clust_table <- matrix(nrow = dim(clabels)[1], ncol = total_cols)
  column_names <- c("Alleles" , colnames(stats_table), "Cluster Size", "Copy Number", as.character(CN$SAMPLE_ID))

  colnames(clust_table) <- column_names
  
  
  
  #add alleles
  clust_table[,1] <- as.character(clabels$ALLELES)
  
  #add copy numbers
  clust_CN <- t(CN)[3:length(CN),]
  
  j <- as.integer(dim(stats_table)[2]+4)
  
  clust_table[, j:dim(clust_table)[2] ] <- clust_CN
  
  #add the statistics
  i <- as.integer(dim(stats_table)[2]+1)
  clust_table[, 2:i] <- stats_table
  
  #get size of each cluster:
  csize <- vector()
  for (nt in lapply(as.character(clabels$ALLELES), strsplit, ',')){
    csize <- c(csize, length(nt[[1]]))
    
  }
  
  clust_table[,as.integer(dim(stats_table)[2]+2)] <- csize
  
  #add column of cluster names for quick look up
 
  label_matrix <- matrix(nrow=dim(clabels)[1], ncol = 1)
  label_matrix[,1] <- as.character(clabels$CLUST_ID)
  colnames(label_matrix) <- 'Cluster ID'
  
  clust_table <- cbind(label_matrix, clust_table)
  
  #return completed matrix:
  clust_table <- as.data.frame(clust_table)
  
  return(clust_table)
}

read_TE_tables<-function(TE_file){
  #function to read in tables because of empty file error handling
  output_DF <- tryCatch(
    {
      #Try catch zero sized file errors  
      TE_df <- read.csv(file=TE_file, header=TRUE)
      #message(TE_file)
      return(TE_df)
    }, 
    
    error = function(cond){ #handle errors and return an empty data frame
      message(paste(TE_file, "is an empty file."))
      return(data.frame())}
  )
  return(output_DF)
}

extractHaplotypes <- function(TE_name, outlier=FALSE, outlier_dir, output_dir, linkage= 'average', minSize = T, dist_cutoff=0.5, hmap_labels=FALSE, dendro_labels=FALSE, plots=TRUE, color_pal=c('#6F0000','#009191','#6DB6FF','orange','#490092'),
                              alleleCN_dir="/Users/iskander/Documents/Barbash_lab/TE_diversity_data/allele_CN/GDL_RUN-11-15-19", sample="/Users/iskander/Documents/Barbash_lab/TE_diversity_data/GDL_sample_sheet.csv"){
  #Function to wrap our analyses into one and then save all of the files into csvs, plots as needed
  TE_desig <- strsplit(basename(TE_name), '\\.')[[1]][1]
  
  if (outlier == FALSE){ #To filter out the outliers we are simply going to read in a dataframe without outliers to call clusters
    input_file <- paste0(TE_name, '.CN.GDL.minor.csv')
    CN_file <- paste0(TE_name, '.CN.GDL.minor.csv') #and use the data frame with outliers to get the copy number of the haplotype clusters
  }
  else{
    input_file <- paste0(outlier_dir, TE_desig, '.CN.GDL.minor.csv')
    CN_file <- paste0(TE_name, '.CN.GDL.minor.csv')
  }
  
  allele_DF <- read_TE_tables(input_file) #without outliers
  allele_CN_DF <- read_TE_tables(CN_file) #This will be used in the downstream population heatmap, but let's use the outlier-less dataframe for the correlation heatmap
  
  #add error handling when tables are empty:
  
  if (dim(allele_DF)[1] == 0){ #This is just gonna acount for how our simulated data actually can be empty for some TEs, like P-element
    
    cluster_table <- matrix(nrow=1, ncol = 2)
    cluster_table[1,1] <- "Cluster_NULL"
    cluster_table[1,2] <- NaN
    colnames(cluster_table) <- c("Cluster ID", "Alleles")
    
    haplo_CN <- matrix(nrow=1, ncol = 2)
    haplo_CN[1,1] <- NaN
    haplo_CN[1,2] <- NaN
    colnames(haplo_CN) <- c("POP", "SAMPLE_ID")
    
  }
  else { # this gets called when have a dataframe with CN data
    
    dendrogram <- construct_linkage(allele_DF, linkage=linkage, TE_name = TE_desig, names = dendro_labels, fontsize = 7)
    
    hmaps<-draw_heatmaps(df=allele_CN_DF, outlier_df = allele_DF, TE_name = TE_desig, dendrogram = dendrogram, names = hmap_labels, sample = sample, color_pal = color_pal)
    
    plotting_dir <- file.path(output_dir,'PLOTS')
    
  
    #Make sure the directory exists before inputting into it
    if (dir.exists(plotting_dir)){
      
      
    }
    else{
      dir.create(plotting_dir)
      
    }
    
    pop_heatmap_out <- file.path(plotting_dir, paste0(TE_desig, '.pop.hmap.png')  )
    haplo_heatmap_out <- file.path(plotting_dir, paste0(TE_desig, '.haplotype.hmap.png')  )
    if (plots == TRUE){
      ggsave(plot = hmaps[[1]], filename = pop_heatmap_out)
      ggsave(plot = hmaps[[2]], filename = haplo_heatmap_out)
    }
    
    cluster_out <- cut_dendrogram(len=dist_cutoff, dendrogram = dendrogram, labs = dendro_labels)
    cluster_labels <- cluster_out[[1]]
    dendro_plot <- cluster_out[[2]]
    dendro_out <- file.path(output_dir ,'PLOTS', paste0(TE_desig, '.dendro.png'))
    
    if (plots == TRUE){
      ggsave(plot = dendro_plot, filename = dendro_out)
    }
    
    
    haplo_list <- cluster_CN(cluster_labels, allele_CN_DF, minSize = minSize, sample=sample)
    haplo_CN <- haplo_list[[1]]
    haplo_desig <- haplo_list[[2]]
    haplo_Var <- haplo_list[[3]]
    
    stats <- getHaploclusterStats(haplo_CN, haplo_desig, TE_desig, CN_dir = alleleCN_dir, sample = sample)
    cluster_table <- formatTable(haplo_CN, haplo_desig, stats, sample = sample)
  }
  
  #write out tables to path
  CN_file <- file.path(output_dir, paste0(TE_desig, '.haplotypeTable.tsv'))
  CN_only <- file.path(output_dir, paste0( TE_desig, '_cluster_CN.tsv'))
  
  write.table(cluster_table, CN_file, sep='\t', row.names = FALSE, quote = FALSE)
  write.table(haplo_CN, CN_only, sep='\t', row.names = FALSE, quote = FALSE)
  
  
}



