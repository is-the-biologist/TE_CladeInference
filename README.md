# ***UNFINISHED***

# Manual:

The included Python and R scripts are used to infer TE clades from copy number data as detailed in [publication]. Detailed Jupyter and Rmarkdown notebooks will walk you through implementing the modules, but this README will include brief overview of usage of the packages. 

## Necessary python libraries:
pandas

numpy

os

seaborn

matplotlib

sys

scipy

scikit-learn

biopython

pysftp

datetime

statsmodels

functools

multiprocessing

pysam

## Necessary R libraries:
pheatmap

plyr

ggplot2

ggdendro

matrixStats

RColorBrewer

reticulate

viridis

igraph

Polychrome


## Aligning data and producing copy number matrices:

Short-read data (link to paper) was aligned using ConTExt (https://github.com/LaptopBiologist/ConTExt) and associated scripts and methods were used to generate copy number and SNP pileup files (https://academic.oup.com/genetics/article-abstract/217/2/iyaa027/6043924). Although we recommend using these methods to generate copy number data for clade inference it is not strictly necessary. Any method that produces allele frequencies from alignments and can estimate copy number from read depth of alignments to TE consensus sequences would be adequate.

It is necessary that the allele copy number data for the clade inference pipeline be formatted as a numpy file with the following dimensions S x n+1 x 4. Where S is the number of individuals in the data set, and n is the number of basepairs in the TE consensus. Each element in the 1st dimension of the matrix corresponds to an individual from your dataset. Each element of the 2nd dimension corresponds to a particular basepair position in the TE sequence plus a placeholder at the first position which should be filled with zeros. Each element in the third dimension is a nucleotide in the order [A, T, C, G]. Thus this matrix will tell us for each individual in our dataset what the copy number is at every position and for each possible nucleotide at that position. Our clade inference pipeline requires a matrix formatted in this way to function.

## Clade inference:

The clade inference tool is broken up into two parts: 1. A python script (haploTE.py) that will filter the allele copy number numpy data and output simplified CSVs with the copy number information. 2. An R script that takes in the copy number CSVs and outputs the clade calls. It is also necessary to have sample sheet file for your data set. This file is a two column comma seperated file where the first column is the population of origin of the individual and the second column is the sample ID of that individual in your dataset. This is necessary for computing population level summary statistics for both pre-processing data, and for the final output files. You also need a sample sheet of the designations for each TE you wish to analyze. This file should also be two columns that are comma seperated. The first column is the name of your TE (which should match the numpy file names), and the second column should contain class/subclass information e.g. LTR, LINE, etc. (second column is not strictly necessary for clade inference and may be filled with placeholders). 

### 1) Generate copy number csv:

First extract the minor alleles from the copy number numpy file to a CSVs by using the haploTE.py modules. An example of this implementation is described in cladeInference1.ipynb. The user must define allele filtering parameters: minimum positional sequence diversity, minimum allele population frequency, minimum allele copy number, and minimum number of strains with an allele. The default parameters for these filters are: 0.1, 0.1, 0.5 and 10, respectively. But the optimal value for the user may depend on the TEs being analyzed, the number of samples, and organism.



### 2) 

