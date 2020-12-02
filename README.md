# Manual:

The included Python and R scripts are used to infer TE haplotypes from copy number data as detailed in [publication]. Detailed Jupyter notebooks as well as Rmarkdown files will walk you through implementing the modules, but this README will include brief overview of usage of the packages. 


## Aligning data and producing copy number matrices:

Data used in the [manuscript] was aligned using ConTExt [ref] and scripts associated with [HTT] were used to generate copy number and SNP pileup files. Although we recommend using these methods to generate copy number data for haplotype inference it is not strictly necessary. Any method that produces allele frequencies from alignments and can estimate copy number from read depth of alignments to TE consensus sequences would be adequate. 

What is necessary is that the allele copy number data for that haplotype inference pipeline be formatted as a numpy file with the following dimensions S x n+1 x 4. Where S is the number of individuals in the data set, and n is the number of basepairs in the TE consensus. Each element in the 1st dimension of the matrix corresponds to an individual from your dataset. Each element of the 2nd dimension corresponds to a particular basepair position in the TE sequence plus a placeholder at the first position which should be filled with zeros (to account for zero-indexing in Python). Each element in the third dimension is a nucleotide: [A, T, C, G]. Thus this matrix will tell us for each individual in our dataset what the copy number is at every position and for each possible nucleotide at that position. Our haplotype inference pipeline requires a matrix formatted in this way to function.

## Haplotype inference:

The haplotype inference tool is broken up into two parts: 1. A python script (haploTE.py) that will filter the allele copy number numpy data and output simplified CSVs with the copy number information. 2. An R script that takes in the copy number CSVs and outputs the haplotype calls. It is also necessary to have sample sheet file for your data set. This file is a two column tab seperated file where the first column is the ID of each individual in your dataset, and the second column is the name of their population of origin. This is necessary for computing population level summary statistics for both pre-processing data, and for the final output files.

## 1) Processing copy number numpy for haplotype inference:

First we must extract minor alleles from the copy number numpy file 

