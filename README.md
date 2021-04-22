# Manual:

The included Python and R scripts are used to infer TE clades from copy number data as detailed in [publication]. Detailed Jupyter and Rmarkdown notebooks will walk you through implementing the modules, but this README will include brief overview of usage of the packages. 


## Aligning data and producing copy number matrices:

Data used in the [manuscript] was aligned using ConTExt and associated scripts and methods were used to generate copy number and SNP pileup files [citations]. Although we recommend using these methods to generate copy number data for clade inference it is not strictly necessary. Any method that produces allele frequencies from alignments and can estimate copy number from read depth of alignments to TE consensus sequences would be adequate. 

What is necessary is that the allele copy number data for that clade inference pipeline be formatted as a numpy file with the following dimensions S x n+1 x 4. Where S is the number of individuals in the data set, and n is the number of basepairs in the TE consensus. Each element in the 1st dimension of the matrix corresponds to an individual from your dataset. Each element of the 2nd dimension corresponds to a particular basepair position in the TE sequence plus a placeholder at the first position which should be filled with zeros. Each element in the third dimension is a nucleotide in the order [A, T, C, G]. Thus this matrix will tell us for each individual in our dataset what the copy number is at every position and for each possible nucleotide at that position. Our clade inference pipeline requires a matrix formatted in this way to function.

## Clade inference:

The clade inference tool is broken up into two parts: 1. A python script (haploTE.py) that will filter the allele copy number numpy data and output simplified CSVs with the copy number information. 2. An R script that takes in the copy number CSVs and outputs the clade calls. It is also necessary to have sample sheet file for your data set. This file is a two column comma seperated file where the first column is the population of origin of the individual and the second column is the sample ID of that individual in your dataset. This is necessary for computing population level summary statistics for both pre-processing data, and for the final output files. You also need a sample sheet of the designations for each TE you wish to analyze. This file should also be two columns that are comma seperated. The first column is the name of your TE (which should match the numpy file names), and the second column should contain class/subclass information e.g. LTR, LINE, etc. (second column is not strictly necessary for clade inference and may be filled with placeholders). 

## 1) Processing copy number numpy for clade inference:

First extract the minor alleles from the copy number numpy file by using the haploTE.py modules. An example of this implementation is described in SCRIPT1. 

## 2) 

