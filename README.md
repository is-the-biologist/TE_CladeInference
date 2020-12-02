# TE Haplotype Inference:

The included Python and R scripts are used to infer TE haplotypes from copy number data as detailed in [publication]. Detailed Jupyter notebooks as well as Rmarkdown files will walk you through implementing the modules, but this README will include brief overview of usage of the packages. 

These scripts take in processed copy number data in the form of numpy matrices and tsvs. 

## Aligning data and producing copy number matrices:

Data used in the [manuscript] was aligned using ConTExt [ref] and scripts associated with [HTT] were used to generate copy number and SNP pileup files. Although we recommend using these methods to generate copy number data for haplotype inference it is not strictly necessary. Any method that produces allele frequencies from alignments and can estimate copy number from read depth of alignments to TE consensus sequences would be adequate. 

What is necessary is that the allele copy number data for that haplotype inference pipeline be formatted as a numpy file with the following dimensions S x n+1 x 4. Where S is the number of individuals in the data set, and n is the number of basepairs in the TE consensus. Each element in the 1st dimension of the matrix corresponds to  
