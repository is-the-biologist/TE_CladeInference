import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as sp
import sys
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from Bio import SeqIO
from Bio.Blast import NCBIXML
from Bio import Seq
import pysftp
from scipy.spatial.distance import pdist
from multiprocessing import Pool
from sklearn.metrics import jaccard_score
from datetime import date
from scikit_posthocs import posthoc_dunn
from Bio.Align.Applications import ClustalOmegaCommandline
from sklearn.manifold import TSNE
import operator
import sklearn.metrics as metrics
from statsmodels.stats.multitest import multipletests
from collections import Counter
from functools import partial
import pysam


class dataCollection:
    """

    This module might need to be removed before final commit.

    Simple class with methods to sftp data and to process allele CN from CN and SNP pileups.

    """
    def __init__(self):

        self.host = ''
        self.user = ''
        self.password = ''
        self.remote_path = ''
        self.local_path = ''
        self.CN_path = '/Users/iskander/Documents/Barbash_lab/TE_diversity_data/B_CN'
        self.SNP_path = '/Users/iskander/Documents/Barbash_lab/TE_diversity_data/SNP_files/RUN_11-15-19'
        self.alleleCN_path = '/Users/iskander/Documents/Barbash_lab/TE_diversity_data/allele_CN/GDL_RUN-11-15-19'
        self.active_tes = pd.read_csv('/Users/iskander/Documents/Barbash_lab/TE_diversity_data/ACTIVE_TES_internal.tsv',header=None).values[:, 0]
        self.conTExt_path = '/Users/iskander/Documents/Barbash_lab/TE_diversity_data/INTERNAL_DELETIONS/cluster_tables/'
        self.sample_sheet = "/Users/iskander/Documents/Barbash_lab/TE_diversity_data/GDL_samples.tsv"

    def sftp(self, SNP=True):
        """

        SFTP the data from the work station

        :param SNP:
        :return:
        """
        srv = pysftp.Connection(host=self.host, username=self.user, password=self.password)


        for te in self.active_tes:
            if SNP:
                TE = te + '_snps.npy'

                full_local = os.path.join(self.local_path, TE)
                full_remote = os.path.join(self.remote_path, TE)
                srv.get(full_remote, full_local)

        srv.close()

    def calculateAlleleCN(self, TE, save=True, placeholder=True):
        """
        This function will create allele CN files from the SNP pileups and the positional copy number estimates.

        :param TE:
        :return:
        """
        CN_np = np.load(os.path.join(self.CN_path, TE+'_CN.npy'), allow_pickle=True)
        SNP_np = np.load(os.path.join(self.SNP_path, TE+"_snps.npy"), allow_pickle=True)
        TE_allele_CN = []
        if placeholder:
            placeholder = np.zeros(shape=(85,1)) #Because mike keeps changing the formatting of these files on me I have to make weird corrections... >:{
            CN_np = np.hstack((placeholder, CN_np))

        assert CN_np.shape[1] == SNP_np.shape[2], "CN and SNP arrays do not have the same shape."

        for strain in range(CN_np.shape[0]):
            strain_CN = CN_np[strain]
            strain_SNP = SNP_np[strain][1:5,:]
            sumSNP = np.sum(SNP_np[strain][1:5,:], axis=0)
            propSNP = strain_SNP / sumSNP
            allele_CN = propSNP * strain_CN

            TE_allele_CN.append(allele_CN)
        TE_allele_CN = np.asarray(TE_allele_CN)
        TE_allele_CN = np.nan_to_num(TE_allele_CN, 0)
        TE_allele_CN[np.isinf(TE_allele_CN)] = 0
        if save:
            filename=os.path.join(self.alleleCN_path, '{0}_CN.npy'.format(TE))
            np.save(filename, TE_allele_CN)
        else:
            return TE_allele_CN

    def getLTR_CN(self):
        LTR_index = np.load('/Users/iskander/Documents/Barbash_lab/TE_diversity_data/LTR_I.npy', allow_pickle=True)
        for row in LTR_index:
            interal_CN = np.load(os.path.join(self.CN_path, row[0]+'_CN.npy'))
            LTR_snps = np.load(os.path.join(self.SNP_path, row[1] + '_snps.npy'))
            alleleCN_internal = np.load(os.path.join(self.alleleCN_path, row[0]+'_CN.npy'))
            med_CN = np.median(interal_CN, axis=1)
            TE_FULL_CN = []
            for strain in range(LTR_snps.shape[0]):
                strain_SNP = LTR_snps[strain][1:5,:]
                sumSNP = np.sum(LTR_snps[strain][1:5, :], axis=0)
                propSNP = strain_SNP / sumSNP
                allele_CN = propSNP * med_CN[strain]
                full_alleleCN = np.hstack((alleleCN_internal[strain], allele_CN[:,1:])) #slice away place holder value

                TE_FULL_CN.append(full_alleleCN)

            TE_FULL_CN = np.asarray(TE_FULL_CN)
            TE_FULL_CN = np.nan_to_num(TE_FULL_CN, 0)
            TE_FULL_CN[np.isinf(TE_FULL_CN)] = 0

            newName = row[0].replace('_I', '_FULL')

            filename=os.path.join(self.alleleCN_path, '{0}_CN.npy'.format(newName))
            np.save(filename, TE_FULL_CN)

    def tandemJcts(self, SatDNA):

        """

        Use the tandem jcts from the cluster table to create a CN numpy file for the satellite DNA. Satellite DNA do not
        have enouch concordant reads to calculate positional CN via read depth. Instead we can use the tandem jcts. which
        are a suitable proxy for copy number, or number of repeat units.

        :param SatDNA:
        :return:
        """

        jct_table = pd.read_csv(os.path.join(self.conTExt_path, f"{SatDNA}.tsv"), sep='\t')
        strains = pd.read_csv(self.sample_sheet, sep='\t', header=None)[0]
        tandems = jct_table.loc[jct_table["Feature Type"] == "Tandem"][strains] #get all the tandem junction CN

        CN_total = np.sum(tandems, axis=0).values

        #get shape the of CN array
        cons_length = np.load(os.path.join(self.SNP_path, f"{SatDNA}_SNPs.npy")).shape[2]

        CN_matrix = np.full(fill_value=np.nan, shape=(len(CN_total), cons_length))
        for col in range(cons_length):
            CN_matrix[:,col] = CN_total

        #output and save the new CN matrix
        np.save(os.path.join(self.CN_path, f"{SatDNA}_CN.npy"), CN_matrix)

class summary_stats:
    """

    Class module to contain the methods to calculate relevant summary statistics for alleles, Pi and Dispersion.

    """
    def __init__(self, sample_sheet='TEST_DATA/GDL_sample_sheet.csv'):
        self.path = 'TEST_DATA/GDL_RUN-11-15-19/'
        self.pi = 'TEST_DATA/Pi'
        self.SNP_path= '/Users/iskander/Documents/Barbash_lab/TE_diversity_data/SNP_files/RUN_11-15-19/'
        self.strains = pd.read_csv(sample_sheet, header=None)

        pops = list(dict.fromkeys(self.strains[0]))

        self.pop_indexes = [list(np.where(self.strains[0].values == p)[0]) for p in pops] + [
            [i for i in range(self.strains.shape[0])]]

    def calcPi(self, TE, indiv_strains=False, nan=False, save=True):


        # Load the numpy files

        SNPs = np.load(os.path.join(self.path, TE + '_CN.npy'))

        pi_vector = []

        if indiv_strains == False:

            # Perform Pi calculations
            for population in self.pop_indexes:
                pop_SNPs = SNPs[population]

                pi = self.pop_Pi(pop_SNPs, nan)
                pi_vector.append(pi)

        else:
            #calculate pi for each strain in each TE
            for p in range(85):
                pop_SNPs = SNPs[p]
                pi = self.pop_Pi(pop_SNPs, indiv_strains=indiv_strains, nan=nan)
                pi_vector.append(pi)
        if save:
            np.save(os.path.join(self.pi, f"{TE}_pi.npy"), np.asarray(pi_vector))
        else:
            return np.asarray(pi_vector)

    def pop_Pi(self, pop_SNPs, min_CN=0.5, indiv_strains=False, nan=False):
        # Sum the CN per allele across all strains in the population

        if indiv_strains == False:
            pop_SNPs[np.where(pop_SNPs <= min_CN)] = 0
            pop_CN = np.sum(pop_SNPs, axis=0)
            total_CN = np.sum(pop_CN, axis=0)
            CN_AF = pop_CN / total_CN
        else: #do the same but make special cases for dimension of one strain
            pop_SNPs[np.where(pop_SNPs <= min_CN)] = 0
            pop_CN = np.sum(pop_SNPs, axis=0)
            CN_AF = pop_SNPs / pop_CN

        # Calculate Pi
        pi = 1 - np.sum(CN_AF ** 2, axis=0)

        if nan:
            pass
        else:
            pi[np.isnan(pi)] = 0

        return pi

    def LRT(self, CN_vector):
        # Take in a vector that contains copy number for a given minor allele and then compute likelihood ratio test

        # We can test Dispersion directly from a chisquare distribution
        # D = Sum([X_i - X_bar]**2 / X_bar)
        D = np.sum((CN_vector - np.average(CN_vector)) ** 2 / np.average(CN_vector))

        # Instant chi square distribution
        p_value = sp.chi2.sf(df=len(CN_vector) - 1, x=D)  # one tailed test for over dispersion

        return p_value

    def poissdisp_test(self, CN_vector):
        """
        this implementation is from the Rfast poissdisp.test package it gives similar results to my LRT function, but
        I think is more conservative which is good.

        The citation for this formulation of the test is:

        Yang Zhao, James W. Hardin, and Cheryl L. Addy. (2009). A score test for overdispersion in Poisson regression based on the generalized Poisson-2 model. Journal of statistical planning and inference 139(4): 1514-1521.

        """

        n = len(CN_vector)
        m = np.average(CN_vector)
        up = (n - 1) * np.var(CN_vector) - n * m

        stat = up / np.sqrt(2 * n * m ** 2)

        p_value = sp.norm.sf(stat)

        return p_value

    def all_dispersion_alleles(self, TE, min_CN=0.5):

        pop_indexes = [
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
            [15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33],
            [34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52],
            [53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70],
            [71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84], [i for i in range(85)]]
        # Load the numpy files
        allele_CN = np.load(os.path.join(self.path, TE + '_CN.npy'))
        allele_CN[np.where(allele_CN <= min_CN)] = 0

        dispersion_matrix = np.zeros(shape=(len(pop_indexes), 3, 4, allele_CN.shape[2]))
        ind = 0
        for p in pop_indexes:
            # For each GDL population:

            pop_CN = allele_CN[p]

            # Get Dispersion test p-value
            p_values = np.apply_along_axis(self.poissdisp_test, axis=2, arr=pop_CN.T).T

            dispersion_matrix[ind][2] = p_values

            # Calc avg and var CN
            avg_CN = np.average(pop_CN.T, axis=2).T
            var_CN = np.var(pop_CN.T, axis=2).T
            dispersion_matrix[ind][0] = avg_CN
            dispersion_matrix[ind][1] = var_CN
            ind += 1

        return dispersion_matrix

    def SFS(self, TE, min_CN=0.5):
        """

        Calculate the site frequency spectrum from the GDL data so that we can analyze the distribution of the alleles.

        :param TE:
        :param min_CN:
        :return:
        """


        CN = np.load(os.path.join(self.path, TE+"_CN.npy"))
        CN[CN <= min_CN] = 0 #remove low CN alleles that are likely errors

        popCN = np.sum(CN, axis=0)
        AF = popCN / np.sum(popCN, axis=0)
        AF[np.isnan(AF)] = 0

        #Remove the major allele:
        maj = np.argmax(AF, axis=0)

        minorAl = np.concatenate([ AF.T[i][[m for m in range(4) if m != maj[i]]] for i in range(AF.shape[1])]) #get only minor alleles and flatten matrix
        AF_vector = np.log10(minorAl[minorAl > 0])

        #plot the SFS:
        with sns.axes_style("whitegrid"):
            sns.distplot(a=AF_vector, bins=25, kde=False)
            plt.xlabel("log10(Freq)")
            plt.ylabel("Number of alleles")
            plt.title(TE)
            plt.show()
            plt.close()

class subfamilyInference:
    """

    Large worker class that performs analysis of SNPs in TEs


    """
    def __init__(self, TE_list='TEST_DATA/ACTIVE_TES_full.tsv', sample_sheet='TEST_DATA/GDL_sample_sheet.csv'):
        self.path = ""
        self.div_path = 'TEST_DATA/Pi'
        self.CN_path = 'TEST_DATA/GDL_RUN-11-15-19'
        #self.active_tes = pd.read_csv(active_int,header=None).values[:, 0]
        self.active_fullLength = pd.read_csv(TE_list, header=None).values[:, 0]
        self.CN_df_path = 'TEST_DATA/CN_tables/FULL'
        self.AP_df_path = 'TEST_DATA/AP_tables'
        self.strains = pd.read_csv(sample_sheet, header=None)
        self.SNP_path = "/Users/iskander/Documents/Barbash_lab/TE_diversity_data/SNP_files/RUN_11-15-19/"
        self.color_map = ['#6F0000','#009191','#6DB6FF','orange','#490092']

        #construct pop indexes:
        pops = list(dict.fromkeys(self.strains[0]))
        self.pop_indexes = [ list(np.where(self.strains[0].values == p)[0]) for p in pops] + [[i for i in range(self.strains.shape[0])]]

        self.strain_hash = {"B":"Beijing", "I":"Ithaca", "N":"Netherlands", "T":"Tasmania", "Z":"Zimbabwe"}

    def KW(self, data):

        s, p = sp.kruskal(data[0], data[1], data[2], data[3], data[4])

        return s, p

    def seqAbundance(self, TE, alleles, save_plot=False, output_dir='/Users/iskander/Documents/Barbash_lab/TE_diversity_data/ALLELE_PLOTS'):
        pop_indexes = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
                       [15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33],
                       [34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52],
                       [53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70],
                       [71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84]]


        CN_np = np.load(os.path.join(self.CN_path, TE + "_CN.npy"))
        NT_encode = {'A': 0, 'T': 1, 'C': 2, 'G': 3}
        NT, positions = zip(*[[NT_encode[nt.split('_')[0]], int(nt.split('_')[1])] for nt in alleles])
        positions = list(positions)
        total_CN = [[] for p in range(len(alleles))]
        for p in pop_indexes:  # iterate through the populations
            pop_CN = CN_np[p].T[positions]
            sumCN = np.sum(pop_CN, axis=1)

            a = 0
            for CN in sumCN:  # iterate through the alleles of interest
                total_CN[a].append(CN)
                a += 1

        names = ['B', 'I', 'N', 'T', 'ZW']
        with sns.axes_style('whitegrid'):

            fig = plt.figure(figsize=(12, 12))
            if len(alleles) % 2 == 0:
                axs = fig.subplots(int(len(alleles) / 2), 2, sharex=True)
            else:
                axs = fig.subplots(len(alleles), sharex=True)

            for pos in range(len(alleles)):
                if len(alleles) % 2 == 0:
                    # create a mapping function to the subplots
                    y = [0 for z in range(int(len(alleles) / 2))] + [1 for v in range(int(len(alleles) / 2))]
                    x = [i % 2 for i in range(len(alleles))]
                    ax_plot = axs[x[pos], y[pos]]

                else:
                    ax_plot = axs[pos]

                P_val = self.KW(total_CN[pos])[1]

                sns.stripplot(data=total_CN[pos], ax=ax_plot, jitter=.2)
                sns.boxplot(data=total_CN[pos], ax=ax_plot, saturation=.2)
                ax_plot.set_title("Position {0}".format(positions[pos]), fontsize=25)
                ax_plot.set_ylabel('Copy number', fontsize=20)
                ax_plot.annotate('p = {0:.4f}'.format(P_val), xy=self.get_axis_limits(ax_plot), fontsize=15)
                ax_plot.set_xticklabels(names, fontsize=25)

            if save_plot == False:
                plt.show()
                plt.close()
            else:
                plot_name = '{0}_posCN.png'.format(TE)
                output = os.path.join(output_dir, plot_name)
                plt.savefig(output, dpi=300)
                plt.close()

    def get_axis_limits(self, ax, scale=.9):
        return ax.get_xlim()[1] * scale, ax.get_ylim()[1] * scale

    def alleleAbundance(self, TE, alleles, save_plot=False, output_dir='/Users/iskander/Documents/Barbash_lab/TE_diversity_data/ALLELE_PLOTS'):
        pop_indexes = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
                       [15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33],
                       [34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52],
                       [53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70],
                       [71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84]]


        CN_np = np.load(os.path.join(self.CN_path, TE + "_CN.npy"))
        NT_encode = {'A': 0, 'T': 1, 'C': 2, 'G': 3}
        NTs = ['A', 'T', 'C', 'G']
        NT, positions = zip(*[[NT_encode[nt.split('_')[0]], int(nt.split('_')[1])] for nt in alleles])
        positions = list(positions)

        total_CN = [[] for p in range(len(alleles))]
        for p in pop_indexes:  # iterate through the populations
            pop_CN = CN_np[p].T[positions]

            a = 0
            for position in NT:  # iterate through the alleles of interest
                total_CN[a].append(pop_CN[a][position])
                a += 1

        names = ['B', 'I', 'N', 'T', 'ZW']
        with sns.axes_style('whitegrid'):
            fig = plt.figure(figsize=(12, 12))
            if len(alleles) % 2 == 0:
                axs = fig.subplots(int(len(alleles)/2), 2, sharex=True)
            else:
                axs = fig.subplots(len(alleles), sharex=True)

            for pos in range(len(alleles)):
                stat, P_val = self.KW(total_CN[pos])

                if len(alleles) % 2 == 0:
                    #create a mapping function to the subplots
                    x = [z for z in range(int(len(alleles)/2))] + [z for z in range(int(len(alleles)/2))]
                    y = [i%2 for i in range(len(alleles))]

                    ax_plot = axs[x[pos], y[pos]]

                else:
                    ax_plot = axs[pos]

                sns.stripplot(data=total_CN[pos], ax=ax_plot, jitter=.2)
                sns.boxplot(data=total_CN[pos], ax=ax_plot, saturation=.2)
                ax_plot.set_title("{1} at position {0}".format(positions[pos], NTs[NT[pos]]), fontsize=25)
                ax_plot.set_ylabel('Copy number', fontsize=20)
                ax_plot.annotate('p = {0:.4f}'.format(P_val), xy=self.get_axis_limits(ax_plot), fontsize=15)
                ax_plot.set_xticklabels(names, fontsize=25)

            if save_plot == False:
                plt.show()
                plt.close()
            else:
                plot_name = '{0}_alleleCN.png'.format(TE)
                output = os.path.join(output_dir, plot_name)
                plt.savefig(output, dpi=300)
                plt.close()

    def pw_alleleCorr(self, TE, alleles, CN_filter=0.5,  Major_allele = False, legend="full", output_dir = '/Users/iskander/Documents/Barbash_lab/TE_diversity_data/ALLELE_PLOTS', save_plot=False):

        """
        perform pairwise correlation of alleles

        :param TE:

        :return:
        """
        pop_indexes = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
                       [15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33],
                       [34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52],
                       [53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70],
                       [71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84]]

        strains = ['Beijing', 'Ithaca', 'Netherlands', 'Tasmania', 'Zimbabwe']
        #create the population legend:
        pop_legend = []
        for p in range(5):
            s_pop = [strains[p] for s in pop_indexes[p]]
            pop_legend = pop_legend + s_pop

        CN_np = np.load(os.path.join(self.CN_path, TE + "_CN.npy"))
        CN_np[np.where(CN_np < CN_filter)] = 0
        NT_encode = {'A': 0, 'T': 1, 'C': 2, 'G': 3}

        NT, positions = zip(*[[NT_encode[nt.split('_')[0]], int(nt.split('_')[1])] for nt in alleles])
        positions = list(positions)

        pw_allele = np.full(fill_value=np.nan, shape=(3, 85))

        for snp in range(len(positions)):
            allele_CN = CN_np[0:85].T[positions[snp]][NT[snp]]
            pw_allele[snp] = allele_CN

        if Major_allele == True:
            assert len(alleles) == 1
            pw_allele[1] = self.get_majorAllele(TE=TE, position=positions[0])
            alleles = alleles + ['Major allele']
        correlation = np.corrcoef(pw_allele[0], pw_allele[1])[0,1]

        pw_allele = np.asarray(pw_allele, dtype='object')
        pw_allele[2] = pop_legend

        df_alleles = pd.DataFrame(pw_allele.T, columns=alleles + ['Population'])

        reg = LinearRegression().fit(pw_allele[0].reshape(-1, 1), pw_allele[1].reshape(-1, 1))
        inter = reg.intercept_[0]
        corr = reg.coef_[0]
        with sns.axes_style('whitegrid'):
            fig = plt.figure(figsize=(12,8))
            sns.scatterplot(data=df_alleles, x=alleles[0], y=alleles[1], hue='Population', s=100, legend=legend, edgecolor=None, alpha=.8, palette=self.color_map)
            plt.text(x=max(df_alleles[alleles[0]]) * 1/2, y=max(df_alleles[alleles[1]]) * 3/4, s="r = {0:.2f}".format(correlation), fontsize=20)
            self.abline(slope=corr, intercept=inter)
            plt.xlabel('{0} copy number'.format(alleles[0]), fontsize=25)
            plt.ylabel('{0} copy number'.format(alleles[1]), fontsize=25)
            plt.xticks(size=20)
            plt.yticks(size=20)
            plt.legend(fontsize=20, markerscale=2.5, edgecolor="black")
            if save_plot == False:
                plt.show()
                plt.close()
            else:
                plt_name = '{0}_{1}.{2}_corr.png'.format(TE, alleles[0], alleles[1])
                out = os.path.join(output_dir, plt_name)
                plt.savefig(out, dpi=300)
                plt.close()

    def TE_CN(self, TE, saveplot=False, output_dir='/Users/iskander/Documents/Barbash_lab/TE_diversity_data/ALLELE_PLOTS'):
        """

        compute the over all copy number of a TE

        :param TE:
        :return:
        """
        pop_indexes = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
                       [15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33],
                       [34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52],
                       [53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70],
                       [71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84]]

        strains = ['Beijing', 'Ithaca', 'Netherlands', 'Tasmania', 'Zimbabwe']
        # create the population legend:
        pop_legend = []
        for p in range(5):
            s_pop = [strains[p] for s in pop_indexes[p]]
            pop_legend = pop_legend + s_pop
        pop_legend = np.asarray(pop_legend, dtype='object')
        CN_np = np.load(os.path.join(self.CN_path, TE + "_CN.npy"))

        #CN_np = CN_np[0:85].T[500:(CN_np.shape[2]-500)].T #we exclude the first and last 1000 bps for mappability issues

        avgCN = np.nanmean(np.nansum(CN_np, axis=1), axis=1)

        CN_df = np.vstack((np.asarray(avgCN, dtype='object'), pop_legend)).T

        CN_df = pd.DataFrame(CN_df, columns=['Copy number', 'POPS'])

        with sns.axes_style('whitegrid'):
            fig = plt.figure(figsize=(10, 8))
            sns.boxplot(data=CN_df, x='POPS', y='Copy number', hue_order=strains, palette=self.color_map)
            #sns.stripplot(data=CN_df, x='POPS', y='Copy number',hue='POPS', jitter=.2, hue_order=strains, size=10)
            plt.xlabel('', fontsize=25)
            plt.ylabel('Copy Number', fontsize=22.5)
            plt.xticks(fontsize=20)
            plt.yticks(fontsize=20)
            #fig.legend(fontsize=20, markerscale=2.5, edgecolor="black")
            #plt.title('{0} copy number'.format(TE), fontsize=25)
            if saveplot == False:
                plt.show()
                plt.close()
            else:
                plotname = '{0}_avgCN.png'.format(TE)
                plt.savefig(os.path.join(output_dir, plotname), dpi=300)
                plt.close()

        return CN_df

    def get_majorAllele(self, TE, position):

        CN_np = np.load(os.path.join(self.CN_path, TE + "_CN.npy"))

        Maj = np.argmax(np.sum(CN_np[0:85].T[position], axis=1))
        Maj_CN = CN_np[0:85].T[position][Maj]

        return Maj_CN

    def PCA_plot(self, TE, outdir='/Users/iskander/Documents/Barbash_lab/TE_diversity_data/PCA_PLOTS', save_plot=False):
        # PCA

        #load in allele proportion matrix:
        alleles = pd.read_csv(os.path.join(self.AP_df_path, f"{TE}.AP.GDL.minor.csv"))

        #read in sample sheet
        sample_sheet = pd.read_csv("/Users/iskander/Documents/Barbash_lab/TE_diversity_data/GDL_samples.tsv", header=None)
        populations = [self.strain_hash[name[0]] for name in sample_sheet[0]]

        # on every strain within the GDL:
        pcs = PCA(n_components=2).fit(alleles)


        EV = pcs.explained_variance_ratio_

        # Pull out the alleles that explain the highest proportion of the variance of PC1
        loading = pcs.components_

        max_alleles = np.argsort(loading[0, :])[::-1][0:5]
        NT_headings = ','.join(alleles.columns[max_alleles])

        print('Most variance on PC1 is explained by: {0}'.format(NT_headings))

        components = pcs.transform(alleles)

        PCA_matrix = {"PC1":components[:,0], "PC2":components[:,1], "Population":populations}

        with sns.axes_style("whitegrid"):
            fig = plt.figure(figsize=(8, 8))
            axs = fig.subplots(1)
            sns.scatterplot(data=PCA_matrix, x="PC1", y="PC2", hue="Population", palette=self.color_map, edgecolor=None, alpha=1, s=75, ax=axs)
            axs.set_aspect('equal', adjustable='box')

            plt.xticks(fontsize=15)
            plt.xlabel(f"PC1 ({EV[0]*100:0.2f}%)", fontsize=20)
            plt.ylabel(f"PC2 ({EV[1]*100:0.2f}%)", fontsize=20)
            #plt.title(f"{TE}", fontsize=20)
            plt.yticks(fontsize=15)
            plt.legend(fontsize=15, markerscale=1.5, edgecolor="black")
            if save_plot == False:
                plt.show()
                plt.close()
            else:
                plot_name = '{0}_PCA.png'.format(TE)
                plot_out = os.path.join(outdir, plot_name)
                plt.savefig(plot_out, dpi=300)
                plt.close()

        return NT_headings.split(',')

    def CNAP_extraction(self, TE, pi_filter=.1, CN_filter=0.5, minFreq=0.1, min_strains=10):

        """

        This is a majorly important function that will extract the allele CN of the TEs and export them into a csv so
        I can pass to my R module that performs the clustering and haplotype marker calling.

        :param TE:
        :param pi_filter:
        :param CN_filter:
        :param Major:
        :param AP:
        :param min_strains:
        :param min_CN:
        :return:
        """
        pi_data = np.load(os.path.join(self.div_path, TE+'_pi.npy'))
        NTs = np.asarray(['A', 'T', 'C', 'G'], dtype="object")
        pi_mask = np.asarray([], dtype=int)

        for p in range(len(self.pop_indexes)): #Get all sites within inter pop diversity > .1 or dataset wide pop div > .1
            tmp_mask = np.where(pi_data[p] > pi_filter)
            pi_mask = np.union1d(pi_mask, tmp_mask)

        allele_CN = np.load(os.path.join(self.CN_path, TE+'_CN.npy'))
        passed_alleles = allele_CN.T[pi_mask].T
        passed_alleles[np.where(passed_alleles < CN_filter)] = 0
        #sanitize data:
        passed_alleles[np.isinf(passed_alleles)] = 0
        passed_alleles[np.isnan(passed_alleles)] = 0
        consensus = np.sum(passed_alleles, axis=0)
        Major = np.argmax(consensus, axis=0)
        CN_strain = np.sum(passed_alleles, axis=1)

        #Get CN of position at population level to filter by minimum AF downstream

        population_CN = np.full(fill_value=np.nan, shape=(len(self.pop_indexes), passed_alleles.shape[2]))

        for p in range(len(self.pop_indexes)):
            population_CN[p,:] = np.sum(CN_strain[self.pop_indexes[p]], axis=0)
        pop_freq_mask = np.asarray([], dtype=bool)
        #Construct data matrix to hold allele CN
        CN_matrix = np.full(fill_value=np.nan, shape=(allele_CN.shape[0], passed_alleles.shape[2]*3))
        AP_matrix = np.full(fill_value=np.nan, shape=(allele_CN.shape[0], passed_alleles.shape[2]*3))
        colnames = np.asarray([], dtype='object')
        row = 0
        for NT in range(passed_alleles.shape[2]):

            minor_alleles = [n for n in range(4) if n != Major[NT]]
            #Get the NT labels
            labels = NTs[minor_alleles]
            colnames = np.concatenate((colnames, np.asarray([l+'_'+ str(pi_mask[NT]) for l in labels], dtype='object')))

            #Get CN
            minorCN = passed_alleles.T[NT, minor_alleles]
            CN_matrix[:,row:row+3] = minorCN.T

            #Get AP
            minorAP = minorCN / CN_strain.T[NT]
            AP_matrix[:,row:row+3] = minorAP.T
            row = row + 3

            #calculate the population frequency of the allele
            pop_freqs = np.full(fill_value=np.nan, shape=(len(self.pop_indexes), 3))
            freq_mask = np.asarray([False, False, False])
            for p in range(len(self.pop_indexes)):
                pop_freqs[p,:] = np.sum(minorCN.T[self.pop_indexes[p]], axis=0) / population_CN[p][NT]
            freq_mask[np.where(np.sum(pop_freqs > minFreq, axis=0) > 0)] = True
            pop_freq_mask = np.concatenate((pop_freq_mask, freq_mask))


        #Sanitize AP matrix:
        AP_matrix[np.isnan(AP_matrix)] = 0
        AP_matrix[np.isinf(AP_matrix)] = 0


        #constructed the CN and AP matrix now let's convert to pandas dataframes
        CN_df = pd.DataFrame(data=CN_matrix, columns=colnames)
        AP_df = pd.DataFrame(data=AP_matrix, columns=colnames)

        #population frequency filtering
        CN_df = CN_df[CN_df.columns[pop_freq_mask]]
        AP_df = AP_df[AP_df.columns[pop_freq_mask]]

        #filter by a minimum number of strains with that allele, this should reduce spurrious correlation
        min_strains_filter = np.where(CN_df.astype(bool).sum(axis=0) >= min_strains)  # take only sites where you have at least 10 strains that contain your variant
        CN_df = CN_df[CN_df.columns[min_strains_filter]]
        AP_df = AP_df[AP_df.columns[min_strains_filter]]

        #Write to buffer
        CN_name = '{0}.CN.GDL.minor.csv'.format(TE)
        pd.DataFrame.to_csv(CN_df, path_or_buf=os.path.join(self.CN_df_path, CN_name), index=False)
        AP_name = '{0}.AP.GDL.minor.csv'.format(TE)
        pd.DataFrame.to_csv(AP_df, path_or_buf=os.path.join(self.AP_df_path, AP_name), index=False)

    def abline(self, slope, intercept, axs=None, col="black"):
        """Plot a line from slope and intercept"""

        if axs == None:
            axes = plt.gca()
            x_vals = np.array(axes.get_xlim())
            y_vals = intercept + slope * x_vals
            plt.plot(x_vals, y_vals, '--', color=col)

        else:

            x_vals = np.array(axs.get_xlim())
            y_vals = intercept + slope * x_vals
            axs.plot(x_vals, y_vals, '--', color=col)

    def correlate_TE_fams(self, TE_1, TE_2, plot=True, minor=False, correlation_method='pearson'):

        wd = self.CN_df_path

        TE_1_df = pd.read_csv(os.path.join(wd, TE_1))
        TE_2_df = pd.read_csv(os.path.join(wd, TE_2))

        if minor == True:
            sum1 = TE_1_df.sum()
            sum2 = TE_2_df.sum()

            # horrible code
            minor_alleles = {}
            for al in sum1.index:
                # remove minor allele
                NT = al.split('_')[1]
                CN = sum1[al]
                if NT not in minor_alleles.keys():
                    minor_alleles[NT] = [CN, al.split('_')[0]]
                else:
                    if minor_alleles[NT][0] < CN:
                        minor_alleles[NT] = [CN, al.split('_')[0]]
            TE_filtered = []
            for k in minor_alleles.keys():
                TE_filtered.append(minor_alleles[k][1] + '_' + k)

            TE_1_df = TE_1_df[TE_filtered]

            ###################
            minor_alleles = {}
            for al in sum2.index:
                # remove minor allele
                NT = al.split('_')[1]
                CN = sum2[al]
                if NT not in minor_alleles.keys():
                    minor_alleles[NT] = [CN, al.split('_')[0]]
                else:
                    if minor_alleles[NT][0] < CN:
                        minor_alleles[NT] = [CN, al.split('_')[0]]
            TE_filtered = []
            for k in minor_alleles.keys():
                TE_filtered.append(minor_alleles[k][1] + '_' + k)

            TE_2_df = TE_2_df[TE_filtered]

        ######

        # Get the length of each of the alleles that correspond to the TE families so we can track the indexing
        TE_2_alleles = TE_2_df.shape[1]
        TE_1_alleles = TE_1_df.shape[1]

        # null_corr = np.corrcoef(TE_1_df.T, TE_2_df.T)[:TE_1_alleles, TE_2_alleles:]

        bindTEs = pd.concat([TE_1_df, TE_2_df], axis=1, ignore_index=True)

        corr_matrix = bindTEs.corr(method=correlation_method).values

        # Now we have a correlation matrix that represents all alleles correlated with each other

        # We can slice the matrix into quadrants to get the correlations for:
        # TE1 x TE1 correlation: Q1
        # TE1 x TE2 correlation: Q2
        # TE2 x TE2 correlation: Q3

        Q1 = corr_matrix[0:TE_1_alleles, 0:TE_1_alleles].flatten()
        Q2 = corr_matrix[0:TE_1_alleles, TE_1_alleles:TE_1_alleles + TE_2_alleles].flatten()
        Q3 = corr_matrix[TE_1_alleles: TE_1_alleles + TE_1_alleles, TE_1_alleles:TE_1_alleles + TE_2_alleles].flatten()
        # print('{0} is maximum correlation in SNPs of unrelated TEs'.format(max(Q2)))

        if plot == True:
            legend = ['{0} vs {0}'.format(TE_1.split('.')[0]),
                      '{0} vs {1}'.format(TE_1.split('.')[0], TE_2.split('.')[0]),
                      '{0} vs {0}'.format(TE_2.split('.')[0])]
            with sns.axes_style('whitegrid'):
                fig = plt.figure(figsize=(10, 6))
                axs = fig.subplots(3, sharex=True)
                sns.distplot(Q1, kde=False, ax=axs[0], color='red')

                sns.distplot(Q2, kde=False, ax=axs[1], color='green')

                sns.distplot(Q3, kde=False, ax=axs[2], color='blue')

                plt.figlegend(legend)
                axs[0].axvline(np.average(Q1), color='red', linestyle='--')
                axs[1].axvline(np.average(Q2), color='green', linestyle='--')
                axs[2].axvline(np.average(Q3), color='blue', linestyle='--')

                plt.show()
                plt.close()

        return Q1, Q2, Q3

    def plot_null(self, DF, crit_value=99.99, saveplot=False, outputdir='/Users/iskander/Documents/Barbash_lab/TE_diversity_data/NULL_CORRELATIONS'):

        #corr_cutoff = np.percentile(null, crit_value)

        #test_perc = sp.percentileofscore(test, corr_cutoff, 'weak')
        #null_perc = sp.percentileofscore(null, corr_cutoff, 'weak')

        with sns.axes_style('whitegrid'):
            fig = plt.figure(figsize=(6, 8))

            sns.boxplot(data=DF, x="DISTR", y="DATA", order=["Test", "Null"], hue_order=["Test", "Null"], palette=["#FF5600", "#00C886"], showfliers=False)

            plt.ylabel('r', fontsize=25)
            plt.xlabel("", fontsize=25)
            plt.xticks(fontsize=20)
            plt.yticks(fontsize=20)
            plt.tight_layout()
            if saveplot == False:
                plt.show()
                plt.close()
            else:
                plt.savefig(os.path.join(outputdir, 'corr_null.png'), dpi=300)
                plt.close()

    def permuteSNPs(self, TE, permutations=1000, save=False, outlier=False):
        """

        Alternative to pairwise comparison version of null distribution. This method computes pairwise permutation of SNPs
        within a TE.

        :param permutations:
        :return:
        """
        permuted_distr = []
        null_wd = '/Users/iskander/Documents/Barbash_lab/TE_diversity_data/NULL_CORRELATIONS'
        for i in range(permutations):
            if not outlier:
                SNPs = pd.read_csv(f"/Users/iskander/Documents/Barbash_lab/TE_diversity_data/CN_dataframes/RUN_1-27-20/FULL/{TE}.CN.GDL.minor.csv")
            else:
                SNPs = pd.read_csv(f"/Users/iskander/Documents/Barbash_lab/TE_diversity_data/CN_dataframes/RUN_1-27-20/NO_OUTLIERS/{TE}.CN.GDL.minor.csv")

            permuted_SNPs = np.asarray([np.random.permutation(S) for S in SNPs.values.T]).T
            perm_corr = np.corrcoef(permuted_SNPs.T)

            SNP_corrs = perm_corr[np.tril_indices(perm_corr.shape[0], k=-1)]

            if len(permuted_distr) == 0:
                permuted_distr = SNP_corrs
            else:
                permuted_distr = np.concatenate((permuted_distr, SNP_corrs))

        corr_distrs = os.path.join(null_wd, TE + '.permut_null_corr.npy')
        test_corr = SNPs.corr()
        test_distr = test_corr.values[np.tril_indices(test_corr.shape[0], k=-1)]
        np_distr = np.asarray([test_distr, permuted_distr])

        if save:
            np.save(corr_distrs, np_distr)
        else:
            return np_distr

    def computePW_nulls(self, save=True, correlation_method='pearson'):


        null_wd = '/Users/iskander/Documents/Barbash_lab/TE_diversity_data/NULL_CORRELATIONS'

        completed_pw = []

        for TE in self.active_fullLength:
            for TE2 in self.active_fullLength:
                pair = sorted([TE, TE2])
                pwTE = TE2 + '.CN.GDL.minor.csv'
                if pair not in completed_pw:
                    quadrants = self.correlate_TE_fams('{0}.CN.GDL.minor.csv'.format(TE), pwTE, plot=False, minor=False,
                                                       correlation_method=correlation_method)
                    completed_pw.append(pair)
                    #save individual numpys for each pairwise comparision
                    if save:
                        null = quadrants[1]
                        np.save(os.path.join(null_wd, f"{'_vs_'.join(pair)}.null.npy"), null)

                else:
                    pass

    def plot_pwTE_nulls(self, TE_distribs, TE_name, crit=95, FDR = "bonf", saveplot=False, outdir='/Users/iskander/Documents/Barbash_lab/TE_diversity_data/NULL_CORRELATIONS/'):

        #convert numpy to dataframe
        all_data = np.concatenate((TE_distribs[0], TE_distribs[1]))
        labels = ['Test' for t in range(len(TE_distribs[0]))] + ['Null' for n in range(len(TE_distribs[1]))]
        TE_df = pd.DataFrame({'DATA':all_data, 'DISTRIBUTION':labels})

        # Get critical values for percentiles
        if FDR == "bonf":
            corrected_alpha = (100 - crit) / len(TE_distribs[0])
            crit = 100 - corrected_alpha
            crit_corr = np.percentile(TE_distribs[1], crit)
            test_percentile = sp.percentileofscore(TE_distribs[0], crit_corr, 'weak')

        elif FDR == "BH": #use a BH correction to calculate the critical percentiles rather than the Bonferonni method

            #Method is probably not working correctly this should be re-evaluated
            test_distr = np.asarray(sorted(TE_distribs[0], reverse=True))
            null = np.asarray(sorted(TE_distribs[1], reverse=True))
            emp_p = 1 - (sp.rankdata(null) / len(null))

            BH_correction = multipletests(pvals=emp_p, method="fdr_bh", alpha= (100-crit)/100, is_sorted=True )


            crit_corr = min(null[BH_correction[0] == True])
            test_percentile = sp.percentileofscore(test_distr, crit_corr, 'weak')
            crit = sp.percentileofscore(TE_distribs[1], crit_corr)

        elif FDR == "none":
            #FDR correction computed manually
            crit_corr = np.percentile(TE_distribs[1], crit)
            test_percentile = sp.percentileofscore(TE_distribs[0], crit_corr, 'weak')

        with sns.axes_style('whitegrid'):
            fig = plt.figure(figsize=(8, 6))
            sns.boxplot(data=TE_df, x='DISTRIBUTION', y='DATA', hue='DISTRIBUTION', whis=[100-crit, crit], saturation=1)
            #sns.violinplot(data=TE_df, x='DISTRIBUTION', y='DATA', hue='DISTRIBUTION', inner='quartile')
            #sns.stripplot(data=TE_df, x='DISTRIBUTION', y='DATA', hue='DISTRIBUTION', jitter=0.2, edgecolor=None)
            plt.title(TE_name, fontsize=25)
            plt.xticks([0, 1], ['Test', 'Null'], fontsize=20)
            plt.yticks(fontsize=15)
            plt.ylabel('Correlation', fontsize=25)
            if crit_corr <= 0.7:
                plt.annotate('{0:.2f} correlation\n{1:.2f}%-ile of Test\n{2:.2f}%-ile of Null'.format(crit_corr, test_percentile, crit), xy=(0.3, crit_corr+.05), fontsize=20)
            else:
                plt.annotate(
                    '{0:.2f} correlation\n{1:.2f}%-ile of Test\n{2:.2f}%-ile of Null'.format(crit_corr, test_percentile, crit),
                    xy=(0.3, crit_corr - .2), fontsize=20)
            plt.xlabel(None)
            plt.axhline(crit_corr, linestyle='--', color='red')
            #plt.axhline(0.5, linestyle='--', color='green')
            if saveplot:
                plt.savefig(os.path.join(outdir, TE_name+'_null.png'), dpi=300)
                plt.close()
            else:
                plt.show()
                plt.close()

        return crit_corr

    def Null_to_DF(self, subsample=1000, extension="permut_null_corr"):
        """
        Construct the a data frame with all of the null distributions for our TEs

        :return:
        """
        full_data = np.asarray([])
        data_labels = []
        distr_labels = []
        for TE in self.active_fullLength:
            TE_distr = np.load('/Users/iskander/Documents/Barbash_lab/TE_diversity_data/NULL_CORRELATIONS/{0}.{1}.npy'.format(TE, extension), allow_pickle=True)
            null = np.random.choice(a=TE_distr[1], size=subsample, replace=False)
            test = TE_distr[0]
            full_data = np.concatenate((full_data, null, test))

            labels = [TE for p in range(len(null) + len(test))]
            data_labels = data_labels + labels
            distr_labels = distr_labels + ["Null" for n in range(len(null))] + ["Test" for i in range(len(test))] #add distribution labels



        null_DF = pd.DataFrame({'DATA':full_data, 'TE':data_labels, 'DISTR':distr_labels})

        return null_DF

    def compareNulls(self, null_df, crit=99.5, saveplot=False, outputdir='/Users/iskander/Documents/Barbash_lab/TE_diversity_data/NULL_CORRELATIONS'):
        """
        Make boxplot of all null distributions for all TEs

        :param null_df:
        :return:
        """

        #re-order by mean r in test distr:
        means = []
        names = []
        for g in null_df.groupby("TE"):
            M = np.average(g[1][g[1]["DISTR"] == "TEST"]["DATA"])
            means.append(M)
            names.append(g[0])

        order = np.asarray(names)[np.argsort(means)]

        with sns.axes_style('whitegrid'):
            fig = plt.figure(figsize=(10, 14))
            sns.boxplot(data=null_df, y='TE', x='DATA', hue="DISTR", whis=[100-crit, crit], showfliers=False, hue_order=["Test", "Null"], palette=["#FF5600", "#00C886"], order=order)
            #plt.axvline(0.5, linestyle='--', color='red')
            plt.xticks(fontsize=20)
            plt.yticks(fontsize=20)
            plt.xlabel('r', fontsize=25)
            plt.ylabel('TE', fontsize=25)
            plt.legend(fontsize=15, edgecolor="black", title="Distribution", title_fontsize=15)
            plt.tight_layout()
            if saveplot == False:
                plt.show()
                plt.close()
            else:
                plt.savefig(os.path.join(outputdir, 'all_pw_nulls.png'), dpi=300)
                plt.close()

    def drop_outliers(self, TE, strain_masks):
        """
        Remove the outliers from the dataframe so that the haplotype inference method doesn't completely break.

        :param TE:
        :param outlier:
        :return:
        """

        outlier = list(np.searchsorted(self.strains[1], strain_masks))

        filename = '{0}.CN.GDL.minor.csv'.format(TE)
        CN_df = pd.read_csv(os.path.join(self.CN_df_path, filename))

        CN_df = CN_df.drop(axis=0, index=outlier)
        pass_filter = np.union1d(np.where(CN_df.astype(bool).sum(axis=0) >= 10), np.where(CN_df.sum(axis=0) >= 5))
        CN_df = CN_df[CN_df.columns[pass_filter]]
        outlier_path = os.path.join(os.path.split(self.CN_df_path)[0], 'NO_OUTLIERS', '{0}.CN.GDL.minor.csv'.format(TE))

        CN_df.to_csv(path_or_buf=outlier_path, index=None)

class haplotypeAnalysis:

    def __init__(self):
        self.active_tes = pd.read_csv('/Users/iskander/Documents/Barbash_lab/TE_diversity_data/ACTIVE_TES_internal.tsv', header=None)
        self.haplo_path='/Users/iskander/Documents/Barbash_lab/TE_diversity_data/HAPLOTYPE_CLUSTERS/HAPLOTYPE_CALL_07-17-2020'
        self.haplo_stats_path = '/Users/iskander/Documents/Barbash_lab/TE_diversity_data/HAPLOTYPE_CLUSTERS/STATS/'
        self.active_fullLength = pd.read_csv('/Users/iskander/Documents/Barbash_lab/TE_diversity_data/ACTIVE_TES_full.tsv', header=None)
        self.CN_path = '/Users/iskander/Documents/Barbash_lab/TE_diversity_data/allele_CN/GDL_RUN-11-15-19'
        self.color_map = ['#6F0000', '#009191', '#6DB6FF', 'orange', '#490092']
        self.NT_encode = {'A': 0, 'T': 1, 'C': 2, 'G': 3}
        self.div_path = '/Users/iskander/Documents/Barbash_lab/TE_diversity_data/seq_diversity_numpys/RUN_1-27-20'
        self.pop_indexes = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
                       [15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33],
                       [34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52],
                       [53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70],
                       [71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84]]
        self.embl = "/Users/iskander/Documents/Barbash_lab/TE_diversity_data/pN_pS_folder/fixed_EMBL_files/ACTIVE_EMBL"
        self.clean_tes = "/Users/iskander/Documents/Barbash_lab/TE_diversity_data/ACTIVE_TES.clean.csv"

        #generate clean TE dictionary
        clean_names = pd.read_csv(self.clean_tes, header=None)[0]
        self.TE_nameTable = {self.active_fullLength[0][n]:clean_names[n] for n in range(self.active_fullLength.shape[0])}


    def plot_haplotypeDispersion(self, te, title="", saveplot=False, plot=False):

        stats = summary_stats()

        haplo_CN = pd.read_csv(os.path.join(self.haplo_path, te + '_cluster_CN.tsv'), sep='\t')
        haplotypeTable = pd.read_csv(os.path.join(self.haplo_path, te + '.haplotypeTable.tsv'), sep='\t')
        cluster_CN = np.asarray(haplo_CN.values[:, 2:], dtype='float')

        if cluster_CN.shape[1] == 0:
            cluster_CN = np.asarray(haplo_CN.values[:, 2], dtype='float')
            p = stats.poissdisp_test(cluster_CN)
            bonf = -np.log10(0.05)
            clust_num = 1
            mean = np.average(cluster_CN)
            var = np.var(cluster_CN)
        else:
            p = np.apply_along_axis(stats.poissdisp_test, arr=cluster_CN, axis=0)
            bonf = -np.log10(0.05 / len(p))
            clust_num = len(p)
            mean = np.log10(np.average(cluster_CN, axis=0))
            var = np.log10(np.var(cluster_CN, axis=0))

        underdisp = -np.log10(1 - p)
        reject_poiss = np.where(underdisp > bonf)
        overdisp = np.where(-np.log10(p) > bonf)
        rejections = np.full(fill_value='Dispersed', shape=clust_num, dtype='object')
        rejections[reject_poiss] = 'Underdispersed'
        rejections[overdisp] = 'Overdispersed'

        # IoD = np.log10(var / mean)

        cn_DF = np.full(fill_value=np.nan, shape=(clust_num, 5), dtype='object')

        cn_DF[:, 0] = mean
        cn_DF[:, 1] = var
        cn_DF[:, 2] = underdisp
        cn_DF[:, 3] = rejections
        cn_DF[:,4] = haplotypeTable["Cluster ID"].values
        cn_DF = pd.DataFrame(cn_DF, columns=['MEAN', 'VAR', 'P', 'Poisson fit', 'Cluster_ID'])
        pal = ["#F8000C", "#FCA800", "#6706A9"]
        if plot:
            with sns.axes_style('whitegrid'):
                fig = plt.figure(figsize=(10, 8))
                sns.scatterplot(data=cn_DF, x='MEAN', y='VAR', hue='Poisson fit', palette=pal,
                                    hue_order=['Overdispersed', 'Dispersed', 'Underdispersed'], alpha=1, edgecolor=None, s=75)
                subfam_method = subfamilyInference()

                subfam_method.abline(1, 0)
                plt.xticks(fontsize=20)
                plt.yticks(fontsize=20)
                plt.title(title, fontsize=25)
                plt.xlabel('log10 Mean', fontsize=22.5)
                plt.ylabel('log10 Var', fontsize=22.5)
                plt.legend(fontsize=15, markerscale=1.5, edgecolor="black")
                if saveplot == False:
                    plt.show()
                    plt.close()
                else:
                    plt.savefig(os.path.join(self.haplo_path, 'PLOTS', te+'_dispPlot.png'), dpi=300)
                    plt.close()
        else:
            return cn_DF

    def populationHaplotypes(self, TE, cluster_ID, saveplot=False):

        clusters = pd.read_csv(os.path.join(self.haplo_path, TE+'_cluster_CN.tsv'), sep='\t')
        stat, pval = self.KW_haploCN(TE, cluster_ID)
        seqAnalysis = subfamilyInference()
        with sns.axes_style('whitegrid'):
            fig = plt.figure(figsize=(10, 8))
            g = sns.boxplot(data=clusters, x='POP', y=cluster_ID, saturation=1, palette=self.color_map)
            #sns.stripplot(data=clusters, x='POP', y=cluster_ID, jitter=.2, s=10, color="black")
            plt.xlabel(None)
            plt.ylabel('Copy Number', fontsize=22.5)
            #plt.title("{0}: Haplotype {1}".format(TE, cluster_ID), fontsize=20)
            x, y = seqAnalysis.get_axis_limits(g)
            #g.annotate('KW-test: -log2(p) = {0:.4f}'.format(-np.log2(pval)), xy=[x*(3/4), y*(7/8)], fontsize=18)

            plt.xticks(fontsize=20, labels=["Beijing", "Ithaca", "Netherlands", "Tasmania", "Zimbabwe"], ticks=[0,1,2,3,4])
            plt.yticks(fontsize=20)
            if saveplot:
                plt.savefig(os.path.join(self.haplo_path, "PLOTS", "{0}_{1}.popCN.png".format(TE, cluster_ID)), dpi=300)
                plt.close()
            else:
                plt.show()
                plt.close()

    def haplotypePCA(self, TE, decomp="PCA", ncol=2, fsize=15):

        cluster_file = os.path.join(self.haplo_path, "{0}_cluster_CN.tsv".format(TE))
        cluster_CN = pd.read_csv(cluster_file, sep='\t')
        CN = cluster_CN.values[:,2:].T
        cluster_list = cluster_CN["SAMPLE_ID"]
        cluster_names = cluster_CN.columns[2:]

        if CN.shape[1] >= 2:

            if decomp == "PCA":
                pcs = PCA(n_components=min(2, CN.shape[1])).fit(CN)
                EV = pcs.explained_variance_ratio_

                # Pull out the alleles that explain the highest proportion of the variance of PC1
                loading = pcs.components_
                max_alleles = np.argsort(loading[0, :])[::-1][0:5]
                cluster_headings = ','.join(cluster_list[max_alleles])

                components = pcs.transform(CN)
                #pops = cluster_CN['POP'].values.reshape(85, 1)


                clustDF = pd.DataFrame(components, columns=['PC1', 'PC2'])
                clustDF["Haplotypes"] = cluster_names
                cmap = plt.get_cmap('jet')
                colors = cmap(np.linspace(0, 1.0, len(clustDF["Haplotypes"]) ))
                with sns.axes_style('whitegrid'):
                    fig = plt.figure(figsize=(12,8))

                    #PC1 v PC2
                    s1 = sns.scatterplot(data=clustDF, x='PC1', y='PC2', hue="Haplotypes",  s=150, edgecolor="black", legend='full', palette=colors)
                    plt.xlabel('PC1 ({0:.2f}%)'.format(EV[0]*100), fontsize=20)
                    plt.ylabel('PC2 ({0:.2f}%)'.format(EV[1]*100), fontsize=20)
                    plt.xticks(fontsize=15)
                    plt.yticks(fontsize=15)

                    plt.legend(edgecolor="black", fontsize=fsize, title_fontsize=fsize, markerscale=2, ncol=ncol)

            elif decomp == "tSNE":
                components = TSNE().fit_transform(CN)

                clustDF = pd.DataFrame(components, columns=['PC1', 'PC2'])
                clustDF["Haplotypes"] = cluster_names

                with sns.axes_style('whitegrid'):
                    fig = plt.figure(figsize=(14, 8))

                    # PC1 v PC2
                    s1 = sns.scatterplot(data=clustDF, x='PC1', y='PC2', hue="Haplotypes", s=75, edgecolor=None,
                                         legend='full')
                    plt.xlabel('tSNE 1', fontsize=20)
                    plt.ylabel('tSNE 2', fontsize=20)



        else:
            sys.stdout.write('Number of clusters called less than two in {0}\n'.format(TE))

    def correlateHaplotypes(self, TE, popMean=True, totalCN=False, correlation="pearson"):
        """

        This method will correlate the "Major" haplotype with the called haplotype to see if there are positive or negative correlations between the major and minor variants.
        This is to determine whether these haplotypes represent variants expanding in populations and displacing the other haplotypes.

        :return:
        """

        haplotype_file = os.path.join(self.haplo_path, TE+'.haplotypeTable.tsv')
        hapTable = pd.read_csv(haplotype_file, sep='\t')
        clustIDs = hapTable['Cluster ID']

        #get the major alleles for each position:
        haplotype_correlations = []
        for haplo in clustIDs:
            corr = self.haploCorr(TE, cluster=haplo, popMean=popMean, totalCN=totalCN, correlation=correlation)
            haplotype_correlations.append(corr)

        return haplotype_correlations, clustIDs.values

    def KW_haploCN(self, TE, cluster):

        """

        Performs Kruskal-wallis test on copy number distributions for each haplotype cluster called. This will give us an
        idea of the differences in copy number between the populations so we can get some idea of the summary statistics involved.

        :return:
        """
        #read in cluster_CN table:
        cluster_file = os.path.join(self.haplo_path, "{0}_cluster_CN.tsv".format(TE))
        cluster_CN = pd.read_csv(cluster_file, sep='\t')
        CN_data = cluster_CN.groupby('POP')
        cluster_data = []
        for p in CN_data[cluster]:
            cluster_data.append(p[1].values)
        seqAnalysis = subfamilyInference()

        stat, pval = seqAnalysis.KW(cluster_data)

        return stat, pval

    def compute_statistic(self, KW=False,  filter_haplos=0, Fst=False):
        """

        Function to wrap the computing of our test statistics and then plot them out in a reasonable way.

        :param KW:
        :param Fst:
        :return:
        """
        KW_statistics = []

        for TE in self.active_fullLength[0]: #Iterate through all TEs
            #we pass to filtering function
            cluster_list = self.filterHaplotypes(TE, minPopFreq=filter_haplos)
            if len(cluster_list) > 0: #if any clusters pass filtering pass to computation step
                for cluster in cluster_list:
                    if KW: # We use the KW_haploCN subroutine to get a kruskal-wallis test statistic and p-value for differences in the population CN
                        s, p = self.KW_haploCN(TE, cluster)

                        KW_statistics.append([s,-np.log2(p),TE,cluster])
                    elif Fst:
                        #calculate Fst statistic
                        self.calculateFst(TE, cluster)
            else: #otherwise just pass
                pass
        if KW and Fst:
            KW_statistics = np.asarray(KW_statistics)
            KW_statistics = pd.DataFrame(KW_statistics, columns=['H', 'P', 'TE', 'Cluster ID'])


            return KW_statistics
        elif KW:
            KW_statistics = np.asarray(KW_statistics)
            KW_statistics = pd.DataFrame(KW_statistics, columns=['H', 'P', 'TE', 'Cluster ID'])

            return KW_statistics

        elif Fst:
            pass

    def calculateFst(self, TE, cluster): #delete this funciton

        #calculate Fst for each population as measure of differentiation between entire GDL
        Pi = np.load(os.path.join(self.div_path, TE+"_pi.npy"))
        cluster_file = os.path.join(self.haplo_path, f"{TE}.haplotypeTable.tsv")
        haplotypeTable = pd.read_csv(cluster_file, sep='\t')
        cluster_position = [int(NT.split("_")[1]) for NT in haplotypeTable.loc[haplotypeTable['Cluster ID'] == cluster]["Alleles"][0].split(",")]

        Fst_matrix = np.full(fill_value=np.nan, shape=(5, len(cluster_position)))

        Fst = lambda x: ( Pi[5][x] - np.mean(Pi[0:5,:][:,x]) ) / Pi[5][x]
        cluster_Fst = [Fst(nt) for nt in cluster_position]
        print(cluster_Fst)

    def plot_KW(self, read_table=True, filter_haplos=0, saveplot=False):

        if read_table:
            KW_statistics = pd.read_csv(os.path.join(self.haplo_stats_path, 'KW.RUN_07-17-2020.csv'))
        else:
            KW_statistics = self.compute_statistic(KW=True, filter_haplos=filter_haplos)

        bonferonni = -np.log2(0.05 / KW_statistics.shape[0])

        KW_statistics.P = KW_statistics.P.astype(float)
        KW_statistics.H = KW_statistics.H.astype(float)

        #compute effect size
        KW_statistics["eta"] = ((KW_statistics["H"].values - 5 + 1) / (85 - 5)).astype(float)
        bin = np.linspace(0, max(KW_statistics['eta']), 20)

        KW_statistics['eta_bin'] = pd.cut(abs(KW_statistics['eta']), bin, precision=0)

        order = np.concatenate([group[1][0].values for group in self.active_fullLength.groupby(1)])
        order = [self.TE_nameTable[u] for u in order]

        clean_class = {"HTT": "Telomeric", "LINE": "LINE-like", "LTR": "LTR retrotransposon", "TIR": "DNA transposon"}
        TE_class = np.asarray([])
        for TE_name in self.active_fullLength.values:
            TE_class = np.concatenate((TE_class,[clean_class[TE_name[1]] for j in range(KW_statistics[KW_statistics["TE"] == TE_name[0]].shape[0])]))

            #TE_class = np.concatenate((TE_class, np.asarray([self.active_fullLength[ self.active_fullLength[0] == group[0] ][1].values[0] for hap in range(group[1].shape[0])]) ) )
        KW_statistics['te_class'] = TE_class

        cleanNames = [self.TE_nameTable[N] for N in KW_statistics["TE"]]
        KW_statistics["TE"] = cleanNames


        diff_percentiles = sp.percentileofscore(KW_statistics['P'], bonferonni)
        print(diff_percentiles)

        with sns.axes_style('whitegrid'):
            fig = plt.figure(figsize=(14, 7), constrained_layout=False)
            axs = fig.subplots(1)

            sns.stripplot(data=KW_statistics, x='TE', y='P', s=5, alpha=0.8, edgecolor='black', jitter=0.2,
                              order=order, ax=axs, dodge=False, linewidth=0.5, hue='te_class',
                          hue_order=["Telomeric", "LINE-like", "LTR retrotransposon", "DNA transposon"], palette=["#430398", "#B40066", "#8BD100", "#E0CC00"])


            sns.boxplot(data=KW_statistics, x='TE', y='P', saturation=0.75, order=order, ax=axs, showfliers=False, color="white")
            axs.axhline(bonferonni, linestyle='--', color='red')

            plt.ylabel('-log2(p)', fontsize=25)
            plt.yticks(fontsize=15)
            plt.xticks(fontsize=15, rotation=90)
            plt.xlabel('')
            plt.legend(fontsize=15, markerscale=2, edgecolor='black')

            fig.tight_layout()
            if saveplot:
                plt.savefig("/Users/iskander/Documents/Barbash_lab/TE_diversity_data/ALLELE_PLOTS/KW_haplotypeClusters.png", dpi=300)
                plt.close()
            else:
                plt.show()
                plt.close()

class PacBio:

        """
        This class handles the analysis of the PacBio alignments
        """

        def __init__(self, active_TE_path='/Users/iskander/Documents/Barbash_lab/TE_diversity_data/ACTIVE_TES_internal.tsv',
                     full_length_path='/Users/iskander/Documents/Barbash_lab/TE_diversity_data/ACTIVE_TES_full.tsv'):

            self.active_tes = pd.read_csv(active_TE_path, header=None)
            self.PacBio_assemblies = ['A4', 'B3', 'A1', 'B1', 'A3', 'A6', 'A7', 'AB8', 'ORE', 'A2', 'A5', 'B4', 'B2', 'B6', "B59", "I23", "N25", "T29A", "ZH26"]
            self.PacBio_path = '/Users/iskander/Documents/Barbash_lab/TE_diversity_data/PacBio/Alignments'
            self.consensus_path = '/Users/iskander/Documents/Barbash_lab/TE_diversity_data/TE_consensus'
            self.alignment_info_path = '/Users/iskander/Documents/Barbash_lab/TE_diversity_data/PacBio/TE_alignment_extract/'
            self.variant_path = '/Users/iskander/Documents/Barbash_lab/TE_diversity_data/PacBio/Variants/'
            self.haplo_path='/Users/iskander/Documents/Barbash_lab/TE_diversity_data/HAPLOTYPE_CLUSTERS/HAPLOTYPE_CALL_07-17-2020'
            self.active_fullLength = pd.read_csv(full_length_path, header=None)
            self.CN_df_Path = "/Users/iskander/Documents/Barbash_lab/TE_diversity_data/CN_dataframes/RUN_1-27-20/FULL/"
            self.PacBio_root = '/Users/iskander/Documents/Barbash_lab/TE_diversity_data/PacBio/'
            self.GDL = ["B59", "I23", "N25", "T29A", "ZH26"]
            self.phylo_path = "/Users/iskander/Documents/Barbash_lab/TE_diversity_data/PacBio/Phylogeny"


        def extract_Alignments(self, TE):

            """

            Wrap the alignment reader script to iterate through all PacBio assemblies

            :param TE:
            :return:
            """

            pop_IDs = []
            all_variants = []
            fasta_headers = []
            consensus_file = os.path.join(self.consensus_path, TE+'.fa')
            for alignment in self.PacBio_assemblies:
                #Iterate through all PacBio alignments
                align_file = os.path.join(self.PacBio_path, TE + '_' + alignment + '.blast.xml')

                variants = self.alignment_reader(xml=align_file, TE_consensus=consensus_file)

                if variants.shape[1] == 0:
                    pass

                else:
                    pop_IDs = pop_IDs + [alignment for pop in range(variants.shape[0])] #add column for which assembly the TE sequence came from
                    fasta_headers = fasta_headers + [f">{TE}_{alignment}_{no+1}" for no in range(variants.shape[0])] #add column for the fasta header of this TE alignment pileup sequence

                    if len(all_variants) == 0:
                        all_variants = variants
                    else:
                        all_variants = np.vstack((all_variants, variants))
            sample_sheet = pd.DataFrame(data={'Headers': fasta_headers, 'Strains':pop_IDs})

            return all_variants, sample_sheet

        def getTE_Seqs(self, xml, genome, TE_name):
            #Simple method that will iterate through an alignment XML and take out start, stop, scaffold data and then create cleaned up fastas
            fasta_extracts = {}
            alignment_info_table = []
            extract_no = 1

            with open(xml, 'r') as myAln:
                blast_records = NCBIXML.read(myAln)
                for alignments in blast_records.alignments:
                    for hsp in alignments.hsps:
                        start_pos = hsp.sbjct_start #start position
                        end_pos = hsp.sbjct_end #end position
                        cleaned_sequence = hsp.sbjct.replace('-', '') #remove the gaps in the PacBio alignment
                        scaffold_id = alignments.title #scaffold name
                        fasta_header = f">{TE_name}_{genome}_{extract_no}"
                        fasta_extracts[fasta_header] = cleaned_sequence
                        tabulated_alignment = [start_pos, end_pos, fasta_header, scaffold_id, TE_name]
                        alignment_info_table.append(tabulated_alignment)
                        extract_no += 1
                myAln.close()
            alignment_info_table = np.asarray(alignment_info_table)

            return alignment_info_table, fasta_extracts

        def extract_Sequences(self, TE):
            """

            Function similar to extract_Alignments but instead we get the actual sequences from the PacBio assembly do to
            our MSA and to do some clustering to merge alignments.

            :param TE:
            :return:
            """
            PacBio_alignment_table = []
            PacBio_fastas = {}

            for alignment in self.PacBio_assemblies:
                #Iterate through all PacBio alignments
                align_file = os.path.join(self.PacBio_path, TE + '_' + alignment + '.blast.xml')
                alignment_data, fastas = self.getTE_Seqs(xml=align_file, genome=alignment, TE_name=TE)
                PacBio_fastas.update(fastas)

                if len(PacBio_alignment_table) == 0:
                    PacBio_alignment_table = alignment_data
                else:
                    if alignment_data.shape[0] == 0:
                        pass
                    else:
                        PacBio_alignment_table = np.vstack((PacBio_alignment_table, alignment_data))

            PacBio_alignment_table = pd.DataFrame(data=PacBio_alignment_table, columns=['Start', 'Stop', 'Header', 'Scaffold', 'TE'])
            PacBio_alignment_table.to_csv(path_or_buf=os.path.join(self.alignment_info_path, f'{TE}.PacBio_alignmentInfo.csv'), index=None)

            #write out Fastas:
            with open(os.path.join(self.alignment_info_path, f'{TE}.PacBio_alignmentExtracts.fa'), 'w') as myFa:

                for header in PacBio_fastas.keys():
                    myFa.write(header+'\n')
                    myFa.write(PacBio_fastas[header] + '\n')
                myFa.close()

        def alignment_reader(self, xml, TE_consensus):

            NT_dict = {"A": 0, "T": 1, "C": 2, "G": 3}
            convert_NT = lambda x: NT_dict[x]
            # Generate a numpy file that contains the conensus sequence coded numerically
            with open(TE_consensus, 'r') as myTE:

                TE = SeqIO.read(myTE, 'fasta')
                TE_seq = np.asarray(list(map(convert_NT, np.asarray(list(TE.seq)))), dtype=float)
                myTE.close()

            variant_TE_pileup = np.asarray([])

            with open(xml, 'r') as myAln:
                blast_records = NCBIXML.read(myAln)
                for alignments in blast_records.alignments:
                    for hsp in alignments.hsps:

                        var_SEQ = np.copy(TE_seq)

                        # Convert strings to numpy arrays to quickly find places where there are mismatches
                        TE_reference_seq = np.asarray(list(hsp.query))
                        PacBio_seq = np.asarray(list(hsp.sbjct))

                        # we now have all positions that would have mismatches/gaps
                        # Let's go through each position and

                        pileup = np.vstack((TE_reference_seq, PacBio_seq)) #we take the alignments side by side and now are going to iterate through the positions


                        #print(len(PacBio_seq))
                        #print(len(TE_reference_seq))
                        #print(len(hsp.match))

                        #print('break')

                        start_index = hsp.query_start-1

                        var_SEQ[:start_index] = np.nan #place nans at the start of the sequence where the alignment begins
                        var_SEQ[hsp.query_end:] = np.nan #place nans at the end of the sequence where the alignment ends

                        consensus_index = start_index

                        for position in range(len(hsp.match)):

                            #let's iterate through every position
                            if hsp.match[position] == ' ':
                                #this is a mismatch
                                if pileup[0][position] == '-':#gap in the TE reference
                                    #we don't add to the index of the consensus sequence and we do not add to the pileup
                                    pass
                                elif pileup[1][position] == '-': #gap in PacBio sequence
                                    var_SEQ[consensus_index] = np.nan #add a NaN to sequence to represent gaps
                                    consensus_index += 1 #add to the consensus index
                                else: #SNP polymorphism
                                    var_SEQ[consensus_index] = NT_dict[pileup[1][position]]
                                    consensus_index += 1
                            else: #we got a match so we need not record any information
                                consensus_index += 1 #we add 1 to the consensus index as we scan along
                                pass


                        if len(variant_TE_pileup) == 0:
                            variant_TE_pileup = var_SEQ
                        else:
                            variant_TE_pileup = np.vstack((variant_TE_pileup, var_SEQ))
                myAln.close()

                # correct for some shape issues:
            if len(variant_TE_pileup.shape) == 1:
                variant_TE_pileup = variant_TE_pileup.reshape(1, variant_TE_pileup.shape[0])
            else:
                pass

            return variant_TE_pileup

        def clusterAlignments(self, TE):
            """
            This method will compute the distances

            :param TE:
            :return:
            """

            alignments = pd.read_csv(os.path.join(self.alignment_info_path, f'{TE}.PacBio_alignmentInfo.csv'))

            for positions in alignments.groupby('Scaffold')['Start', 'Stop']:
                insert_len = round(abs(positions[1]['Start'] - positions[1]['Stop']) / 2)
                midpoint = positions[1]['Start'] + insert_len
                print(midpoint)
                break

        def compute_dist(self):
            """

            compute distance between all insertions for each TE and plot it as a series of boxplots.

            :param TE:
            :return:
            """

            all_distances = np.asarray([])
            all_names = []
            for TE in self.active_tes[0]:
                TE_dists = np.asarray([])
                TE_names = []

                alignments = pd.read_csv(os.path.join(self.alignment_info_path, f'{TE}.PacBio_alignmentInfo.csv'))
                num_alignments = 0
                for positions in alignments.groupby('Scaffold')['Start', 'Stop']:
                    insert_len = round(abs(positions[1]['Start'] - positions[1]['Stop']) / 2)
                    midpoint = positions[1]['Start'].values + insert_len.values
                    midpoint = midpoint[:,None]
                    distance = pdist(midpoint, 'cityblock')
                    TE_dists = np.concatenate((TE_dists, distance))

                TE_names = TE_names + [TE for n in range(TE_dists.shape[0])]

                all_distances = np.concatenate((all_distances, TE_dists))
                all_names = all_names + TE_names
            all_distances = all_distances / 1000000
            distance_df = pd.DataFrame(data={'Distance':all_distances, 'TE':all_names})

            with sns.axes_style('whitegrid'):
                fig = plt.figure(figsize=(12, 16))
                axs = fig.subplots(2, sharex=True, gridspec_kw={'height_ratios': [3, 1]})
                sns.boxplot(data=distance_df, y='TE', x='Distance',  ax=axs[0])
                sns.stripplot(data=distance_df, y='TE', x='Distance', alpha=.5, size=.5, jitter=0.2, color='black', edgecolor=None, ax=axs[0])
                sns.distplot(a=distance_df['Distance'])

                axs[0].set_xlabel('')
                axs[1].set_xlabel('Distance between insertions', fontsize=15)
                axs[0].set_xticklabels(labels=axs[0].get_xticks(), fontsize=15)
                axs[0].set_ylabel('TE', fontsize=25)

                fig.tight_layout()
                plt.show()
                plt.close()

        def PacBio_haplotype_corr(self, cluster, TE_pb, TE_haplo, AF_filter=0, GDL=False):
            """

            Method to calculate the correlations of alleles called from the haplotype inference method but using the PacBio
            fully phased genomes. This can give me an idea of how accurate I actually am at detecting these haplotypes.

            :param TE:
            :param cluster:
            :return:
            """
            NT_dict = {'A':0, 'T':1, 'C':2, 'G':3}
            PacBio_variants = np.load(os.path.join(self.variant_path, f'{TE_pb}.PacBio_variants.npy'))
            haplotypeTable = pd.read_csv(os.path.join(self.haplo_path, f'{TE_haplo}.haplotypeTable.tsv'), sep='\t')
            CN_df = pd.read_csv(os.path.join(self.CN_df_Path, f'{TE_haplo}.CN.GDL.minor.csv'))

            SNPs = [s.split('_') for s in haplotypeTable.loc[haplotypeTable['Cluster ID'] == cluster]['Alleles'].values[0].split(',')]

            haplotype_matrix = {}
            whole_elements = np.asarray([])

            #Parameter for subsetting data by only GDL
            if GDL:
                sample_sheet = pd.read_csv(os.path.join(self.variant_path, f"{TE_pb}.PacBio_samples.csv"))

                PacBio_variants = PacBio_variants[np.isin(sample_sheet["Strains"].values, self.GDL)]

            for allele, position in SNPs:

                #Check to see if the position falls within an LTR of element, for now we skip those positions until we can think of a good way to add in the LTR alignments
                if int(position) > PacBio_variants.shape[1]:
                    pass
                else:
                    numeric_allele = NT_dict[allele]
                    SNP_col = PacBio_variants[:,int(position)-1]
                    SNP_variants = np.zeros(shape=SNP_col.shape)
                    SNP_variants[np.where(SNP_col == numeric_allele)] = 1
                    #For each allele in the cluster we made a column where presence of allele is 1 and absence is 0

                    #Let's search the SNP_col for NaNs and create a mask of those positions so we can remove those alignments
                    #These are elements that had a deletion at our variant position and would be causing issues
                    deletion_mask = np.argwhere(~np.isnan(SNP_col))
                    if len(whole_elements) == 0:
                        whole_elements = deletion_mask
                    else:
                        whole_elements = np.intersect1d(whole_elements, whole_elements)

                    haplotype_matrix[f'{allele}_{position}'] = SNP_variants

            haplotype_df = pd.DataFrame(data=haplotype_matrix)

            if haplotype_df.shape[1] > 1:
                # Ask if there is more than one SNP called that passed criteria
                haplotype_df = haplotype_df.iloc[whole_elements]
                num_elements = haplotype_df.shape[0]

                #this parameter will remove any alleles where found in low copies. This will remove alleles that were perhaps low freq in the population that we only found in GDL but not in PacBio data
                # I think that these alleles are inflating the 0s and in the case of correlations they actually return NaNs
                #We have a tuneable parameter AF_filter that we can change what is the minimum number of alleles discovered to consider this for our validation
                allele_count = np.sum(haplotype_df.values, axis=0) / haplotype_df.shape[0]
                found_alleles = haplotype_df.columns[np.where(allele_count > AF_filter)]
                haplotype_df = haplotype_df[found_alleles]

                if haplotype_df.shape[1] > 1:
                    #lost_SNPs = abs(len(SNPs) - haplotype_df.shape[1])

                    #calculate the true positive rate of the inferred haplotypes in the PacBio data by directly taking the total number of true hits
                    #get the frequency of the haplotype in the dataset
                    true_positive = np.sum((np.sum(haplotype_df.values, axis=1) == haplotype_df.shape[1])) / num_elements#/ np.sum(np.sum(haplotype_df.values, axis=1) > 0)

                    #Calculate Jaccard Scores
                    score_matrix = self.calculate_Jaccard_score(haplotype_df)
                    jaccard_triangle = np.triu_indices(n=score_matrix.shape[0], k=1)
                    jaccard_scores = score_matrix[jaccard_triangle]


                    PacBio_corr = haplotype_df.corr()

                    #Correlations are now calculated for PacBio assemblies we can now take these SNPs and look at our GDL CN dataframe and perform the same sort of analysis:
                    haplo_corr = CN_df[PacBio_corr.columns].corr()

                    #Now we must extract the the triangles of both correlation matrices
                    triangle = np.triu_indices(n=PacBio_corr.shape[0], k=1)

                    assert len(PacBio_corr.values[triangle]) == len(haplo_corr.values[triangle])

                    #return the PacBio correlation values and the Haplotype inference correlation values

                    #calculate the cluster averages of the corr and jaccard scores
                    avg_PBcorr = np.nanmean(PacBio_corr.values[triangle])
                    avg_SRcorr = np.nanmean(haplo_corr.values[triangle])
                    avg_JS = np.nanmean(jaccard_scores)
                    num_snps = haplotype_df.shape[1]
                    pw_PB_snps = PacBio_corr.values[triangle]
                    pw_SR_snps = haplo_corr.values[triangle]

                    return avg_PBcorr, avg_SRcorr, avg_JS, num_snps, num_elements, pw_PB_snps, pw_SR_snps, jaccard_scores, true_positive

                #SORRY THIS CODE IS GROSS BUT I AM LAZY
                else: #when we have only 1 SNP called from our cluster in the PacBio data set we have to handle it and so we return these NaN values
                    return np.nan, np.nan, np.nan, 1, 0, np.asarray([np.nan]), np.asarray([np.nan]), np.asarray([np.nan]), 0
            else:  # when we have only 1 SNP called from our cluster in the PacBio data set we have to handle it and so we return these NaN values
                return np.nan, np.nan, np.nan, 1, 0, np.asarray([np.nan]), np.asarray([np.nan]), np.asarray([np.nan]), 0

        def calculate_Jaccard_score(self, SNP_matrix):
            """

            An alternative to calculating correlation for binary data. This might be a superior metric because it weighs 1,1 more heavily.
            It is not directly comparable to correlation, which kind of sucks, but it is still a nice metric to have. This will throw out less
            data which is good.


            :param SNP_matrix:
            :return:
            """


            score_matrix = np.full(fill_value=np.nan, shape=(SNP_matrix.shape[1], SNP_matrix.shape[1]))
            for snp1 in range(SNP_matrix.shape[1]):
                for snp2 in range(SNP_matrix.shape[1]):

                    score = jaccard_score(SNP_matrix.values[:,snp1], SNP_matrix.values[:,snp2])
                    score_matrix[snp1, snp2] = score

            return score_matrix

        def calculate_PacBio_correlations(self, AF=0, GDL=False):
            """

            This function wraps the PacBio_haplotype_corr function so that it runs on all TEs and haplotypes. This way
            we will construct distributions of PacBio SNP correlations.

            :return:
            """

            PacBio_correlations = []
            haplotype_correlations = []
            jaccard_coeff_vector = []
            TE_IDs = []
            cluster_size = []
            elements = []
            cluster_ids = []
            snp_PacBio = []
            snp_ShortRead = []
            snp_Jaccard = []
            true_pos_list = []

            for TE in range(self.active_fullLength.shape[0]):
                TE_pb = self.active_tes[0][TE]
                TE_haplo = self.active_fullLength[0][TE]
                haplotypeTable = pd.read_csv(os.path.join(self.haplo_path, f"{TE_haplo}.haplotypeTable.tsv"), sep='\t')

                #iterate through clusters
                for cluster in haplotypeTable['Cluster ID']:
                    PB_corr, HT_corr, jaccard_coeffs, num_SNPs, num_elements, pw_PB_snps, pw_SR_snps, pw_jaccard, true_pos = self.PacBio_haplotype_corr(TE_pb=TE_pb, TE_haplo=TE_haplo, cluster=cluster, AF_filter=AF, GDL=GDL)

                    #include all clusters now so we don't upwardly bias our validation

                    jaccard_coeff_vector.append(jaccard_coeffs)
                    PacBio_correlations.append(PB_corr)
                    haplotype_correlations.append(HT_corr)
                    TE_IDs.append(TE_pb)
                    cluster_size.append(num_SNPs)
                    elements.append(num_elements)
                    cluster_ids.append(cluster)
                    snp_PacBio.append(",".join(pw_PB_snps.astype(str))) #each correlation/JS for each SNP is added to dataframe as one row for that cluster
                    snp_ShortRead.append(",".join(pw_SR_snps.astype(str)))
                    snp_Jaccard.append(",".join(pw_jaccard.astype(str)))
                    true_pos_list.append(true_pos)

            assert len(TE_IDs) == len(PacBio_correlations)

            PacBio_df = pd.DataFrame(data={'TE':TE_IDs, 'PacBio Correlations': PacBio_correlations, 'Short-read Correlations':haplotype_correlations,
                                           "Jaccard Score": jaccard_coeff_vector, "PacBio Pairwise": snp_PacBio, "Short-read Pairwise": snp_ShortRead,
                                           "Jaccard Score Pairwise": snp_Jaccard, "True Positives": true_pos_list,
                                           "Cluster Size": cluster_size, "Elements":elements, "Cluster ID":cluster_ids})

            return PacBio_df

        def plot_PacBio_correlations(self, input_df, saveplot=False, colname="True Positives", xlab="Population Frequency"):
            """

            This function takes in the PacBio correlation dataframes and performs a few statistical tests and constructs
            null distributions. This is in order to create some robust analysis on the efficacy of the short-read haplotype
            marker calling method.

            :return:
            """

            #log transform jaccard scores:
            PacBio_df = input_df.copy()


            mean_corr = np.full(fill_value=np.nan, shape=(len(set(PacBio_df['TE'].values)), 2), dtype='object')
            i = 0
            for data in PacBio_df.groupby('TE')[colname]:  # order the groups of TEs from smallest to largest mean correlations in the distribution
                name = data[0]
                corr = np.mean(data[1])
                mean_corr[i, 0] = corr
                mean_corr[i, 1] = name
                i += 1
            order_TE = mean_corr[:, 1][np.argsort(mean_corr[:, 0])]

            #R_50 = 100 - sp.percentileofscore(PacBio_df[colname], 0.5)
            #print(R_50)
            #obsv_R = np.corrcoef(PacBio_df["PacBio Correlations"][~np.isnan(PacBio_df["PacBio Correlations"])], #remove the NaNs that seem to have appeared for some strange reason
            #                     PacBio_df["Short-read Correlations"][~np.isnan(PacBio_df["PacBio Correlations"])])[0,1]
            #print(sp.spearmanr(a=PacBio_df["PacBio Correlations"][~np.isnan(PacBio_df["PacBio Correlations"])], #remove the NaNs that seem to have appeared for some strange reason
                                #b=PacBio_df["Short-read Correlations"][~np.isnan(PacBio_df["PacBio Correlations"])]))
            #print(obsv_R)
            print(np.sum(PacBio_df[colname] > 0) / PacBio_df.shape[0])
    
            with sns.axes_style('whitegrid'):
                fig = plt.figure(figsize=(10, 12), constrained_layout=False)
                axs = fig.subplots(2, sharex=True, gridspec_kw={'height_ratios': [3, 1]})

                sns.boxplot(data=PacBio_df, y='TE', x=colname, ax=axs[0], order=order_TE, showfliers = False)
                sns.stripplot(data=PacBio_df, y='TE', x=colname, ax=axs[0], order=order_TE, color='black', jitter=0.3, s=4)

                axs[0].set_xlabel("")
                axs[0].set_yticklabels(axs[0].get_yticklabels(), size=13)
                axs[0].set_ylabel("TE", fontsize=15)

                sns.distplot(a=PacBio_df[colname], ax=axs[1], kde=False, bins=20)

                axs[1].set_xlabel(xlab, fontsize=15)

                xticks = ["{0:.1f}".format(x) for x in axs[1].get_xticks()]

                axs[1].set_xticklabels(labels=xticks, size=15)
                yticks = [int(y) for y in axs[1].get_yticks()]
                axs[1].set_yticks(yticks)
                axs[1].set_yticklabels(labels=yticks, size=15)
                axs[1].set_ylabel("Number of Haplotypes", fontsize=15)
                plt.tight_layout()

                if not saveplot:
                    plt.show()
                    plt.close()
                else:
                    plt.savefig(os.path.join(self.PacBio_root,"PacBio_validation_js.png"), dpi=300)
                    plt.close()

        def alignmentLengths(self, TE):

            """

            Compute the length of alignments in relation to the full length element in that way we can get a distribution of alignment lengths so
            we can quantify what proportion of the TE sequences are full length or less.

            :return:
            """

            alignmentInfo = "{0}.PacBio_alignmentInfo.csv".format(TE)
            info_table = pd.read_csv(os.path.join(self.alignment_info_path, alignmentInfo))
            alignment_length = abs(info_table["Stop"] - info_table["Start"]).values

            with open(os.path.join(self.consensus_path, TE+".fa"), "r") as consensusSeq:
                for record in SeqIO.parse(consensusSeq, "fasta"):
                    seqLen = len(record.seq)
                consensusSeq.close()

            alignment_props = alignment_length / seqLen

            return alignment_length, alignment_props

        def plot_alignmentLengths(self, p_lim=0.8):
            """

            Wrapper function to plot out lengths of all TEs from PacBio alignments and display them as a series of boxplots
            like what I have done for many of my other things.

            :return:
            """
            alignment_matrix = {"Proportion":np.asarray([]), "TE":[]}
            for TE in self.active_tes[0]:
                L, P = self.alignmentLengths(TE)
                alignment_matrix["Proportion"] = np.concatenate((alignment_matrix["Proportion"], np.log10(P) ))
                alignment_matrix["TE"] = alignment_matrix["TE"] + [TE for name in range(len(P))]

            alignment_df = pd.DataFrame(data=alignment_matrix)

            #data extracted now plot:

           # print(len(np.where(alignment_df["Proportion"] >= np.log10(p_lim))[0]))
            #first organize the order of TEs in the boxplot by mean proportion:
            passed_inserts = []
            order = np.full(fill_value=np.nan, shape=(len(set(alignment_df['TE'].values)), 2), dtype='object')
            i = 0
            for data in alignment_df.groupby('TE')['Proportion']:  # order the groups of TEs from smallest to largest mean proportion of full length
                name = data[0]


                prop =  len(np.where(data[1] >= np.log10(p_lim))[0]) / len(data[1])
                total_num = len(np.where(data[1] >= np.log10(p_lim))[0])
                passed_inserts.append(total_num)

                order[i, 0] = prop
                order[i, 1] = name
                i += 1
            order_TE = order[:, 1][np.argsort(order[:, 0])]
            passed_inserts = np.asarray(passed_inserts)[np.argsort(order[:, 0])]


            #plot it out:
            with sns.axes_style('whitegrid'):
                fig = plt.figure(figsize=(10, 12), constrained_layout=False)
                axs = fig.subplots(2, sharex=True, gridspec_kw={'height_ratios': [3, 1]})
                sns.boxplot(data=alignment_df, x='Proportion', y="TE", ax=axs[0], order=order_TE, showfliers=False, color='lightgrey')
                sns.stripplot(data=alignment_df, x='Proportion', y="TE", ax=axs[0], order=order_TE, color='black', jitter=0.2, size=1)
                sns.distplot(a=alignment_df["Proportion"], kde=False, color="black")

                for sample in range(len(passed_inserts)):
                    axs[0].text(s=f"{passed_inserts[sample]}", y=sample, x=.05, fontsize=10)

                axs[0].set_xlabel("")
                axs[0].set_yticklabels(axs[0].get_yticklabels(), size=13)
                axs[0].set_ylabel("TE", fontsize=15)

                axs[1].set_xlabel("Insertion Length", fontsize=15)

                axs[0].axvline(np.log10(p_lim), color='red', linestyle='--')
                axs[1].axvline(np.log10(p_lim), color='red', linestyle='--')
                #axs[1].set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
                #axs[1].set_xticklabels(labels=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0], size=15)
                #axs[1].set_yticks(axs[1].get_yticks())
                #axs[1].set_yticklabels([0, 2000, 4000, 6000, 8000, 10000, 12000, 14000], size=15)

                fig.tight_layout()
                plt.show()
                plt.close()

        def summarizeValidation(self, saveplot=False):

            #function to make violin/boxplots of all the cluster correlations

            SIM_df = pd.read_csv("/Users/iskander/Documents/Barbash_lab/TE_diversity_data/PacBio/Simulations/SIM_01-27-2020_clusterTest.csv")
            GDL_df = pd.read_csv(os.path.join(self.PacBio_root, "GDL_clusterValidation.csv"))
            DSPR_GDL_df = pd.read_csv(os.path.join(self.PacBio_root, "GDL_clusterValidation.csv"))

            #get the PacBio correlation values into a matrix/df
            correlations = np.concatenate((SIM_df["PacBio Correlations"].values, GDL_df["PacBio Correlations"].values,
                                           DSPR_GDL_df["PacBio Correlations"], DSPR_GDL_df["Short-read Correlations"], SIM_df["Short-read Correlations"]))
            labels = ["Simulation" for s in range(SIM_df.shape[0])] + ["GDL PacBio" for g in range(GDL_df.shape[0])] + [
                "PacBio" for d in range(DSPR_GDL_df.shape[0])] + ["GDL Short-reads" for sr in range(DSPR_GDL_df.shape[0])] + ["Simulation Short-reads" for s in range(SIM_df.shape[0])]

            data_matrix = {"r": correlations, "Validation Set": labels}
            valid_df = pd.DataFrame(data=data_matrix)

            #plot it out
            with sns.axes_style("whitegrid"):
                fig=plt.figure(figsize=(12, 6))
                sns.swarmplot(data=valid_df, x="Validation Set", y="r", order=["PacBio",  "Simulation"], palette=sns.color_palette("colorblind"))
                #sns.stripplot(data=valid_df, x="Validation Set", y="r", order=["PacBio", "GDL Short-reads", "Simulation", "Simulation Short-reads"], color="black", alpha=0.8, edgecolor=None, jitter=0.1, s=3)

                plt.xticks(size=15)
                plt.yticks(size=15)
                plt.ylabel("Haplotype Marker Correlation", size=20)
                plt.xlabel("Validation Method", size=20)
                if saveplot:
                    plt.savefig("/Users/iskander/Documents/Barbash_lab/TE_diversity_data/ALLELE_PLOTS/validation.png", dpi=300)
                    plt.close()
                else:
                    plt.show()
                    plt.close()

        def filterFastas(self, TE, minProp=0.9):

            """

            Function to filter out TE sequence fragment from the fasta files so that we can output full length TEs to
            our MSA, tree-building step.

            :return:
            """
            L, P = self.alignmentLengths(TE)

            alignmentInfo = pd.read_csv(os.path.join(self.alignment_info_path, f"{TE}.PacBio_alignmentInfo.csv"))
            fullLength_inserts = alignmentInfo["Header"].values[np.where(P >= minProp)]
            fullLength_inserts = [insert[1:] for insert in fullLength_inserts]#strip carrot
            TE_records = []
            #got all the headers that pass not lets read in the fastas
            with open(os.path.join(self.alignment_info_path, f"{TE}.PacBio_alignmentExtracts.fa"), "r") as extracts:
                for record in SeqIO.parse(extracts, "fasta"):
                    if record.id in fullLength_inserts:
                        TE_records.append(record)

                extracts.close()

            #all sequences are filtered and extracted
            #let's write them out

            SeqIO.write(TE_records, os.path.join(self.phylo_path, f"{TE}.clean_extract.fa"), "fasta")

        def clustalOmega(self, TE, threads=7):
            """

            Function that will wrap clustalOmega MSA. Takes a fasta from our /Phylogeny directory as input. These fastas
            should be cleaned up so as to only have long alignments, and minimal fragments.

            :param TE:
            :return:
            """

            input_fasta = os.path.join(self.phylo_path, f"{TE}.clean_extract.fa")
            output_fasta = os.path.join(self.phylo_path, f"{TE}.clustalOmega.out.fasta")

            clustal_cmd = ClustalOmegaCommandline(infile=input_fasta, outfile=output_fasta, outfmt="fasta", verbose=True, auto=False, threads=threads)

            return(clustal_cmd)

        def calculateLD(self, TE, TE_pb):
            """

            calculate pairwise LD of haplotype markers by using the PacBio dataset

            :param TE:
            :return:
            """
            NT_dict = {"A":0, "T":1, "C":2, "G":3}

            haploTable = pd.read_csv(os.path.join(self.haplo_path, TE+".haplotypeTable.tsv"), sep='\t')
            variants = np.load(os.path.join(self.variant_path, TE_pb+".PacBio_variants.npy"))

            #unpack SNP clusters
            alleles = np.concatenate([np.asarray([[NT_dict[SNP.split("_")[0]], int(SNP.split("_")[1]) ] for SNP in cluster.split(",")]) for cluster in haploTable["Alleles"].values])

            #get column names and SNPs for data matrix
            SNP_names = np.concatenate(np.asarray([cluster.split(",") for cluster in haploTable["Alleles"].values]))
            SNP_names = np.asarray([SNP_names[i] for i in range(len(SNP_names)) if alleles[i][1] <= variants.shape[1]])
            alleles = np.asarray([S for S in alleles if S[1] <= variants.shape[1]])


            #calculate allele frequencies
            get_AF = lambda S: np.nansum(variants[:,S[1]-1] == S[0]) / variants.shape[0] #func to get allele freq
            AF = np.asarray([get_AF(S) for S in alleles])
            AF_mask = AF > 0
            AF = AF[AF_mask]
            SNP_names = SNP_names[AF_mask]
            alleles = alleles[AF_mask]

            AF_matrix = np.full(shape=(len(AF), len(AF)), fill_value=np.nan)
            complement_AF_matrix = np.full(shape=(len(AF), len(AF)), fill_value=np.nan)
            #compute pairwise matrix of allele frequency products & products of their complement
            for j in range(len(AF)):
                for i in range(len(AF)):
                    AF_matrix[j,i] = AF[j] * AF[i]
                    complement_AF_matrix[j,i] = (1-AF[j]) * (1-AF[i])

            #now we compute pairwise haplotype freqs:
            haplo_freq = np.full(shape=(len(AF), len(AF)), fill_value=np.nan)

            #lambda func to get haplo freq -- get sum of two boolean arrays and get freq of when you have T+T
            get_haploFreq = lambda h1, h2: np.sum( np.asarray(np.asarray(variants[:,h1[1]-1] == h1[0]).astype(int) + np.asarray(variants[:,h2[1]-1] == h2[0]).astype(int)) == 2)  / variants.shape[0] #get a haplotype freq of two SNPs

            for j in range(haplo_freq.shape[0]): #perform the pairwise comparisons
                for i in range(haplo_freq.shape[0]):
                    haplo_freq[j,i] = get_haploFreq(alleles[j], alleles[i])

            #now we compute D based on the haplotype freqs vs the product of allele frequencies
            D = haplo_freq - AF_matrix

            #mask the diagonal bc the math on the diagonal is weird
            #np.fill_diagonal(D, 0)


            D_table = pd.DataFrame(data=D, columns=SNP_names, index=SNP_names)

            return D_table

class simulateHaplotypes:

    """

    This class module will contain the necessary funtions to simulate a set of short read data of TE haplotypes to
    benchmark the haplotype inference approach that I have developed.


    The first set of simulation tools will use PacBio data to create an NGS representation of the data. This is to create
    a biological set of data with a ground truth that we can test against. We are drawing directly from this data here
    so we should be better at reconstructing the true relationships.

    The second set of functions for the simulation are going to create biologically agnostics sets of data. Here we want
    to generate TEs in a branching process where each branch split is a polymorphism accumulating on the TE sequence. We
    then can sample from the tips of the branches in a specified probability distribution. This gives us the ability to
    directly assay


    """

    def __init__(self, active_TE_path='/Users/iskander/Documents/Barbash_lab/TE_diversity_data/ACTIVE_TES_internal.tsv',
                 full_length_path='/Users/iskander/Documents/Barbash_lab/TE_diversity_data/ACTIVE_TES_full.tsv'):
        self.active_tes = pd.read_csv(active_TE_path, header=None)
        self.PacBio_assemblies = ['A4', 'B3', 'A1', 'B1', 'A3', 'A6', 'A7', 'AB8', 'ORE', 'A2', 'A5', 'B4', 'B2', 'B6',
                                  "B59", "I23", "N25", "T29A", "ZH26"]
        self.PacBio_path = '/Users/iskander/Documents/Barbash_lab/TE_diversity_data/PacBio/Alignments'
        self.alignment_info_path = '/Users/iskander/Documents/Barbash_lab/TE_diversity_data/PacBio/TE_alignment_extract/'
        self.variant_path = '/Users/iskander/Documents/Barbash_lab/TE_diversity_data/PacBio/Variants/'
        self.active_fullLength = pd.read_csv(full_length_path, header=None)
        self.PacBio_root = '/Users/iskander/Documents/Barbash_lab/TE_diversity_data/PacBio/'
        self.reference_path = "/Users/iskander/Documents/Barbash_lab/TE_diversity_data/PacBio/Reference_variants"
        self.haplo_path = "/Users/iskander/Documents/Barbash_lab/TE_diversity_data/HAPLOTYPE_CLUSTERS/HAPLOTYPE_CALL_SIM_01-22-2020/"
        self.haplo_root = "/Users/iskander/Documents/Barbash_lab/TE_diversity_data/HAPLOTYPE_CLUSTERS/"
        self.CN_df_Path = "/Users/iskander/Documents/Barbash_lab/TE_diversity_data/CN_dataframes/SIM_01-22-2020"
        self.dummy_path = active_TE_path
        self.consensus_path = '/Users/iskander/Documents/Barbash_lab/TE_diversity_data/TE_consensus'
        self.CN_path = '/Users/iskander/Documents/Barbash_lab/TE_diversity_data/allele_CN/GDL_RUN-11-15-19'

    def reference_pileups(self, TE):

        """

        Method to construct a reference SNP allele proportion pileup for each of the TEs from the PacBio genomes and to
        also create a CN pileup for each of the TEs. This way we can use the CN and AP pileups to stochastically generated
        allele CN pileups in proportion to our PacBio data set, but with some noise.


        :return:
        """

        #Read in PacBio numpy files
        variants = np.load(os.path.join(self.variant_path, f"{TE}.PacBio_variants.npy"), allow_pickle=True)
        sample_sheet = pd.read_csv(os.path.join(self.variant_path, f"{TE}.PacBio_samples.csv"))
        #First let us construct a CN numpy for each genome
        CN_matrix = np.full(fill_value=np.nan, shape=(len(self.PacBio_assemblies), variants.shape[1] + 1))

        strain_index = 0
        for strain in self.PacBio_assemblies:
            strain_insertions = variants[np.where(sample_sheet['Strains'] == strain)]
            strain_CN = np.sum(~np.isnan(strain_insertions), axis=0)
            CN_matrix[strain_index][1:] = strain_CN
            strain_index += 1

        #Now let's make the allele proportion matrices by using the CN for each allele at a site
        strain_index = 0
        allele_proportion_matrix = np.full(fill_value=np.nan, shape=(len(self.PacBio_assemblies), 4, variants.shape[1] + 1))
        for strain in self.PacBio_assemblies:
            strain_insertions = variants[np.where(sample_sheet['Strains'] == strain)]

            for allele in range(4):
                alleleCN = np.sum(strain_insertions == allele, axis=0)
                AP = alleleCN / CN_matrix[strain_index][1:]

                allele_proportion_matrix[strain_index][allele][1:] = AP

            strain_index += 1

        return CN_matrix, allele_proportion_matrix

    def generateReference(self):

        """

        function to wrap our generation of reference pileups for our PacBio data

        :return:
        """

        for TE in self.active_tes[0]:
            CN, AP = self.reference_pileups(TE)

            CN_out = os.path.join(self.reference_path, f"{TE}.PacBio_CN.npy")
            AP_out = os.path.join(self.reference_path, f"{TE}.PacBio_AP.npy")

            np.save(CN_out, CN)
            np.save(AP_out, AP)

    def sim_alleleCN(self, coverage):
        """

        We use this function to create CN estimates for our simulated data based on our reference numpys.

        For a given TE and an Expected read coverage we generate SNP pileups from our PacBio data by using a Poiss ~(E(R)*CN).
        This gives us a random number of reads that would be proportional to the copy number at a position, and to a reads parameter
        in the function. This will output and Obsv(R) value that we can divide by our E(R) to obtain an Obsv(CN), which should be
        an estimate of the true CN of a position. We can then take this Obsv(CN) and multiply it to the allele proportion matrix



        :param coverage: input the target coverage for the libraries
        :return:
        """
        #create the new directory for the simulated data
        simDir = os.path.join(self.PacBio_root, "Simulations", date.today().strftime("%m-%d-%Y"))
        try:#make new directory for simulated data
            os.mkdir(simDir)
        except FileExistsError:
            pass


        #Generate coverage for the libraries
        library_coverage = np.random.poisson(coverage, size=19)

        for TE in self.active_tes[0]:
            alleleCN = self.readSimulator(TE, library_coverage)

            np.save(os.path.join(simDir, f"{TE}.simulated_CN.npy"), alleleCN)

    def readSimulator(self, TE, library_coverage):
        """


        :param TE:
        :param library_coverage:
        :return:
        """

        #load in CN file
        CN = np.load(os.path.join(self.reference_path, f"{TE}.PacBio_CN.npy"), allow_pickle=True)
        lam = (CN.T*library_coverage).T #E(R) * CN) = lambda parameter

        #sanitize our lambda parameter matrix of NaNs
        lam[np.isnan(lam)] = 0

        #Use poisson process to generate reads
        Obsv_R = np.asarray(list(map(np.random.poisson, lam)))

        #Now Obsv(R) / E(R) = estimated CN
        est_CN = (Obsv_R.T / library_coverage).T

        #Now we take our estimated copy number at our positions and we must use our AP matrix to get the allele CN
        AP = np.load(os.path.join(self.reference_path, f"{TE}.PacBio_AP.npy"), allow_pickle=True)
        #sanitize AP dataframe
        AP[np.isnan(AP)] = 0

        alleleCN = np.full(fill_value=np.nan, shape=AP.shape)

        #This process could be more stochastic in its generation
        for strain in range(AP.shape[0]): #Get the allele CN per position given our library
            strain_alleleCN = AP[strain] * est_CN[strain]
            alleleCN[strain] = strain_alleleCN
        #An additional feature could be to draw copies, or reads from a multiN distribution with the parameters of AP matrix
        #That would create a bit more randomness to simulation, but I don't think it is a critical feature

        return alleleCN

    def sampleTEs_from_Phylo(self, fasta, copy_number, strains, target_coverage=12.5, erorr_rate=0.001):

        headers = []
        fasta_seqs = []
        nt_dict = {"a":0, "t":1, "c":2, "g":3}
        DNA_to_num = lambda x: [nt_dict[nt] for nt in x]
        with open(fasta, "r") as myFA:
            for records in SeqIO.parse(myFA, "fasta"):
                headers.append(records.id)
                fasta_seqs.append(records.seq)

        fasta_seqs = np.asarray(fasta_seqs)
        fasta_np = np.apply_along_axis(func1d=DNA_to_num, arr=fasta_seqs, axis=1)

        poissCN = np.random.poisson(copy_number, strains)
        trueCN = poissCN + poissCN*(erorr_rate*4) #add in psuedocounts for error rate

        sample_sheet = {"Headers": [], "Strains": []}
        AP_matrix = np.full(fill_value=np.nan, shape=(strains, 4, fasta_np.shape[1] + 1), dtype=float)

        TE_index = np.arange(start=0, stop=fasta_np.shape[0])

        reorder_val = np.asarray([])

        for S in range(strains):
        #get the TEs that are sampled in each individual from the fastas

            TEs = np.random.choice(a=TE_index, size=poissCN[S], replace=True) #sample w/ replacement

            if len(reorder_val) > 0:
                reorder_val = np.concatenate((reorder_val, TEs))
            else:
                reorder_val = TEs
            #TE_index = np.asarray(list(set(TE_index) - set(TEs)))

            TE_strain_matrix = fasta_np[TEs]

            #add to sample sheet BS
            if len(sample_sheet["Headers"]) > 0:
                sample_sheet["Headers"] = np.concatenate((sample_sheet["Headers"], TEs))
            else:
                sample_sheet["Headers"] = TEs

            sample_sheet["Strains"] = sample_sheet["Strains"] + [S for n in range(poissCN[S])]

            # TEs have been sampled now we create the allele proportion matrices:
            for SNP_col in range(TE_strain_matrix.shape[1]):
                for NT in range(4):

                    #effectively generates a 0.1% error rate by adding in pseudocounts for blank alleles
                    nt_CN = np.sum(TE_strain_matrix[:, SNP_col] == NT) + poissCN[S]*erorr_rate #copy number for this allele by counts of matrix + pseudocounts for error
                    AP_matrix[S, NT, SNP_col + 1] = nt_CN / trueCN[S] #allele proportion

        validation_variants = fasta_np[reorder_val]
        AP_matrix[np.isnan(AP_matrix)] = 0  # sanitize sim data

        # AP matrix is obtained for each strain we can now use poisson distribution to generate observed copy number
        # b/c we are only simulating full length elements
        trueCN = np.asarray(trueCN)
        library_coverage = np.random.poisson(target_coverage, size=strains)  # generate coverage
        obsv_coverage = np.random.poisson(library_coverage * trueCN)
        est_CN = obsv_coverage / library_coverage  # generate obsv_CN as described in sim_alleleCN

        alleleCN = np.full(fill_value=np.nan, shape=AP_matrix.shape)

        # This process could be more stochastic in its generation
        for strain in range(AP_matrix.shape[0]):  # Get the allele CN per position given our library

            #generate allele proportion based on read counts following a multinomial process with error terms equal to illumina sequencing error
            #Thus we can add stochasticity to the read sampling of the simulation modelling some error processes
            AP_randomized = np.asarray([np.random.multinomial(n=obsv_coverage[strain], pvals=pval) / obsv_coverage[strain] for pval in AP_matrix[strain].T ]).T

            #note the first index 0 SNP is a G b/c of the way numpy.multin handles errors -- can be disregarded
            strain_alleleCN = AP_randomized * est_CN[strain]
            alleleCN[strain] = strain_alleleCN

        sample_sheet = pd.DataFrame(data=sample_sheet)

        return alleleCN, sample_sheet, validation_variants

    def simulateTE_population(self, TE_name, phylog, copy_number, strains=85, divergence=5, extinction=0.5, newDir="null"):

        #wrapper function to bring together the functions needed to simulate a population of TEs and output/save files
        alleleCN, sample_sheet, full_length = self.sampleTEs_from_Phylo(fasta=phylog, copy_number=copy_number, strains=strains)

        simName = f"{TE_name}"

        # create the new directory for the simulated data

        if newDir == "null":
            simDir = os.path.join(self.PacBio_root, "Simulations", date.today().strftime("%m-%d-%Y"))
        else:
            simDir = os.path.join(self.PacBio_root, "Simulations", newDir)
        try:  # make new directory for simulated data
            os.mkdir(simDir)
        except FileExistsError:
            pass

        strain_names = np.asarray(sample_sheet["Strains"]).reshape(len(sample_sheet["Strains"]), 1)
        pop_sample_sheet = pd.DataFrame(data={0:["A" for a in range(strains)], 1:[f"A{i}" for i in range(strains)]})
        pop_sample_sheet.to_csv(os.path.join(simDir, simName+"_sample_sheet.csv"), index=False, header=False)
        var_csv = np.hstack((full_length, strain_names))
        var_df = pd.DataFrame(data=var_csv, columns=[x+1 for x in range(full_length.shape[1])]+["Strains"])
        np.save(os.path.join(simDir, simName+"_CN.npy"), alleleCN)
        np.save(os.path.join(simDir, simName + ".PacBio_variants.npy"), full_length)
        sample_sheet.to_csv(os.path.join(simDir, simName+".PacBio_samples.csv"), index=False)
        var_df.to_csv(os.path.join(simDir, simName + ".PacBio_variants.csv"), index=False)

class piRNA:

    def __init__(self, active_te_path='/Users/iskander/Documents/Barbash_lab/TE_diversity_data/ACTIVE_TES_internal.tsv', active_full_path='/Users/iskander/Documents/Barbash_lab/TE_diversity_data/ACTIVE_TES_full.tsv'):
        self.active_tes = pd.read_csv(active_te_path, header=None)
        self.haplo_path='/Users/iskander/Documents/Barbash_lab/TE_diversity_data/HAPLOTYPE_CLUSTERS/HAPLOTYPE_CALL_07-17-2020'
        self.haplo_stats_path = '/Users/iskander/Documents/Barbash_lab/TE_diversity_data/HAPLOTYPE_CLUSTERS/STATS/'
        self.active_fullLength = pd.read_csv(active_full_path, header=None)
        self.CN_path = '/Users/iskander/Documents/Barbash_lab/TE_diversity_data/allele_CN/GDL_RUN-11-15-19'
        self.color_map = ['#6F0000', '#009191', '#6DB6FF', 'orange', '#490092']
        self.NT_encode = {'A': 0, 'T': 1, 'C': 2, 'G': 3}
        self.div_path = 'TEST_DATA/Pi'
        self.pop_indexes = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
                       [15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33],
                       [34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52],
                       [53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70],
                       [71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84]]
        self.piRNA_path = "/Users/iskander/Documents/Barbash_lab/TE_diversity_data/piRNA"
        self.piRNA_strains = ['B10.pckl','B12.pckl', 'I06.pckl', 'I17.pckl','ISO-1.pckl','Misy.pckl','N10.pckl',
                              'N16.pckl', 'Paris.pckl', 'RAL313.pckl', 'RAL358.pckl', 'RAL362.pckl', 'RAL375.pckl',
                              'RAL379.pckl', 'RAL380.pckl', 'RAL391.pckl', 'RAL399.pckl', 'RAL427.pckl', 'RAL437.pckl',
                              'RAL555.pckl', 'RAL705.pckl', 'RAL707.pckl', 'RAL712.pckl', 'RAL714.pckl', 'RAL732.pckl',
                              'T05.pckl', 'T07.pckl', 'ZW155.pckl', 'ZW184.pckl']
        self.GDL = ['B10.pckl','B12.pckl', 'I06.pckl', 'I17.pckl','N10.pckl', 'N16.pckl', 'T05.pckl', 'T07.pckl', 'ZW155.pckl', 'ZW184.pckl']
        self.consensus_path = '/Users/iskander/Documents/Barbash_lab/TE_diversity_data/TE_consensus'
        self.pirna_plotDir = "/Users/iskander/Documents/Barbash_lab/TE_diversity_data/piRNA/plots"
        self.pingPong_dir = "/Users/iskander/Documents/Barbash_lab/TE_diversity_data/piRNA/pingpong_output/"
        self.bam_dir = "/Users/iskander/Documents/Barbash_lab/TE_diversity_data/piRNA/BAM_files/"
        self.pileup_dir = "/Users/iskander/Documents/Barbash_lab/TE_diversity_data/piRNA/read_pileups"
        self.pi_path = "/Users/iskander/Documents/Barbash_lab/TE_diversity_data/seq_diversity_numpys/RUN_1-27-20/"

    def analyzePingPongPro(self, TE, FDR=0.1):

        pingPong_sites = np.asarray([])
        library = []
        for d in os.listdir(self.pingPong_dir):
            pp_pro = pd.read_csv(os.path.join(self.pingPong_dir, d, "ping-pong_signatures.appended.tsv"), sep='\t')
            TE_sites = pp_pro[pp_pro["contig"] == TE]
            passed_sites = TE_sites[TE_sites["FDR"] <= FDR]
            pingPong_sites = np.concatenate( (pingPong_sites, passed_sites["position"].values))
            library = library + [d for i in range(passed_sites.shape[0])]

        pp_DF = {"position":pingPong_sites+1, "library":library}
        pp_DF = pd.DataFrame(pp_DF)


        return pp_DF

    def PingPong_SFS(self):
        """

        Generate SFS plots for ping pong sites, such that we can find out what is the frequency of sites of ping pong amplification
        across the samples. This is to see how unique each site of ping pong is to each sample.

        :return:
        """
        complete_DF = pd.DataFrame(data={"freq":[], "TE":[], "positions":[]})
        for TE in self.active_fullLength[0]:

            #aggregate all TE data into dataframe for analysis
            ping_pong_DF = self.analyzePingPongPro(TE)

            #get counts of non-unique sites in each TE
            conservation = Counter(ping_pong_DF["position"])
            positions, counts = zip(*list(conservation.items()))
            freqs = np.asarray(list(counts) ) / 29 #divide counts by total number of individuals to get frequency
            names = [TE for i in range(len(freqs))]
            temp_DF = pd.DataFrame(data={"freq":freqs, "TE":names, "positions":positions})
            complete_DF = pd.concat((complete_DF.reset_index(drop=True), temp_DF), ignore_index=True)


        return complete_DF

    def PingPongConservationDiversity(self, SFS):
        """

        Correlate the conservation of ping pong sites with sequene diversity. The expectation is that sites that are more
        conserved might have more sequence diversity if TEs adapt to ping pong. Or piRNAs might be biased towards producing
        conserved/low seq diversity ping pong sites.

        :param SFS:
        :return:
        """
        diversity = np.asarray([])
        for TE_group in SFS.groupby("TE"):
            pi = np.load(os.path.join(self.pi_path, TE_group[0]+"_pi.npy"))
            PingPong_sites = np.asarray(TE_group[1]["positions"].values, dtype=int)
            PingPong_conservation = TE_group[1]["freq"].values
            PingPong_diversity = pi[5][PingPong_sites]
            diversity = np.concatenate((diversity, PingPong_diversity))

        pi_df = pd.DataFrame(data={"pi":diversity})

        SFS = pd.concat((SFS, pi_df), axis=1, sort=False)

        return SFS

    def concat_PP_LTR(self):

        LTR = np.load("/Users/iskander/Documents/Barbash_lab/TE_diversity_data/LTR_I.npy", allow_pickle=True)

        for d in os.listdir(self.pingPong_dir): #create new ping pong pro file with concatenated terminal repeats
            pp_pro = pd.read_csv(os.path.join(self.pingPong_dir, d, "ping-pong_signatures.tsv"), sep='\t')
            for TE in range(LTR.shape[0]):  # iterate through LTR TEs and concatenate LTRs to internals
                with open(os.path.join(self.consensus_path, LTR[TE, 0] + ".fa"), "r") as consensus:
                    for record in SeqIO.parse(consensus, "fasta"):
                        consensus_seq = record.seq
                internal_sites = pp_pro[pp_pro["contig"] == LTR[TE, 0 ]].copy()
                ltr_sites = pp_pro[pp_pro["contig"] == LTR[TE, 1]].copy()
                ltr_sites["position"] = ltr_sites["position"] + len(consensus_seq)
                full_sites = pd.concat((internal_sites, ltr_sites), axis=0)
                full_sites["contig"] = LTR[TE, 0 ].replace("_I", "_FULL")
                pp_pro = pd.concat((pp_pro, full_sites), axis=0)


            pp_pro.to_csv(os.path.join(self.pingPong_dir, d, "ping-pong_signatures.appended.tsv"), sep='\t')

    def sizeFactor_normalize(self, extension=".stranded.pirna.npy", mat_size=29):
        """

        Use size factor normalization taken from https://hbctraining.github.io/DGE_workshop/lessons/02_DGE_count_normalization.html
        to normalize the read counts for the piRNA libraries. We want to do this so that we can have easily comparable
        read counts across samples.

        :return:
        """
        TE_list = []
        files = [f for f in os.listdir(self.pileup_dir) if f.endswith(f"{extension}")]
        read_counts = np.full(fill_value=np.nan, shape=(mat_size,len(files)))

        index = 0
        for TE in files:
            piRNA_reads = np.load(os.path.join(self.pileup_dir, TE))
            reads_per_sample = np.sum(piRNA_reads, axis=(1,2))
            TE_list.append(TE.replace(f"{extension}", ""))
            read_counts[:,index] = reads_per_sample
            index += 1

        read_counts[np.isnan(read_counts)] = 0 #sanitize matrix
        #filtered_reads = read_counts[:,np.sum(read_counts, axis=0) > 0]
        filtered_reads = read_counts[:,~np.any(read_counts == 0, axis = 0)] #remove any rows that contain 0 read counts in any cell
        geom_mean = sp.gmean(filtered_reads, axis=0)
        ratio = filtered_reads / geom_mean

        norm_factor = np.median(ratio, axis=1)
        filtered_TEs = np.asarray(TE_list)[~np.any(read_counts == 0, axis = 0)]

        for TE in filtered_TEs:
            unNormed = np.load(os.path.join(self.pileup_dir,TE+f"{extension}"))
            normalized_piRNA = (unNormed.T/norm_factor).T
            np.save(os.path.join(self.pileup_dir, f"{TE}.normalized{extension}"), normalized_piRNA)

    def get_piRNA_pileups(self, BAM, TE, matrix_size, minRL=21, maxRL=30, Q=30):

        """

        Get read pileups for piRNA BAM file for each repeat of interest.

        :param BAM:
        :return:
        """
        bam_file = os.path.join(self.bam_dir, BAM)

        samfile = pysam.AlignmentFile(bam_file, "rb")
        all_headers = [seq["SN"] for seq in samfile.header["SQ"]]

        stranded_NT = {'A': 0, 'T': 1, 'C': 2, 'G': 3, "a":4, "t":5, "c":6, "g":7}

        if TE in all_headers: #handle when reference sequence is missing from alignments
            # Generate matrix to store read pileups:
            pileup_matrix = np.full(fill_value=0, shape=(8, matrix_size + 1))#each position has 8 elements in the vector: A,T,C,G,a,t,c,g lower case letters represent antisense reads
            iterable = samfile.pileup(TE, min_base_quality=Q)

            for R in iterable:

                SNPs = np.asarray([S for S in R.get_query_sequences(mark_matches=False)])
                position = R.reference_pos + 1
                name = R.reference_name
                read_len = np.asarray([S.alignment.reference_length for S in R.pileups])

                #compute a mapping quality for the alignments using Mike's method of percent ident.
                MAPQ = []
                for S in R.pileups:
                    #convert alignment scores to ~ % identity by considering each -1 as a single mismatch -> not perfect but ok
                    mi = {si[0]: 1 - ((S.alignment.reference_length + si[1]/2)/S.alignment.reference_length) for si in S.alignment.get_tags() if si[0] == "AS" or si[0] == "XS"}

                    #when there are 2+ possible alignments:
                    if len(mi.keys()) == 2:
                        M = ( (mi["AS"] - mi["XS"])**2 ) / (1 - mi["XS"])
                        if M > 0.05:
                            MAPQ.append(True)
                        else:
                            MAPQ.append(False)

                    else: #when there is only one alignment
                        if mi["AS"] < 0.2: #alignment is less then 20% diverged
                            MAPQ.append(True)
                        else:
                            MAPQ.append(False)


                length_filter = (np.asarray(read_len >= minRL, int) + np.asarray(read_len <= maxRL, int)) == 2 #filter by read length such that we only use 21-30 NT long reads (avoid miRNAs/fragments)
                final_filter = (length_filter*1 + np.asarray(MAPQ, int) ) == 2
                passed_SNPs = SNPs[final_filter]

                try:
                    for snp in passed_SNPs:
                        if snp in stranded_NT.keys(): #handle indels
                            pileup_matrix[stranded_NT[snp], position] += 1
                except IndexError: #handle errors
                        sys.stdout.write(f"Error: {TE} reference sequence length does not match BAM reference lengths\n")
            samfile.close()
        else:
            # Generate matrix to store read pileups:
            sys.stdout.write(f"Error: {TE} not found in BAM\n")
            pileup_matrix = np.full(fill_value=np.nan, shape=(4, matrix_size + 1))

        return pileup_matrix

    def generatePileups(self, threads=1, samplesheet="/data/is372/TE_diversity_data/piRNA/BAM_files/piRNA.samplesheet.txt"):
        """

        Iterate through the list of all repeats from the consensus file and generate the read pileups for each library. Output
        pileups as a 20x4xN matrix

        :return:
        """
        BAM_list = pd.read_csv(samplesheet, header=None)[0].values
        fasta = os.path.join(self.consensus_path, "repbase_19.06_DM_plus_tandems_2018-8-6.fa")
        with open(fasta, "r") as consensusRefs:
            for record in SeqIO.parse(consensusRefs, "fasta"):
                name = record.name
                seqlen = len(record.seq)

                #use multithreading
                pileup_gen = partial(self.get_piRNA_pileups, TE=name, matrix_size=seqlen)
                pileupJobs = Pool(processes=threads)
                pileups = np.asarray(pileupJobs.map(pileup_gen, BAM_list))
                pileupJobs.close()
                #save output from file:
                np.save(os.path.join(self.pileup_dir, f"{name}.stranded.pirna.npy"), pileups)

    def piRNA_alleles(self, TE, error_rate = 0.001, alpha=0.05, minFreq = 0, minorAlleles=False, stranded="none"):

        """

        Create the allele proportion matrices from the piRNA numpy files

        :param TE:
        :return:
        """
        NTs = ["A", "T", "C", "G"]
        if stranded == "none":
            piRNA_reads = np.load(os.path.join(self.pileup_dir, TE+'.pirna.npy'))
        elif stranded == "antisense": #check only antisense reads
            piRNA_reads = np.load(os.path.join(self.pileup_dir, TE + '.stranded.pirna.npy'))
            piRNA_reads = piRNA_reads[:, 4:8]


        elif stranded == "sense":  # check only antisense reads
            piRNA_reads = np.load(os.path.join(self.pileup_dir, TE + '.stranded.pirna.npy'))
            piRNA_reads = piRNA_reads[:, 0:4]


        allele_CN = np.load(os.path.join(self.CN_path, TE + '_CN.npy'))
        consensus = np.sum(allele_CN, axis=0)
        Major = np.argmax(consensus, axis=0)

        #Generate allele frequency matrix:
        columnNames = []
        if minorAlleles:
            col_spacer = 3
        else:
            col_spacer = 4
        AF_matrix = np.full(fill_value=np.nan, shape=(piRNA_reads.shape[2]*col_spacer, piRNA_reads.shape[0]))

        #Iterate through all positions in the piRNA data and get the allele frequencies of the piRNA reads
        row = 0
        for NT in range(len(Major)):
            positPileup = piRNA_reads.T[NT,:]
            totalReads_per_strain = np.sum(positPileup, axis=0)
            #Correct for sequencing error in basecalling: Q value of base filtering was 30 (99.9% error rate) which gives
            #us a Poiss distribution where lambda = 0.001*Reads. We must ask what is the probability of seeing X reads at minor
            #allele under poiss distribution with error probability.

            if minorAlleles: #option for analyzing only the "minor" alleles
                positMajor = Major[NT]
                minorAllele_mask = np.asarray([True if p != positMajor else False for p in range(4)]).T
                minorAllele_reads = positPileup[minorAllele_mask]
                SNP_names = [nuc + "_" + str(NT) for nuc in np.asarray(NTs)[minorAllele_mask]]
            else:
                #take all reads for all positions
                minorAllele_reads = positPileup
                SNP_names = [nuc + "_" + str(NT) for nuc in NTs]

            lambda_param = error_rate*totalReads_per_strain
            p_error = sp.poisson.sf(mu=lambda_param, k=minorAllele_reads)

            #Bonferonni correction very stringent:
            FDR = alpha/ np.product(AF_matrix.shape) #all tests done

            minorAllele_reads[p_error > FDR] = 0
            #for C in range(col_spacer):
            #    FDR = multipletests(p_error[C], alpha=alpha, method="fdr_bh")
            #    minorAllele_reads[C][~FDR[0]] = 0


            columnNames = columnNames + SNP_names

            minorAlleleFreq = minorAllele_reads / totalReads_per_strain

            AF_matrix[row:row+col_spacer,:] = minorAlleleFreq #assign values into matrix
            row = row+col_spacer

        columnNames = np.asarray(columnNames)
        #sanitize data and remove empty values:
        AF_matrix[np.isnan(AF_matrix)] = 0
        AF_matrix[AF_matrix <= minFreq] = 0
        #Remove all columns where no counts of minor allele
        mask = np.sum(AF_matrix, axis=1) > 0
        AF_matrix = AF_matrix[mask,:]
        columnNames = columnNames[mask]
        columnNames = columnNames.reshape(columnNames.shape[0], 1)

        AF_matrix = np.hstack((columnNames, AF_matrix))

        return AF_matrix

    def piRNA_correlation(self, TE, cluster, saveplot=False, scale="linear", abline=True, pop=True):

        piRNA = np.load(os.path.join(self.pileup_dir, f"{TE}.normalized.stranded.pirna.npy"))
        haploTable = pd.read_csv(os.path.join(self.haplo_path, f"{TE}.haplotypeTable.tsv"), sep='\t')

        # get indices of GDL strains in piRNA data:
        GDL_pirna = [0, 1, 2, 3, 6, 7, 25, 26, 27, 28]
        GDL_ngs = [2, 4, 19, 23, 39, 44, 55, 56, 81, 83]

        antisense_reads = piRNA[:, 4:8]
        sense_reads = piRNA[:, 0:4]

        alleles = haploTable[haploTable["Cluster ID"] == cluster]["Alleles"].values[0]
        CN = haploTable[haploTable["Cluster ID"] == cluster].values[:, 16::].T[GDL_ngs].reshape(10, )

        NT, position = zip(*[S.split("_") for S in alleles.split(',')])
        position = [int(i) for i in position]
        encoded_NTs = [self.NT_encode[n] for n in NT]

        sense_matrix = np.full(fill_value=np.nan, shape=(10, len(NT)))
        # get the allele reads from the piRNA data
        p = 0
        for pos in sense_reads.T[position].T[GDL_pirna].T:
            sense_matrix[:, p] = pos[encoded_NTs[p]]
            p += 1


        antisense_matrix = np.full(fill_value=np.nan, shape=(10, len(NT)))
        # get the allele reads from the piRNA data
        p = 0
        for pos in antisense_reads.T[position].T[GDL_pirna].T:
            antisense_matrix[:, p] = pos[encoded_NTs[p]]
            p += 1


        sense_haplo = np.average(sense_matrix, axis=1)
        antisense_haplo = np.average(antisense_matrix, axis=1)

        piRNA_matrix = np.concatenate((sense_haplo, antisense_haplo))

        sense_r, sense_p = sp.spearmanr(CN, sense_haplo)
        antisense_r, antisense_p = sp.spearmanr(CN, antisense_haplo)

        stranded = ["+" for S in range(len(sense_haplo))] + ["-" for M in range(len(antisense_haplo))]
        piRNA_df = pd.DataFrame(data={"piRNA": piRNA_matrix, "CN": np.concatenate((CN, CN)),
                                      "Population": ["Beijing", "Beijing", "Ithaca", "Ithaca", "Netherlands", "Netherlands",
                                                     "Tasmania", "Tasmania", "Zimbabwe", "Zimbabwe", "Beijing", "Beijing", "Ithaca", "Ithaca", "Netherlands", "Netherlands",
                                                     "Tasmania", "Tasmania", "Zimbabwe", "Zimbabwe"], "Strand":stranded})


        seqAnal = subfamilyInference()
        # make plot
        with sns.axes_style("whitegrid"):
            fig = plt.figure(figsize=(10, 8))
            axs = fig.subplots(1)
            pal = ["#FF0D00","#01B2C2"]
            #regplot = sns.lmplot(data=piRNA_df, x="piRNA", y="CN", scatter=True, height=5, aspect=2, ci=None, hue="Stranded", line_kws={"linestyle": "--"},
                                 #scatter_kws={"s":75}, legend_out=False, palette=["#FF0D00","#01B2C2"], hue_order=["+", "-"])
            if pop == False:
                scatplot = sns.scatterplot(data=piRNA_df, y="piRNA", x="CN", hue="Strand", palette=pal, hue_order=["+", "-"], edgecolor="black", s=250, alpha=0.75, ax=axs)
            else:
                scatplot = sns.scatterplot(data=piRNA_df, y="piRNA", x="CN", hue="Population", palette=self.color_map,
                                           hue_order=["Beijing", "Ithaca", "Netherlands", "Tasmania", "Zimbabwe"], style="Strand", style_order=["+", "-"], markers=["P", "s"], edgecolor="black", s=150, ax=axs)
            plt.ylabel("piRNA read depth", size=20)
            plt.xlabel("Copy Number", size=20)
            #axs.set_aspect('equal', adjustable='box')
            if abline:
                j = 0
                for g in piRNA_df.groupby("Strand"):
                    reg = LinearRegression().fit(g[1]["CN"].values.reshape(-1, 1), g[1]["piRNA"].values.reshape(-1, 1))
                    inter = reg.intercept_[0]
                    corr = reg.coef_[0]
                    seqAnal.abline(slope=corr, intercept=inter, col=pal[j], axs=axs)
                    j += 1
            #plt.yscale(value=scale)
            plt.yscale(value=scale)
            h, l = scatplot.get_legend_handles_labels()
            if pop == True:
                leg1 = scatplot.legend(h[0:6], l[0:6], fontsize=15, markerscale=2, edgecolor="black", ncol=1, title_fontsize=15, loc="lower right") #(0.1, 0.615)
                leg2 = scatplot.legend(h[7:9], l[7:9], fontsize=15, markerscale=2, edgecolor="black", ncol=1, title_fontsize=15, title="Strand", loc=(0.765, 0.1))
                scatplot.add_artist(leg1)
            else:
                scatplot.legend(fontsize=15, markerscale=2, edgecolor='black', ncol=1, title_fontsize=15)

            plt.tight_layout()
            plt.xticks(size=15)
            plt.yticks(size=15)

            if saveplot:
                plt.savefig(os.path.join(self.piRNA_path, "plots", f"{TE}.{cluster}.piRNA_corr.png"), dpi=300, bbox_inches="tight")
                plt.close()
            else:
                plt.show()
                plt.close()
        return sense_r, sense_p, antisense_r, antisense_p

    def haplotype_piRNA(self, TE, stranded="antisense", cluster="none"):
        """

        Get the piRNA read coverage vs haplotype copy number of the GDL individuals.


        :param TE:
        :param stranded:
        :return:
        """
        piRNA = np.load(os.path.join(self.pileup_dir, f"{TE}.normalized.stranded.pirna.npy"))
        haploTable = pd.read_csv(os.path.join(self.haplo_path, f"{TE}.haplotypeTable.tsv"), sep='\t')

        #get indices of GDL strains in piRNA data:
        GDL_pirna = [0,1,2,3,6,7,25,26,27,28]
        GDL_ngs = [2,4,19,23,39,44,55,56,81,83]
        if stranded == "none":
            piRNA_reads = piRNA[:, 4:8] + piRNA[:, 0:4]
        elif stranded == "antisense":
            piRNA_reads = piRNA[:, 4:8]
        elif stranded == "sense":
            piRNA_reads = piRNA[:, 0:4]

        haplotypeGDL = haploTable.values[:, 16::].T[GDL_ngs]
        haplotype_piRNA = np.full(fill_value=np.nan, shape=haplotypeGDL.shape)

        if cluster == "none":
            i = 0
            for alleles in haploTable["Alleles"]:

                NT, position = zip(*[S.split("_") for S in alleles.split(',')])
                position = [int(i) for i in position]
                encoded_NTs = [self.NT_encode[n] for n in NT]
                piRNA_matrix = np.full(fill_value=np.nan, shape=(10, len(NT)))

                #get the allele reads from the piRNA data
                p = 0
                for pos in piRNA_reads.T[position].T[GDL_pirna].T:
                    piRNA_matrix[:,p] = pos[encoded_NTs[p]]
                    p += 1
                avg_coverage = np.average(piRNA_matrix, axis=1)
                haplotype_piRNA[:,i] = avg_coverage
                i += 1

            #average piRNA reads for each haplotype calculated we can compute the correlation coefficients
            corr = []
            p_val = []
            haplotypes = haploTable["Cluster ID"].values
            for hap in range(haplotypeGDL.shape[1]):
                try:
                    r,p = sp.spearmanr(haplotypeGDL[:,hap], haplotype_piRNA[:,hap])
                    corr.append(r)
                    p_val.append(p)
                except ValueError:
                    pass

            return corr,p_val, haplotypes

    def piRNA_dispersion(self, TE, disp_table):
        """

        Get the sense piRNA read depth for all variants that are dispersed/underdispersed/overdispersed
        This way we can check some assumptions about activity levels.
        :param TE:
        :return:
        """

        haploTable = pd.read_csv(os.path.join(self.haplo_path, f"{TE}.haplotypeTable.tsv"), sep='\t')

        alleles = [S.split(",") for S in haploTable["Alleles"]]

        #load piRNA:
        piRNA = np.load(os.path.join(self.pileup_dir, f"{TE}.normalized.stranded.pirna.npy"))
        sense_piRNA = piRNA[:, 0:4]

        haplo_piRNA = {"reads":np.asarray([]), "activity":np.asarray([])}
        xi = 0
        for clust in alleles:
            haplotypeRead_depth = np.average([sense_piRNA.T[int(SNP.split("_")[1])][self.NT_encode[SNP.split("_")[0]]] for SNP in clust], axis=0)
            haplo_piRNA["reads"] = np.concatenate((haplo_piRNA["reads"], haplotypeRead_depth))
            haplo_piRNA["activity"] = np.concatenate((haplo_piRNA["activity"],np.asarray([disp_table["Poisson fit"][xi] for i in range(29)])))

            xi += 1
        box_order = sorted(list(set(haplo_piRNA["activity"])))
        sns.boxplot(data=haplo_piRNA, x="activity", y="reads", order=box_order)
        plt.show()
        plt.close()

    def piRNA_haplotype_plot(self, TE, cluster, saveplot=False, plot=True, scale="linear"):
        """

        Create a table with the sense and antisense reads for a given haplotype across all of the piRNA samples that we have.
        This can be used to generate a scatterplot showing the depth of sense and antisense reads for each sample to get an idea
        of which strains are producing piRNAs against a particular haplotype.
        :param TE:
        :param cluster:
        :return:
        """
        piRNA = np.load(os.path.join(self.pileup_dir, f"{TE}.normalized.stranded.pirna.npy"))
        haploTable = pd.read_csv(os.path.join(self.haplo_path, f"{TE}.haplotypeTable.tsv"), sep='\t')


        antisense_reads = piRNA[:, 4:8]
        sense_reads = piRNA[:, 0:4]
        if cluster == "total": #get total piRNA data
            antisense_haplo = np.average(np.sum(antisense_reads, axis=1), axis=1)
            sense_haplo = np.average(np.sum(sense_reads, axis=1), axis=1)
        else: #get cluster piRNA data
            alleles = haploTable[haploTable["Cluster ID"] == cluster]["Alleles"].values[0]
            NT, position = zip(*[S.split("_") for S in alleles.split(',')])
            position = [int(i) for i in position]
            encoded_NTs = [self.NT_encode[n] for n in NT]

            sense_matrix = np.full(fill_value=np.nan, shape=(29, len(NT)))
            # get the allele reads from the piRNA data
            p = 0
            for pos in sense_reads.T[position]:
                #print(pos[:,encoded_NTs[p]].shape)
                sense_matrix[:, p] = pos[encoded_NTs[p]]
                p += 1


            antisense_matrix = np.full(fill_value=np.nan, shape=(29, len(NT)))
            # get the allele reads from the piRNA data
            p = 0
            for pos in antisense_reads.T[position]:

                antisense_matrix[:, p] = pos[encoded_NTs[p]]
                p += 1

            sense_haplo = np.average(sense_matrix, axis=1)
            antisense_haplo = np.average(antisense_matrix, axis=1)

        #####
        piRNA_df = pd.DataFrame(data={"+": sense_haplo, "-": antisense_haplo,
                                      "Population/strain": pd.read_csv("/Users/iskander/Documents/Barbash_lab/TE_diversity_data/piRNA/piRNA.samplesheet.csv", header=None)[1],
                                      "Haplotype":[cluster for t in range(29)]})
        if plot == True:
            with sns.axes_style("whitegrid"):
                fig = plt.figure(figsize=(8,8))

                cmap = self.color_map + ["#ED0086", "#B400E0","#E5FD00", "#A6F900"]
                sns.scatterplot(data=piRNA_df, x="+", y="-", hue="Population/strain", s=100, edgecolor="black", palette=cmap, hue_order=["Beijing", "Ithaca", "Netherlands","Tasmania", "Zimbabwe","Raleigh", "ISO-1", "Misy", "Paris"])

                plt.yscale(value=scale)
                plt.xscale(value=scale)
                plt.xticks(fontsize=20)
                plt.yticks(fontsize=20)
                plt.xlabel("Sense piRNA read depth", fontsize=22.5)
                plt.ylabel("Antisense piRNA read depth", fontsize=22.5)
                plt.legend(fontsize=15, markerscale=1.5, edgecolor="black", ncol=2, title_fontsize=15)

                if saveplot:
                    plt.savefig(os.path.join(self.piRNA_path, "plots", f"{TE}.{cluster}.piRNA_readDepth.png"), dpi=300)
                    plt.close()
                else:
                    plt.show()
                    plt.close()
        else:
            return piRNA_df

    def piRNA_seq(self, TE, cluster):

        """

        Visualize the TE sequence with haplotypes annotated and piRNA coverage over the element.

        :param TE:
        :param cluster:
        :return:
        """

        piRNA = np.load(os.path.join(self.pileup_dir, f"{TE}.normalized.stranded.pirna.npy"))

        #sum of reads across whole element
        antisense_reads = np.sum(piRNA[:, 4:8],axis=1)
        sense_reads = np.sum(piRNA[:, 0:4], axis=1)

        #compute percent reads at each position in each strain to normalize for libary differences

        anti_perc = (antisense_reads.T / np.sum(antisense_reads, axis=1)).T
        sense_perc = (sense_reads.T / np.sum(sense_reads, axis=1)).T


        #extract the average read depth across samples for each position
        antisense_average = np.median(anti_perc, axis=0)*100
        sense_average = np.median(sense_perc, axis=0)*100
        pos = np.concatenate((np.arange(1, sense_average.shape[0]+1), np.arange(1, sense_average.shape[0]+1)))
        stranded = ["+" for p in range(len(sense_average))] + ["-" for m in range(len(sense_average))]
        piRNA_df = pd.DataFrame(data={"reads":np.concatenate((sense_average, -antisense_average)), "pos":pos, "Strand":stranded})

        #get haplotype allele positions:
        haploTable = pd.read_csv(os.path.join(self.haplo_path, f"{TE}.haplotypeTable.tsv"), sep='\t')
        cluster_positions = []
        cluster_labels = []

        for p in range(haploTable.shape[0]):
            SNPs = haploTable["Alleles"][p]
            c_pos = [int(S.split("_")[1]) for S in SNPs.split(",")]
            cluster_labels = cluster_labels + [haploTable["Cluster ID"][p] for h in range(len(c_pos))]
            cluster_positions = cluster_positions + c_pos

        allele_DF = pd.DataFrame(data={"Haplotype":cluster_labels, "pos":cluster_positions})
        with sns.axes_style("whitegrid"):
            fig = plt.figure(figsize=(12,5))
            sns.lineplot(data=piRNA_df, y="reads", x="pos", hue="Strand")

            i = 0
            for c in cluster:
                posit = allele_DF[allele_DF["Haplotype"] == c]["pos"]
                sns.rugplot(a=posit, color=self.color_map[i])

                i += 1
            read_ticks = ["{0:.2f}".format(abs(y)) for y in plt.yticks()[0]]
            plt.yticks(ticks=plt.yticks()[0], labels=read_ticks)

            plt.show()
            plt.close()

    def piRNA_AF(self, TE, cluster):

        piRNA = np.load(os.path.join(self.pileup_dir, f"{TE}.normalized.stranded.pirna.npy"))
        haploTable = pd.read_csv(os.path.join(self.haplo_path, f"{TE}.haplotypeTable.tsv"), sep='\t')

        # get indices of GDL strains in piRNA data:
        GDL_pirna = [0, 1, 2, 3, 6, 7, 25, 26, 27, 28]
        GDL_ngs = [2, 4, 19, 23, 39, 44, 55, 56, 81, 83]

        antisense_reads = piRNA[:, 4:8]
        sense_reads = piRNA[:, 0:4]

        alleles = haploTable[haploTable["Cluster ID"] == cluster]["Alleles"].values[0]
        CN = haploTable[haploTable["Cluster ID"] == cluster].values[:, 16::].T[GDL_ngs].reshape(10, )


        NT, position = zip(*[S.split("_") for S in alleles.split(',')])
        position = [int(i) for i in position]
        encoded_NTs = [self.NT_encode[n] for n in NT]

        #Get the AF of the haplotypes:
        CN_matrix = np.load(os.path.join(self.CN_path, f"{TE}_CN.npy"))
        totalHaplo_CN = np.mean(np.sum(CN_matrix[GDL_ngs].T[position], axis=1), axis=0 )

        haplo_AF = np.divide(CN, totalHaplo_CN, out=np.zeros_like(CN), where=totalHaplo_CN!=0)

        sense_matrix = np.full(fill_value=np.nan, shape=(10, len(NT)))
        # get the allele reads from the piRNA data
        p = 0
        for pos in sense_reads.T[position].T[GDL_pirna].T:
            sense_matrix[:, p] = pos[encoded_NTs[p]]
            p += 1

        antisense_matrix = np.full(fill_value=np.nan, shape=(10, len(NT)))
        # get the allele reads from the piRNA data
        p = 0
        for pos in antisense_reads.T[position].T[GDL_pirna].T:
            antisense_matrix[:, p] = pos[encoded_NTs[p]]
            p += 1

        sense_haplo = np.average(sense_matrix, axis=1)
        antisense_haplo = np.average(antisense_matrix, axis=1)

        TE_designations = [TE for i in range(10)]
        cluster_desig = [cluster for j in range(10)]

        piRNA_df = pd.DataFrame(data={"+":sense_haplo, "-":antisense_haplo, "AF":haplo_AF, "CN":CN, "Cluster":cluster_desig, "TE":TE_designations, "Population": ["Beijing", "Beijing", "Ithaca", "Ithaca", "Netherlands", "Netherlands",
                                                     "Tasmania", "Tasmania", "Zimbabwe", "Zimbabwe"]})

        return piRNA_df

    def aggregate_piRNA_AF_data(self):

        total_piRNA_data = pd.DataFrame(data={"+":[], "-":[], "AF":[], "CN":[], "Cluster":[], "TE":[], "Population":[]})
        #iterate through piRNA_AF to aggregate all data
        for TE in self.active_fullLength[0]:
            hapTable = pd.read_csv(os.path.join(self.haplo_path, f"{TE}.haplotypeTable.tsv"), sep='\t')
            for hap in hapTable["Cluster ID"]:
                piRNA_df = self.piRNA_AF(TE, hap)
                total_piRNA_data = pd.concat((total_piRNA_data, piRNA_df ))

        return total_piRNA_data

    def piRNA_diversity(self, TE, position=True):
        """

        Calculate the sequence diversity of piRNA reads and compare that to the diversity of the genomic TE sequences.

        :param TE:
        :return:
        """

        piRNA = np.load(os.path.join(self.pileup_dir, f"{TE}.normalized.stranded.pirna.npy"))
        sum_stats = summary_stats()
        # get indices of GDL strains in piRNA data:
        GDL_pirna = [0, 1, 2, 3, 6, 7, 25, 26, 27, 28]
        GDL_ngs = [2, 4, 19, 23, 39, 44, 55, 56, 81, 83]

        #CN = np.sum(np.load(os.path.join(self.CN_path, f"{TE}_CN.npy")), axis=1)


        antisense_reads = piRNA[:, 4:8]
        sense_reads = piRNA[:, 0:4]
        sense_pi = []
        antisense_pi = []
        for p in range(29):
            sense_pi.append(sum_stats.pop_Pi(sense_reads[p], indiv_strains=True, min_CN=1))
            antisense_pi.append(sum_stats.pop_Pi(antisense_reads[p], indiv_strains=True, min_CN=1))

        sense_pi = np.asarray(sense_pi)
        antisense_pi = np.asarray(antisense_pi)
        CN_pi = sum_stats.calcPi(TE, indiv_strains=True, nan=False, save=False)


        #compute the average of the per nucleotide ratios of sequence diversity & linear model for each strain
        antisense_ratios = []
        sense_ratios = []
        sense_r2 = []
        antisense_r2 = []
        nuisance_s_r2 = []
        nuisance_a_r2 = []

        for strain in range(10):
            piRNA_strain = GDL_pirna[strain]
            GDL_strain = GDL_ngs[strain]

            sense_positions = (CN_pi[GDL_strain,:] > 0) #get all positions that have polymorphism
            antisense_positions = (CN_pi[GDL_strain,:] > 0)

            #ratios of diversity
            if position:
                avg_sense_ratios = np.nanmean( np.divide(sense_pi[piRNA_strain][sense_positions],  CN_pi[GDL_strain,:][sense_positions]) )
                avg_antisese_ratios = np.nanmean( np.divide(antisense_pi[piRNA_strain][antisense_positions], CN_pi[GDL_strain,:][antisense_positions]))
            else:


                avg_sense_ratios = np.nanmean(sense_pi[piRNA_strain]) / np.nanmean(CN_pi[GDL_strain, :])
                avg_antisese_ratios = np.nanmean(antisense_pi[piRNA_strain][antisense_positions]) / np.nanmean(CN_pi[GDL_strain, :][antisense_positions])

            antisense_ratios.append(avg_antisese_ratios)
            sense_ratios.append(avg_sense_ratios)

            #compute linear model to fit data; can get r^2 to ask how well does genomic diversity explain piRNA diversity
            antisense_x = np.vstack((CN_pi[GDL_strain,:][antisense_positions], np.sum(antisense_reads[piRNA_strain], axis=0)[antisense_positions])).T
            sense_x = np.vstack((CN_pi[GDL_strain, :][sense_positions], np.sum(sense_reads[piRNA_strain], axis=0)[sense_positions])).T
            #secondary X variable is read depth

            #antisense_model = sm.api.OLS(antisense_pi[piRNA_strain][antisense_positions], sm.api.add_constant(antisense_x) ).fit()
            #print(antisense_model.summary())
            if len(sense_x[:,0]) == 0 or len(antisense_x[:,0]) == 0: #for empty data frames

                nuisance_r2_sense = np.nan
                nuisance_r2_antisense = np.nan
                r2_sense = np.nan
                r2_antisense = np.nan
                sense_corr = np.nan
                antisense_corr = np.nan
            else:

                antisense_corr = sp.pearsonr(antisense_pi[piRNA_strain][antisense_positions], CN_pi[GDL_strain, :][antisense_positions])
                sense_corr = sp.pearsonr(antisense_pi[piRNA_strain][sense_positions], CN_pi[GDL_strain, :][sense_positions])

                antisense_model = LinearRegression().fit(y=antisense_pi[piRNA_strain][antisense_positions], X=antisense_x[:, 0].reshape(-1, 1))
                sense_model = LinearRegression().fit(y=sense_pi[piRNA_strain][sense_positions], X=sense_x[:,0].reshape(-1, 1))

                r2_antisense = antisense_model.score(y=antisense_pi[piRNA_strain][antisense_positions], X=antisense_x[:,0].reshape(-1, 1))
                r2_sense = sense_model.score(y=sense_pi[piRNA_strain][sense_positions], X=sense_x[:,0].reshape(-1, 1))

                #compute or linear model with read depth as the X variable to see how well read depth explains piRNA diversity
                nuisance_antisense_model = LinearRegression().fit(y=antisense_pi[piRNA_strain][antisense_positions], X=antisense_x[:, 1].reshape(-1, 1))
                nuisance_sense_model = LinearRegression().fit(y=sense_pi[piRNA_strain][sense_positions], X=sense_x[:,1].reshape(-1, 1))
                nuisance_r2_antisense = nuisance_antisense_model.score(y=antisense_pi[piRNA_strain][antisense_positions], X=antisense_x[:,1].reshape(-1, 1))
                nuisance_r2_sense = nuisance_sense_model.score(y=sense_pi[piRNA_strain][sense_positions], X=sense_x[:,1].reshape(-1, 1))

            nuisance_s_r2.append(nuisance_r2_sense)
            nuisance_a_r2.append(nuisance_r2_antisense)
            sense_r2.append(r2_sense)
            antisense_r2.append(r2_antisense)

        #over all TE data
        data = {"pi_ratio": antisense_ratios+sense_ratios, "r-squared_pi":antisense_r2+sense_r2, "r-squared_depth":nuisance_a_r2+nuisance_s_r2, "strand":["-" for m in range(10)]+["+" for p in range(10)], "TE": [TE for t in range(20)]}
        pi_data = pd.DataFrame(data=data)

        return pi_data

    def calculate_GDL_diversity(self, minReads=0.5):
        """

        Calculate the sequence diversity statistics pooling the 10 GDL piRNA and 10 GDL genomic data. This way we can match
        our dataset and bulk up our sample.

        :param TE:
        :return:
        """
        clean_class = {"HTT": "Telomeric", "LINE": "LINE-like", "LTR": "LTR retrotransposon", "TIR": "DNA transposon"}

        seqDiversity = {"piRNA_pi":[], "strand":[], "genomic_pi":[], "TE":[], "class":[]}
        sum_stats = summary_stats()
        hapAnalysis = haplotypeAnalysis()
        GDL_pirna = [0, 1, 2, 3, 6, 7, 25, 26, 27, 28]
        GDL_ngs = [2, 4, 19, 23, 39, 44, 55, 56, 81, 83]
        for TE, superclass in self.active_fullLength.values:

            piRNA = np.load(os.path.join(self.pileup_dir, f"{TE}.normalized.stranded.pirna.npy"))


            antisense_reads = piRNA[:, 4:8][GDL_pirna]
            sense_reads = piRNA[:, 0:4][GDL_pirna]

            #in order to account for some sequencing error I will remove SNPs from the piRNA data that appear in less than 0.5 copies in the genomic data.
            CN = np.load(os.path.join(hapAnalysis.CN_path, f"{TE}_CN.npy"))

            SNP_filter = CN[GDL_ngs] < minReads

            #print(np.sum(sense_reads[GDL_pirna][SNP_filter]))
            sense_reads[SNP_filter] = 0
            antisense_reads[SNP_filter] = 0

            #print(sense_reads[GDL_pirna][SNP_filter])

            sense_pi = np.average(sum_stats.pop_Pi(sense_reads, indiv_strains=False, min_CN=0))

            antisense_pi = np.average(sum_stats.pop_Pi(antisense_reads, indiv_strains=False, min_CN=0))



            CN_pi = np.average(sum_stats.pop_Pi(CN[GDL_ngs], indiv_strains=False, min_CN=minReads))

            seqDiversity["piRNA_pi"] = seqDiversity["piRNA_pi"]+ [np.average(sense_pi), np.average(antisense_pi)]
            seqDiversity["strand"] = seqDiversity["strand"] + ["+", "-"]
            seqDiversity["TE"] = seqDiversity["TE"] + [hapAnalysis.TE_nameTable[TE], hapAnalysis.TE_nameTable[TE]]
            seqDiversity["genomic_pi"] = seqDiversity["genomic_pi"] + [CN_pi, CN_pi]
            seqDiversity["class"] = seqDiversity["class"] + [clean_class[superclass], clean_class[superclass]]

        seqDiversity = pd.DataFrame(data=seqDiversity)

        return seqDiversity

    def diversityData(self, p=True):

        data = {"pi_ratio": [], "r-squared_pi": [],
                "r-squared_depth": [],
                "strand": [], "TE": []}
        pi_data = pd.DataFrame(data=data)

        for TE in self.active_fullLength[0]:

            DF = self.piRNA_diversity(TE, p)
            pi_data = pd.concat((pi_data, DF), axis=0)

        clean_class = {"HTT": "Telomeric", "LINE": "LINE-like", "LTR": "LTR retrotransposon", "TIR": "DNA transposon"}
        class_arr = np.full(fill_value="", shape=pi_data.shape[0], dtype="object")
        for ind, TE in self.active_fullLength.iterrows():
            class_arr[pi_data["TE"] == TE[0]] = clean_class[TE[1]]

        pi_data["class"] = class_arr

        return pi_data

    def diversityPlot(self, pi_df, saveplot=False):
        hapAnalysis = haplotypeAnalysis()

        clean_TEs = [hapAnalysis.TE_nameTable[n] for n in pi_df["TE"]]
        pi_df["TE"] = clean_TEs
        TE_arr = []

        for group in self.active_fullLength.groupby(1):
            names = [hapAnalysis.TE_nameTable[n] for n in sorted(list(set(group[1][0])))]
            TE_arr = TE_arr + names
        #for g in pi_df.groupby("TE"):
        #    avg_ratio = np.nanmean(g[1]["pi_ratio"])
        #    TE_arr.append([g[0], avg_ratio])
        #box_order = list(zip(*sorted(TE_arr, key=operator.itemgetter(1))))[0]

        with sns.axes_style("whitegrid"):
            fig = plt.figure(figsize=(18, 6))
            sns.boxplot(data=pi_df, x="TE", y="pi_ratio", hue="strand", hue_order=["+", "-"], order=TE_arr, palette=["#FF0D00","#01B2C2"], dodge=True)
            plt.xticks(rotation=90, fontsize=20)
            plt.yticks(fontsize=15)
            plt.axhline(1, linestyle="--", color="grey")
            plt.xlabel("")
            plt.ylabel("piRNA diversity ratio", fontsize=20)
            plt.legend(fontsize=15, edgecolor="black", title="Strand", title_fontsize=15)
            plt.tight_layout()
            if saveplot:
                plt.savefig(os.path.join(self.pirna_plotDir, "pi_diversity.png"), dpi=300)
                plt.close()
            else:
                plt.show()
                plt.close()


