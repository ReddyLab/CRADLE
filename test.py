#!/data/reddylab/software/miniconda2/envs/YoungSook/bin/python2.7

import pyBigWig
import numpy as np
import statsmodels.api as sm
import sys
import math


def findStartGibbs(seq, seqLen):
        gibbs = 0
        subtract = -1

        for i in range(seqLen-1):
                dimer = seq[i:(i+2)].upper()
                
                dimer_idx = 0
                for j in range(2):
			if(dimer[j]=='A'):
				dimer_idx = dimer_idx + np.power(4, 1-j) * 0
			elif(dimer[j]=='C'):
				dimer_idx = dimer_idx + np.power(4, 1-j) * 1
			elif(dimer[j]=='G'):
				dimer_idx = dimer_idx + np.power(4, 1-j) * 2
			elif(dimer[j]=='T'):
				dimer_idx = dimer_idx + np.power(4, 1-j) * 3
		gibbs = gibbs + GIBBS[dimer_idx][1]

                if(i==0):
                        subtract = gibbs

        start_gibbs = gibbs - subtract

        return start_gibbs, gibbs


def editStartGibbs(oldDimer, newDimer, past_start_gibbs):
        gibbs = past_start_gibbs
        subtract = -1

        # newDimer
	dimer_idx = 0
	for j in range(2):
		if(newDimer[j]=='A'):
			dimer_idx = dimer_idx + np.power(4, 1-j) * 0
		elif(newDimer[j]=='C'):
			dimer_idx = dimer_idx + np.power(4, 1-j) * 1
		elif(newDimer[j]=='G'):
			dimer_idx = dimer_idx + np.power(4, 1-j) * 2
		elif(newDimer[j]=='T'):
			dimer_idx = dimer_idx + np.power(4, 1-j) * 3
	gibbs = gibbs + GIBBS[dimer_idx][1]

        ## remove the old dimer for the next iteration
	dimer_idx = 0
	for j in range(2):
		if(oldDimer[j]=='A'):
			dimer_idx = dimer_idx + np.power(4, 1-j) * 0
		elif(oldDimer[j]=='C'):
			dimer_idx = dimer_idx + np.power(4, 1-j) * 1
		elif(oldDimer[j]=='G'):
			dimer_idx = dimer_idx + np.power(4, 1-j) * 2
		elif(oldDimer[j]=='T'):
			dimer_idx = dimer_idx + np.power(4, 1-j) * 3
	subtract = GIBBS[dimer_idx][1]

	start_gibbs = gibbs - subtract

        return start_gibbs, gibbs


def convertGibbs(gibbs):
	ENTROPY = -0.02485
	MIN_TM = -0.12 / ENTROPY
	MAX_TM = -2.7 / ENTROPY
	PARA1 = (math.pow(10, 6) - math.exp(1)) / (math.pow(10, 6) - 1)
	PARA2 =  math.pow(10, -6) / (1-PARA1)
	
        tm = gibbs / (ENTROPY*(fragLen-1))
        tm = (tm - MIN_TM) / (MAX_TM - MIN_TM)

        ## anneal
        anneal = ( math.exp(tm) - PARA1 ) * PARA2
        anneal = np.log(anneal)

        ## denature
        tm = tm - 1
        denature = ( math.exp( tm*(-1) ) - PARA1 ) * PARA2
        denature = math.log(denature)

        return anneal, denature



########## set varialbes
global fragLen
chromo = "chr17"
analysis_start = 7981103
analysis_end = 7981203
fragLen = 50
binSize = 1

########## necessary files
## sequence file
fa = 'TTTTCATGTGGTTACTGACCATTCATATCTTCTTTTGTGAGGAAAGAGTCTATTCAAATGTTTTGCCAATTGTTTATTTGGGCTGTTTGCCTTCTTATTATTGAGTTGTAAGAGTTATTTATATATTCTGGATACAAGCCTTTGTTAAATATACATATAAAAATATTTTCTGGTGGGGCACAGTGGCTCACGCCTATAATCC'

## gibbs energy
global GIBBS
GIBBS = [['AA', -1.04], ['AC', -2.04], ['AG', -1.29], ['AT', -1.27], ['CA', -0.78], ['CC', -1.97], ['CG', -1.44], ['CT', -1.29], ['GA', -1.66], ['GC', -2.7], ['GG', -1.97], ['GT', -2.04], ['TA', -0.12], ['TC', -1.66], ['TG', -0.78], ['TT', -1.04]]


########## calculate covariates
binStart = int((analysis_start + analysis_start + binSize) / float(2))
binEnd = int((analysis_end - binSize + analysis_end) / float(2))

frag_start = binStart + 1 - fragLen
frag_end = binEnd + fragLen  # not included
shear_start = frag_start - 2
shear_end = frag_end + 2 # not included

## GENERATE A RESULT MATRIX 
result = np.ones((frag_end-frag_start, 3), dtype=np.float64)


## INDEX IN 'fa'
start_idx = 2
end_idx = (frag_end - fragLen) - shear_start + 1

past_start_gibbs = -1

for idx in range(start_idx, end_idx):
	idx_fa = fa[idx:(idx+fragLen)]
	if(past_start_gibbs == -1):		
		start_gibbs, gibbs = findStartGibbs(idx_fa, fragLen)
	else:
		oldDimer = idx_fa[0:2].upper()
		newDimer = idx_fa[(fragLen-2):fragLen].upper()
		start_gibbs, gibbs = editStartGibbs(oldDimer, newDimer, past_start_gibbs)

	idx_anneal, idx_denature = convertGibbs(gibbs)
	
	for pos in range(0, fragLen):
		result[idx+pos-2, 1] = result[idx+pos-2, 1] + idx_anneal
		result[idx+pos-2, 2] = result[idx+pos-2, 2] + idx_denature	

## TRUNCATE REGIONS OUTSIDE OF THE ANALYSIS REGION
analysis_start_idx = analysis_start - frag_start
analysis_end_idx = analysis_end - frag_start
result = result[analysis_start_idx:analysis_end_idx]


## perform regression
bwName = "test.bw"
bw = pyBigWig.open(bwName)
Y = bw.values(chromo, analysis_start, analysis_end) 
bw.close()
model = sm.GLM(Y, result, family=sm.families.Poisson(link=sm.genmod.families.links.log)).fit()

coef = model.params
coef[0] = np.round(coef[0], 0)
coef[1] = np.round(coef[1], 2)
coef[2] = np.round(coef[2], 3)


if( (coef[0] == 5) and (coef[1] == -0.04) and (coef[2] == 0.001)):
	sys.exit(0) # without error
else:
	sys.exit(1)





