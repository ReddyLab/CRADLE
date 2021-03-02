
import math
import sys
import numpy as np
import pyBigWig
import statsmodels.api as sm


def findStartGibbs(seq, seqLen):
	gibbs = 0
	subtract = -1

	for i in range(seqLen-1):
		dimer = seq[i:(i+2)].upper()

		dimerIdx = 0
		for j in range(2):
			if(dimer[j]=='A'):
				dimerIdx = dimerIdx + np.power(4, 1-j) * 0
			elif(dimer[j]=='C'):
				dimerIdx = dimerIdx + np.power(4, 1-j) * 1
			elif(dimer[j]=='G'):
				dimerIdx = dimerIdx + np.power(4, 1-j) * 2
			elif(dimer[j]=='T'):
				dimerIdx = dimerIdx + np.power(4, 1-j) * 3
		gibbs = gibbs + GIBBS[dimerIdx][1]

		if(i==0):
			subtract = gibbs

	startGibbs = gibbs - subtract

	return startGibbs, gibbs


def editStartGibbs(oldDimer, newDimer, pastStartGibbs):
	gibbs = pastStartGibbs
	subtract = -1

	# newDimer
	dimerIdx = 0
	for j in range(2):
		if(newDimer[j]=='A'):
			dimerIdx = dimerIdx + np.power(4, 1-j) * 0
		elif(newDimer[j]=='C'):
			dimerIdx = dimerIdx + np.power(4, 1-j) * 1
		elif(newDimer[j]=='G'):
			dimerIdx = dimerIdx + np.power(4, 1-j) * 2
		elif(newDimer[j]=='T'):
			dimerIdx = dimerIdx + np.power(4, 1-j) * 3
	gibbs = gibbs + GIBBS[dimerIdx][1]

	## remove the old dimer for the next iteration
	dimerIdx = 0
	for j in range(2):
		if(oldDimer[j]=='A'):
			dimerIdx = dimerIdx + np.power(4, 1-j) * 0
		elif(oldDimer[j]=='C'):
			dimerIdx = dimerIdx + np.power(4, 1-j) * 1
		elif(oldDimer[j]=='G'):
			dimerIdx = dimerIdx + np.power(4, 1-j) * 2
		elif(oldDimer[j]=='T'):
			dimerIdx = dimerIdx + np.power(4, 1-j) * 3
	subtract = GIBBS[dimerIdx][1]

	startGibbs = gibbs - subtract

	return startGibbs, gibbs


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
analysisStart = 7981103
analysisEnd = 7981203
fragLen = 50
binSize = 1

########## necessary files
## sequence file
fa = 'TTTTCATGTGGTTACTGACCATTCATATCTTCTTTTGTGAGGAAAGAGTCTATTCAAATGTTTTGCCAATTGTTTATTTGGGCTGTTTGCCTTCTTATTATTGAGTTGTAAGAGTTATTTATATATTCTGGATACAAGCCTTTGTTAAATATACATATAAAAATATTTTCTGGTGGGGCACAGTGGCTCACGCCTATAATCC'

## gibbs energy
global GIBBS
GIBBS = [['AA', -1.04], ['AC', -2.04], ['AG', -1.29], ['AT', -1.27], ['CA', -0.78], ['CC', -1.97], ['CG', -1.44], ['CT', -1.29], ['GA', -1.66], ['GC', -2.7], ['GG', -1.97], ['GT', -2.04], ['TA', -0.12], ['TC', -1.66], ['TG', -0.78], ['TT', -1.04]]


########## calculate covariates
binStart = int((analysisStart + analysisStart + binSize) / float(2))
binEnd = int((analysisEnd - binSize + analysisEnd) / float(2))

fragStart = binStart + 1 - fragLen
fragEnd = binEnd + fragLen  # not included
shearStart = fragStart - 2
shearEnd = fragEnd + 2 # not included

## GENERATE A RESULT MATRIX
result = np.ones((fragEnd-fragStart, 3), dtype=np.float64)


## INDEX IN 'fa'
startIdx = 2
endIdx = (fragEnd - fragLen) - shearStart + 1

pastStartGibbs = -1

for idx in range(startIdx, endIdx):
	idx_fa = fa[idx:(idx+fragLen)]
	if(pastStartGibbs == -1):
		startGibbs, gibbs = findStartGibbs(idx_fa, fragLen)
	else:
		oldDimer = idx_fa[0:2].upper()
		newDimer = idx_fa[(fragLen-2):fragLen].upper()
		startGibbs, gibbs = editStartGibbs(oldDimer, newDimer, pastStartGibbs)

	idx_anneal, idx_denature = convertGibbs(gibbs)

	for pos in range(0, fragLen):
		result[idx+pos-2, 1] = result[idx+pos-2, 1] + idx_anneal
		result[idx+pos-2, 2] = result[idx+pos-2, 2] + idx_denature

## TRUNCATE REGIONS OUTSIDE OF THE ANALYSIS REGION
analysisStartIdx = analysisStart - fragStart
analysisEndIdx = analysisEnd - fragStart
result = result[analysisStartIdx:analysisEndIdx]


## perform regression
bwName = "test.bw"
bw = pyBigWig.open(bwName)
Y = bw.values(chromo, analysisStart, analysisEnd)
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
