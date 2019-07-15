import sys
import os
import numpy as np
import math
import time
import tempfile
import gc
import scipy
import py2bit
import pyBigWig
import warnings
import statsmodels.api as sm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import random
import warnings
import h5py


from CRADLE.CorrectBias import vari


cpdef calculateContinuousFrag(chromo, analysis_start, analysis_end, binStart, binEnd, nBins, lastBin):

	warnings.filterwarnings('ignore', r'All-NaN slice encountered')
	warnings.filterwarnings('ignore', r'Mean of empty slice')


	###### CALCULATE INDEX VARIABLE
	frag_start = binStart + 1 - vari.FRAGLEN 
	frag_end = binEnd + vari.FRAGLEN  # not included
	shear_start = frag_start - 2
	shear_end = frag_end + 2 # not included

	faFile = py2bit.open(vari.FA)
	chromoEnd = int(faFile.chroms(chromo))

	if(shear_start < 1):
		shear_start = 1
		frag_start = 3
		binStart = max(binStart, frag_start)
		analysis_start = max(analysis_start, frag_start)
		
		###### EDIT BINSTART/ BINEND    
		nBins = int( (analysis_end-analysis_start) / float(vari.BINSIZE) )
		leftValue = (analysis_end - analysis_start) % int(vari.BINSIZE)
		if(leftValue != 0):
			nBins = nBins + 1
			lastBin = True
			binEnd = int( (analysis_start + (nBins-1) * vari.BINSIZE + analysis_end) / float(2) )
		else:
			lastBin = False
			binEnd = binStart + (nBins-1) * vari.BINSIZE

		frag_end = binEnd + vari.FRAGLEN
		shear_end = frag_end + 2	

	if(shear_end > chromoEnd):
		analysis_end_modified = min(analysis_end, chromoEnd - 2)  # not included

		if(analysis_end_modified == analysis_end):
			shear_end = chromoEnd
			frag_end = shear_end - 2
		else:
			analysis_end = analysis_end_modified

			###### EDIT BINSTART/ BINEND 	
			nBins = int( (analysis_end-analysis_start) / float(vari.BINSIZE) )
			leftValue = (analysis_end - analysis_start) % int(vari.BINSIZE)
			if(leftValue != 0):
				nBins = nBins + 1
				lastBin = True
				binEnd = int( (analysis_start + (nBins-1) * vari.BINSIZE + analysis_end) / float(2) )
			else:
				lastBin = False
				binEnd = binStart + (nBins-1) * vari.BINSIZE
			
			frag_end = binEnd + vari.FRAGLEN
			shear_end = frag_end + 2

			if(shear_end > chromoEnd):
				shear_end = chromoEnd
				frag_end = shear_end - 2
	
	###### GENERATE A RESULT MATRIX 
	result = makeMatrix_ContinuousFrag(binStart, binEnd, nBins)

	###### GET SEQUENCE
	fa = faFile.sequence(chromo, (shear_start-1), (shear_end-1))
	faFile.close()

	##### OPEN BIAS FILES	
	if(vari.MAP == 1):
		mapFile = pyBigWig.open(vari.MAPFILE)
		mapValue = np.array(mapFile.values(chromo, frag_start, frag_end))

		mapValue[np.where(mapValue == 0)] = np.nan
		mapValue = np.log(mapValue)
		mapValue[np.where(np.isnan(mapValue) == True)] = float(-6)

		mapValue_view = memoryView(mapValue)
		mapFile.close()
		del mapFile, mapValue

	if(vari.GQUAD == 1):
		gquadFile = [0] * len(vari.GQAUDFILE)
		gquadValue = [0] * len(vari.GQAUDFILE)

		for i in range(len(vari.GQAUDFILE)):
			gquadFile[i] = pyBigWig.open(vari.GQAUDFILE[i])
			gquadValue[i] = gquadFile[i].values(chromo, frag_start, frag_end)
			gquadFile[i].close()

		gquadValue = np.array(gquadValue)
		gquadValue = np.nanmax(gquadValue, axis=0)
		gquadValue[np.where(gquadValue == 0)] = np.nan
		gquadValue = np.log(gquadValue / float(vari.GQAUD_MAX))

		gquadValue[np.where(np.isnan(gquadValue) == True)] = float(-5)
		gquad_view = memoryView(gquadValue)

		del gquadFile, gquadValue


	##### INDEX IN 'fa'
	start_idx = 2  # index in the fasta file (Included in the range)
	end_idx = (frag_end - vari.FRAGLEN) - shear_start + 1   # index in the fasta file (Not included in the range)   

	##### INITIALIZE VARIABLES
	if(vari.SHEAR == 1):
		past_mer1 = -1
		past_mer2 = -1
	if(vari.PCR == 1):
		past_start_gibbs = -1

	result_startIdx = -1
	result_endIdx = -1

	##### STORE COVARI RESULTS
	covariFile_temp = tempfile.NamedTemporaryFile(suffix=".hdf5", dir=vari.OUTPUT_DIR, delete=True)
	covariFileName = covariFile_temp.name
	covariFile_temp.close()

	f = h5py.File(covariFileName, "w")
	covariFile = f.create_dataset("X", (nBins, vari.COVARI_NUM), dtype='f', compression="gzip")
	
	for idx in range(start_idx, end_idx):
		idx_covari = [0] * vari.COVARI_NUM
		idx_covari_ptr = 0

		if(vari.SHEAR == 1):
			###  mer1
			mer1 = fa[(idx-2):(idx+3)]
			if('N' in mer1):
				past_mer1 = -1
				idx_mgw = vari.N_MGW
				idx_prot = vari.N_PROT
			else:
				if(past_mer1 == -1): # there is no information on past_mer1
					past_mer1, idx_mgw, idx_prot = find5merProb(mer1)
				else:
					past_mer1, idx_mgw, idx_prot = edit5merProb(past_mer1, mer1[0], mer1[4])

			###  mer2
			fragEnd_idx = idx + vari.FRAGLEN
			mer2 = fa[(fragEnd_idx-3):(fragEnd_idx+2)]
			if('N' in mer2):
				past_mer2 = -1
				idx_mgw = idx_mgw + vari.N_MGW
				idx_prot = idx_prot + vari.N_PROT
			else:
				if(past_mer2 == -1):
					past_mer2, add1, add2 = findComple5merProb(mer2)
				else:
					past_mer2, add1, add2 = editComple5merProb(past_mer2, mer2[0], mer2[4])
				idx_mgw = idx_mgw + add1
				idx_prot = idx_prot + add2

			idx_covari[idx_covari_ptr] = idx_mgw
			idx_covari[idx_covari_ptr+1] = idx_prot
			idx_covari_ptr = idx_covari_ptr + 2

	
		if(vari.PCR == 1):
			idx_fa = fa[idx:(idx+vari.FRAGLEN)]
			if(past_start_gibbs == -1):
				start_gibbs, gibbs = findStartGibbs(idx_fa, vari.FRAGLEN)
			else:
				oldDimer = idx_fa[0:2].upper()
				newDimer = idx_fa[(vari.FRAGLEN-2):vari.FRAGLEN].upper()
				start_gibbs, gibbs = editStartGibbs(oldDimer, newDimer, past_start_gibbs)

			idx_anneal, idx_denature = convertGibbs(gibbs)

			idx_covari[idx_covari_ptr] = idx_anneal
			idx_covari[idx_covari_ptr+1] = idx_denature
			idx_covari_ptr = idx_covari_ptr + 2

		if(vari.MAP == 1):
			map1 = mapValue_view[(idx-2)]
			map2 = mapValue_view[(idx+vari.FRAGLEN-2-vari.KMER)]
			idx_map = map1 + map2

			idx_covari[idx_covari_ptr] = idx_map
			idx_covari_ptr = idx_covari_ptr + 1
	
		if(vari.GQUAD == 1):
			idx_gquad = np.nanmax(np.asarray(gquad_view[(idx-2):(idx+vari.FRAGLEN-2)]))

			idx_covari[idx_covari_ptr] = idx_gquad
			idx_covari_ptr = idx_covari_ptr + 1

		### DETERMINE WHICH ROWS TO EDIT IN RESULT MATRIX		
		thisFrag_start = idx + shear_start
		thisFrag_end = thisFrag_start + vari.FRAGLEN
		
		if(result_startIdx == -1):
			result_startIdx = 0
			result_endIdx = 1 # not included
			if(np.isnan(result[result_endIdx, 0]) == False):
				while( result[result_endIdx, 0] < thisFrag_end):
					result_endIdx = result_endIdx + 1
					if(result_endIdx > vari.FRAGLEN):
						result_endIdx = result_endIdx - (vari.FRAGLEN+1)
					if(np.isnan(result[result_endIdx, 0]) == True):
						break
			maxBinPos = binStart + vari.FRAGLEN
			numPoppedPos = 0
		else:

			while( result[result_startIdx, 0] < thisFrag_start ):
				## pop the element
				line = []
				for covari_pos in range(vari.COVARI_NUM):
					line.extend([ result[result_startIdx, (covari_pos+1)]  ])
					result[result_startIdx, (covari_pos+1)] = float(0)
				covariFile[numPoppedPos] = line

	
				numPoppedPos = numPoppedPos + 1
				if(maxBinPos >= binEnd):
					result[result_startIdx, 0] = np.nan
				else:
					result[result_startIdx, 0] = maxBinPos + 1
					maxBinPos = maxBinPos + 1

				result_startIdx = result_startIdx + 1
				if(result_startIdx > vari.FRAGLEN):
					result_startIdx = result_startIdx - (vari.FRAGLEN+1)

			if(np.isnan(result[result_endIdx, 0]) == False):
				while( result[result_endIdx, 0] < thisFrag_end):
					result_endIdx = result_endIdx + 1
					if(result_endIdx > vari.FRAGLEN):
						result_endIdx = result_endIdx - (vari.FRAGLEN+1)
					if(np.isnan(result[result_endIdx, 0]) == True):
						break

		if(result_endIdx < result_startIdx):
			for pos in range(result_startIdx, (vari.FRAGLEN+1)):
				for covari_pos in range(vari.COVARI_NUM):
					result[pos, covari_pos+1] = result[pos, covari_pos+1] + idx_covari[covari_pos]
			for pos in range(0, result_endIdx):
				for covari_pos in range(vari.COVARI_NUM):
					result[pos, covari_pos+1] = result[pos, covari_pos+1] + idx_covari[covari_pos]
		else:
			for pos in range(result_startIdx, result_endIdx):
				for covari_pos in range(vari.COVARI_NUM):
					result[pos, covari_pos+1] = result[pos, covari_pos+1] + idx_covari[covari_pos]
	
		if(idx == (end_idx-1)): # the last fragment
			### pop the rest of positions that are not np.nan
			if(result_endIdx < result_startIdx):
				for pos in range(result_startIdx, (vari.FRAGLEN+1)):
					line = []
					for covari_pos in range(vari.COVARI_NUM):
						line.extend([ result[pos, (covari_pos+1)]  ])
					covariFile[numPoppedPos] = line

					numPoppedPos = numPoppedPos + 1
					
				for pos in range(0, result_endIdx):
					line = []
					for covari_pos in range(vari.COVARI_NUM):
						line.extend([ result[pos, (covari_pos+1)]  ])
					covariFile[numPoppedPos] = line

					numPoppedPos = numPoppedPos + 1
			else:
				for pos in range(result_startIdx, result_endIdx):
					line = []
					for covari_pos in range(vari.COVARI_NUM):
						line.extend([ result[pos, (covari_pos+1)]  ])
					covariFile[numPoppedPos] = line

					numPoppedPos = numPoppedPos + 1

	f.close()	
	

	return correctReadCount(covariFileName, chromo, analysis_start, analysis_end, lastBin, nBins)




cpdef calculateDiscreteFrag(chromo, analysis_start, analysis_end, binStart, binEnd, nBins, lastBin):
	warnings.filterwarnings('ignore', r'All-NaN slice encountered')
	warnings.filterwarnings('ignore', r'Mean of empty slice')  


	###### CALCULATE INDEX VARIABLE
	frag_start = binStart + 1 - vari.FRAGLEN
	frag_end = binEnd + vari.FRAGLEN  # not included
	shear_start = frag_start - 2
	shear_end = frag_end + 2 # not included

	faFile = py2bit.open(vari.FA)
	chromoEnd = int(faFile.chroms(chromo))

	if(shear_start < 1):
		shear_start = 1
		frag_start = 3
		binStart = max(binStart, frag_start)
		analysis_start = max(analysis_start, frag_start)

		###### EDIT BINSTART/ BINEND    
		nBins = int( (analysis_end-analysis_start) / float(vari.BINSIZE) )
		leftValue = (analysis_end - analysis_start) % int(vari.BINSIZE)
		if(leftValue != 0):
			nBins = nBins + 1
			lastBin = True
			binEnd = int( (analysis_start + (nBins-1) * vari.BINSIZE + analysis_end) / float(2) )
		else:
			lastBin = False
			binEnd = binStart + (nBins-1) * vari.BINSIZE

		frag_end = binEnd + vari.FRAGLEN
		shear_end = frag_end + 2

	if(shear_end > chromoEnd):
		analysis_end_modified = min(analysis_end, chromoEnd - 2)  # not included

		if(analysis_end_modified == analysis_end):
			shear_end = chromoEnd
			frag_end = shear_end - 2
		else:
			analysis_end = analysis_end_modified

			###### EDIT BINSTART/ BINEND    
			nBins = int( (analysis_end-analysis_start) / float(vari.BINSIZE) )
			leftValue = (analysis_end - analysis_start) % int(vari.BINSIZE)
			if(leftValue != 0):
				nBins = nBins + 1
				lastBin = True
				binEnd = int( (analysis_start + (nBins-1) * vari.BINSIZE + analysis_end) / float(2) )
			else:
				lastBin = False
				binEnd = binStart + (nBins-1) * vari.BINSIZE

			frag_end = binEnd + vari.FRAGLEN
			shear_end = frag_end + 2

			if(shear_end > chromoEnd):
				shear_end = chromoEnd
				frag_end = shear_end - 2


	###### GENERATE A RESULT MATRIX 
	result = makeMatrix_DiscreteFrag(binStart, binEnd, nBins)

	###### GET SEQUENCE
	fa = faFile.sequence(chromo, (shear_start-1), (shear_end-1))
	faFile.close()

        ##### OPEN BIAS FILES   
	if(vari.MAP == 1):
		mapFile = pyBigWig.open(vari.MAPFILE)
		mapValue = np.array(mapFile.values(chromo, frag_start, frag_end))

		mapValue[np.where(mapValue == 0)] = np.nan
		mapValue = np.log(mapValue)
		mapValue[np.where(np.isnan(mapValue) == True)] = float(-6)

		mapValue_view = memoryView(mapValue)
		mapFile.close()
		del mapFile, mapValue

	if(vari.GQUAD == 1):
		gquadFile = [0] * len(vari.GQAUDFILE)
		gquadValue = [0] * len(vari.GQAUDFILE)

		for i in range(len(vari.GQAUDFILE)):
			gquadFile[i] = pyBigWig.open(vari.GQAUDFILE[i])
			gquadValue[i] = gquadFile[i].values(chromo, frag_start, frag_end)
			gquadFile[i].close()

		gquadValue = np.array(gquadValue)
		gquadValue = np.nanmax(gquadValue, axis=0)
		gquadValue[np.where(gquadValue == 0)] = np.nan
		gquadValue = np.log(gquadValue / float(vari.GQAUD_MAX))

		gquadValue[np.where(np.isnan(gquadValue) == True)] = float(-5)
		gquad_view = memoryView(gquadValue)

		del gquadFile, gquadValue	


	##### STORE COVARI RESULTS
	covariFile_temp = tempfile.NamedTemporaryFile(suffix=".hdf5", dir=vari.OUTPUT_DIR, delete=True)
	covariFileName = covariFile_temp.name
	covariFile_temp.close()

	f = h5py.File(covariFileName, "w")
	covariFile = f.create_dataset("X", (nBins, vari.COVARI_NUM), dtype='f', compression="gzip")

	resultIdx = 0
	while(resultIdx < nBins): # for each bin
		if(resultIdx == (nBins-1)):
			pos = binEnd
		else:
			pos = binStart + resultIdx * vari.BINSIZE

		thisBin_firstFragStart = pos + 1 - vari.FRAGLEN
		thisBin_lastFragStart = pos 

		if(thisBin_firstFragStart < 3):
			thisBin_firstFragStart = 3
		if( (thisBin_lastFragStart + vari.FRAGLEN) > (chromoEnd - 2)):
			thisBin_lastFragStart = chromoEnd - 2 - vari.FRAGLEN

		thisBin_numFrag = thisBin_lastFragStart - thisBin_firstFragStart + 1

		thisBin_firstFragStartIdx = thisBin_firstFragStart - shear_start

		##### INITIALIZE VARIABLES
		if(vari.SHEAR == 1):
			past_mer1 = -1
			past_mer2 = -1
		if(vari.PCR == 1):
			past_start_gibbs = -1

		line = [0.0] * vari.COVARI_NUM		
		for binFragIdx in range(thisBin_numFrag):
			idx = thisBin_firstFragStartIdx + binFragIdx

			idx_covari = [0] * vari.COVARI_NUM
			idx_covari_ptr = 0

			if(vari.SHEAR == 1):
				###  mer1
				mer1 = fa[(idx-2):(idx+3)]
				if('N' in mer1):
					past_mer1 = -1
					idx_mgw = vari.N_MGW
					idx_prot = vari.N_PROT
				else:
					if(past_mer1 == -1): # there is no information on past_mer1
						past_mer1, idx_mgw, idx_prot = find5merProb(mer1)
					else:
						past_mer1, idx_mgw, idx_prot = edit5merProb(past_mer1, mer1[0], mer1[4])

                        	###  mer2
				fragEnd_idx = idx + vari.FRAGLEN
				mer2 = fa[(fragEnd_idx-3):(fragEnd_idx+2)]
				if('N' in mer2):
					past_mer2 = -1
					idx_mgw = idx_mgw + vari.N_MGW
					idx_prot = idx_prot + vari.N_PROT
				else:
					if(past_mer2 == -1):
						past_mer2, add1, add2 = findComple5merProb(mer2)
					else:
						past_mer2, add1, add2 = editComple5merProb(past_mer2, mer2[0], mer2[4])
					idx_mgw = idx_mgw + add1
					idx_prot = idx_prot + add2

				idx_covari[idx_covari_ptr] = idx_mgw
				idx_covari[idx_covari_ptr+1] = idx_prot
				idx_covari_ptr = idx_covari_ptr + 2

			if(vari.PCR == 1):
				idx_fa = fa[idx:(idx+vari.FRAGLEN)]
				if(past_start_gibbs == -1):
					start_gibbs, gibbs = findStartGibbs(idx_fa, vari.FRAGLEN)
				else:
					oldDimer = idx_fa[0:2].upper()
					newDimer = idx_fa[(vari.FRAGLEN-2):vari.FRAGLEN].upper()
					start_gibbs, gibbs = editStartGibbs(oldDimer, newDimer, past_start_gibbs)

				idx_anneal, idx_denature = convertGibbs(gibbs)

				idx_covari[idx_covari_ptr] = idx_anneal
				idx_covari[idx_covari_ptr+1] = idx_denature
				idx_covari_ptr = idx_covari_ptr + 2

			if(vari.MAP == 1):
				map1 = mapValue_view[(idx-2)]
				map2 = mapValue_view[(idx+vari.FRAGLEN-2-vari.KMER)]
				idx_map = map1 + map2

				idx_covari[idx_covari_ptr] = idx_map
				idx_covari_ptr = idx_covari_ptr + 1

			if(vari.GQUAD == 1):
				idx_gquad = np.nanmax(np.asarray(gquad_view[(idx-2):(idx+vari.FRAGLEN-2)]))

				idx_covari[idx_covari_ptr] = idx_gquad
				idx_covari_ptr = idx_covari_ptr + 1
	

			for covari_pos in range(vari.COVARI_NUM):
				line[covari_pos] = line[covari_pos] + idx_covari[covari_pos]

		covariFile[resultIdx] = line		

		resultIdx = resultIdx + 1 

		if(resultIdx == nBins):
			break

	f.close()

	return correctReadCount(covariFileName, chromo, analysis_start, analysis_end, lastBin, nBins)




cpdef calculateTrainCovariates(args):
	### supress numpy nan-error message
	warnings.filterwarnings('ignore', r'All-NaN slice encountered')
	warnings.filterwarnings('ignore', r'Mean of empty slice')
	
	chromo = args[0]
	analysis_start = int(args[1])  # Genomic coordinates(starts from 1)
	analysis_end = int(args[2])  # not included

	binStart = int((analysis_start + analysis_start + vari.BINSIZE) / float(2))
	binEnd = int((analysis_end - vari.BINSIZE + analysis_end) / float(2))

	###### CALCULATE INDEX VARIABLE
	frag_start = binStart + 1 - vari.FRAGLEN
	frag_end = binEnd + vari.FRAGLEN  # not included
	shear_start = frag_start - 2
	shear_end = frag_end + 2 # not included

	faFile = py2bit.open(vari.FA)
	chromoEnd = int(faFile.chroms(chromo))

	if(shear_start < 1):
		shear_start = 1
		frag_start = 3
		binStart = max(binStart, frag_start)
		analysis_start = max(analysis_start, frag_start)

	if(shear_end > chromoEnd):
		shear_end = chromoEnd
		frag_end = shear_end - 2
		binEnd = min(binEnd, (frag_end-1))
		analysis_end = min(analysis_end, frag_end)  # not included

	###### GENERATE A RESULT MATRIX 
	nBins = binEnd - binStart + 1 
	result = makeMatrix_ContinuousFrag_train(binStart, binEnd, nBins)	

	###### GET SEQUENCE
	fa = faFile.sequence(chromo, (shear_start-1), (shear_end-1))
	faFile.close()

        ##### OPEN BIAS FILES   
	if(vari.MAP == 1):
		mapFile = pyBigWig.open(vari.MAPFILE)
		mapValue = np.array(mapFile.values(chromo, frag_start, frag_end))

		mapValue[np.where(mapValue == 0)] = np.nan
		mapValue = np.log(mapValue)
		mapValue[np.where(np.isnan(mapValue) == True)] = float(-6)

		mapValue_view = memoryView(mapValue)
		mapFile.close()
		del mapFile, mapValue

	if(vari.GQUAD == 1):
		gquadFile = [0] * len(vari.GQAUDFILE)
		gquadValue = [0] * len(vari.GQAUDFILE)

		for i in range(len(vari.GQAUDFILE)):
			gquadFile[i] = pyBigWig.open(vari.GQAUDFILE[i])
			gquadValue[i] = gquadFile[i].values(chromo, frag_start, frag_end)
			gquadFile[i].close()

		gquadValue = np.array(gquadValue)
		gquadValue = np.nanmax(gquadValue, axis=0)
		gquadValue[np.where(gquadValue == 0)] = np.nan
		gquadValue = np.log(gquadValue / float(vari.GQAUD_MAX))

		gquadValue[np.where(np.isnan(gquadValue) == True)] = float(-5)
		gquad_view = memoryView(gquadValue)

		del gquadFile, gquadValue

		
	##### INDEX IN 'fa'
	start_idx = 2  # index in the fasta file (Included in the range)
	end_idx = (frag_end - vari.FRAGLEN) - shear_start + 1   # index in the fasta file (Not included in the range)   

        ##### INITIALIZE VARIABLES
	if(vari.SHEAR == 1):
		past_mer1 = -1
		past_mer2 = -1
	if(vari.PCR == 1):
		past_start_gibbs = -1

	result_startIdx = -1
	result_endIdx = -1

	##### STORE COVARI RESULTS
	covariFile_temp = tempfile.NamedTemporaryFile(suffix=".hdf5", dir=vari.OUTPUT_DIR, delete=True)
	covariFileName = covariFile_temp.name
	covariFile_temp.close()

	f = h5py.File(covariFileName, "w")
	covariFile = f.create_dataset("X", (nBins, vari.COVARI_NUM), dtype='f', compression="gzip")

	for idx in range(start_idx, end_idx):
		idx_covari = [0] * vari.COVARI_NUM
		idx_covari_ptr = 0

		if(vari.SHEAR == 1):
			###  mer1
			mer1 = fa[(idx-2):(idx+3)]
			if('N' in mer1):
				past_mer1 = -1
				idx_mgw = vari.N_MGW
				idx_prot = vari.N_PROT
			else:
				if(past_mer1 == -1): # there is no information on past_mer1
					past_mer1, idx_mgw, idx_prot = find5merProb(mer1)
				else:
					past_mer1, idx_mgw, idx_prot = edit5merProb(past_mer1, mer1[0], mer1[4])

			##  mer2
			fragEnd_idx = idx + vari.FRAGLEN
			mer2 = fa[(fragEnd_idx-3):(fragEnd_idx+2)]
			if('N' in mer2):
				past_mer2 = -1
				idx_mgw = idx_mgw + vari.N_MGW
				idx_prot = idx_prot + vari.N_PROT
			else:
				if(past_mer2 == -1):
					past_mer2, add1, add2 = findComple5merProb(mer2)
				else:
					past_mer2, add1, add2 = editComple5merProb(past_mer2, mer2[0], mer2[4])
				idx_mgw = idx_mgw + add1
				idx_prot = idx_prot + add2

			idx_covari[idx_covari_ptr] = idx_mgw
			idx_covari[idx_covari_ptr+1] = idx_prot
			idx_covari_ptr = idx_covari_ptr + 2


		if(vari.PCR == 1):
			idx_fa = fa[idx:(idx+vari.FRAGLEN)]
			if(past_start_gibbs == -1):
				start_gibbs, gibbs = findStartGibbs(idx_fa, vari.FRAGLEN)
			else:
				oldDimer = idx_fa[0:2].upper()
				newDimer = idx_fa[(vari.FRAGLEN-2):vari.FRAGLEN].upper()
				start_gibbs, gibbs = editStartGibbs(oldDimer, newDimer, past_start_gibbs)

			idx_anneal, idx_denature = convertGibbs(gibbs)

			idx_covari[idx_covari_ptr] = idx_anneal
			idx_covari[idx_covari_ptr+1] = idx_denature
			idx_covari_ptr = idx_covari_ptr + 2

		if(vari.MAP == 1):
			map1 = mapValue_view[(idx-2)]
			map2 = mapValue_view[(idx+vari.FRAGLEN-2-vari.KMER)]
			idx_map = map1 + map2

			idx_covari[idx_covari_ptr] = idx_map
			idx_covari_ptr = idx_covari_ptr + 1

		if(vari.GQUAD == 1):
			idx_gquad = np.nanmax(np.asarray(gquad_view[(idx-2):(idx+vari.FRAGLEN-2)]))

			idx_covari[idx_covari_ptr] = idx_gquad
			idx_covari_ptr = idx_covari_ptr + 1


		### DETERMINE WHICH ROWS TO EDIT IN RESULT MATRIX               
		thisFrag_start = idx + shear_start
		thisFrag_end = thisFrag_start + vari.FRAGLEN

		if(result_startIdx == -1):
			result_startIdx = 0
			result_endIdx = 1 # not include
			if(np.isnan(result[result_endIdx, 0]) == False):
				while( result[result_endIdx, 0] < thisFrag_end):
					result_endIdx = result_endIdx + 1
					if(result_endIdx > vari.FRAGLEN):
						result_endIdx = result_endIdx - (vari.FRAGLEN+1)
					if(np.isnan(result[result_endIdx, 0]) == True):
						break
			maxBinPos = binStart + vari.FRAGLEN 
			numPoppedPos = 0
		else:
			while( result[result_startIdx, 0] < thisFrag_start ):
				## pop the element
				line = []
				for covari_pos in range(vari.COVARI_NUM):
					line.extend([ result[result_startIdx, (covari_pos+1)]  ])
					result[result_startIdx, (covari_pos+1)] = float(0)
				covariFile[numPoppedPos] = line

				numPoppedPos = numPoppedPos + 1
				if(maxBinPos >= binEnd):
					result[result_startIdx, 0] = np.nan
				else:
					result[result_startIdx, 0] = maxBinPos + 1
					maxBinPos = maxBinPos + 1					
								
				result_startIdx = result_startIdx + 1
				if(result_startIdx > vari.FRAGLEN):
					result_startIdx = result_startIdx - (vari.FRAGLEN+1)
	
	
			if(np.isnan(result[result_endIdx, 0]) == False):
				while( result[result_endIdx, 0] < thisFrag_end):
					result_endIdx = result_endIdx + 1
					if(result_endIdx > vari.FRAGLEN):
						result_endIdx = result_endIdx - (vari.FRAGLEN+1)
					if(np.isnan(result[result_endIdx, 0]) == True):
						break	


		if(result_endIdx < result_startIdx):
			for pos in range(result_startIdx, (vari.FRAGLEN+1)):
				for covari_pos in range(vari.COVARI_NUM):
					result[pos, covari_pos+1] = result[pos, covari_pos+1] + idx_covari[covari_pos]
			for pos in range(0, result_endIdx):
				for covari_pos in range(vari.COVARI_NUM):
					result[pos, covari_pos+1] = result[pos, covari_pos+1] + idx_covari[covari_pos]
		else:
			for pos in range(result_startIdx, result_endIdx):
				for covari_pos in range(vari.COVARI_NUM):
					result[pos, covari_pos+1] = result[pos, covari_pos+1] + idx_covari[covari_pos]

		if(idx == (end_idx-1)): # the last fragment
			### pop the rest of positions that are not np.nan
			if(result_endIdx < result_startIdx):
				for pos in range(result_startIdx, (vari.FRAGLEN+1)):
					line = []
					for covari_pos in range(vari.COVARI_NUM):
						line.extend([ result[pos, (covari_pos+1)]  ])
					covariFile[numPoppedPos] = line

					numPoppedPos = numPoppedPos + 1

				for pos in range(0, result_endIdx):
					line = []
					for covari_pos in range(vari.COVARI_NUM):
						line.extend([ result[pos, (covari_pos+1)]  ])
					covariFile[numPoppedPos] = line

					numPoppedPos = numPoppedPos + 1
			else:
				for pos in range(result_startIdx, result_endIdx):
					line = []	
					for covari_pos in range(vari.COVARI_NUM):
						line.extend([ result[pos, (covari_pos+1)]  ])
					covariFile[numPoppedPos] = line

					numPoppedPos = numPoppedPos + 1		
		
	f.close()
	
	return_Line = [covariFileName, numPoppedPos]

	
	### output read counts
	for rep in range(vari.SAMPLE_NUM):	
		rcFile_temp = tempfile.NamedTemporaryFile(suffix=".hdf5", dir=vari.OUTPUT_DIR, delete=True)
		rcFileName = rcFile_temp.name		
		rcFile_temp.close()

		f = h5py.File(rcFileName, "w")
		rcFile = f.create_dataset("Y", (nBins, ), dtype='f', compression="gzip")

		
		return_Line.extend([ rcFileName ])

		if(rep < vari.CTRLBW_NUM):
			bw = pyBigWig.open(vari.CTRLBW_NAMES[rep])
		else:
			bw = pyBigWig.open(vari.EXPBW_NAMES[rep-vari.CTRLBW_NUM])

		temp = np.array(bw.values(chromo, analysis_start, analysis_end))		
		numPos = len(temp)
	

		for binIdx in range(nBins):
			if(rep < vari.CTRLBW_NUM):
				if( (binIdx + vari.BINSIZE) >= numPos):
					rc = np.nanmean(temp[binIdx:]) / float(vari.CTRLSCALER[rep])
				else:
					rc = np.nanmean(temp[binIdx:(binIdx + vari.BINSIZE)]) / float(vari.CTRLSCALER[rep])
			else:
				if( (binIdx + vari.BINSIZE) >= numPos):
					rc = np.nanmean(temp[binIdx:]) / float(vari.EXPSCALER[rep-vari.CTRLBW_NUM])
				else:
					rc = np.nanmean(temp[binIdx:(binIdx + vari.BINSIZE)]) / float(vari.EXPSCALER[rep-vari.CTRLBW_NUM])
		
			if(np.isnan(rc) == True):
				rc = float(0)

			rcFile[binIdx] = rc


		f.close()
		bw.close()		

	return return_Line
	


cpdef calculateTaskCovariates(args):
	### supress numpy nan-error message
	warnings.filterwarnings('ignore', r'All-NaN slice encountered')
	warnings.filterwarnings('ignore', r'Mean of empty slice')

	chromo = args[0]
	analysis_start = int(args[1])  # Genomic coordinates(starts from 1)
	analysis_end = int(args[2])

	#### DECIDE IF 'calculateContinuousFrag' or 'calculateDiscreteFrag'
	if( (analysis_start + vari.BINSIZE) >= analysis_end):
		firstBinPos = int((analysis_start + analysis_end) / float(2))
		lastBinPos = firstBinPos
		nBins = 1
		lastBin = False
		result = calculateContinuousFrag(chromo, analysis_start, analysis_end, firstBinPos, lastBinPos, nBins, lastBin)
		return result

	firstBinPos = int((2*analysis_start + vari.BINSIZE) / float(2))
	if( (analysis_start + 2*vari.BINSIZE) > analysis_end):
		secondBinPos = int((analysis_start + vari.BINSIZE + analysis_end) / float(2))
		lastBinPos = secondBinPos
		nBins = 2
		lastBin = True	
	else:
		secondBinPos = int((2*analysis_start + 3 * vari.BINSIZE) / float(2)) 		
		leftValue = (analysis_end - analysis_start) % int(vari.BINSIZE)
		nBins = int((analysis_end - analysis_start) / float(vari.BINSIZE))
		if(leftValue != 0):
			nBins = nBins + 1
			lastBin = True
			lastBinPos = int( (analysis_start + (nBins-1) * vari.BINSIZE + analysis_end) / float(2) )  ## should be included in the analysis
		else:
			lastBin = False
			lastBinPos = firstBinPos + (nBins-1) * vari.BINSIZE ## should be included in the analysis

	if( (secondBinPos-firstBinPos) <= vari.FRAGLEN):
		result = calculateContinuousFrag(chromo, analysis_start, analysis_end, firstBinPos, lastBinPos, nBins, lastBin)
		return result
	else:
		result = calculateDiscreteFrag(chromo, analysis_start, analysis_end, firstBinPos, lastBinPos, nBins, lastBin)
		return result



cpdef performRegression(covariFiles):
	### Read covariates values (X)
	X_numRows = 0
	for i in range(len(covariFiles)):
		X_numRows = X_numRows + int(covariFiles[i][1])
	X_numCols = vari.COVARI_NUM + 1

	cdef double [:,:] X_view = np.ones((X_numRows, X_numCols), dtype=np.float64)

	cdef int row_ptr = 0
	cdef int rowIdx 
	cdef int col_ptr

	for fileIdx in range(len(covariFiles)):
		subfileName = covariFiles[fileIdx][0]
		f = h5py.File(subfileName, "r")
		
		rowIdx = 0
		while(rowIdx < f['X'].shape[0]):
			temp = f['X'][rowIdx]
			col_ptr = 0
			while(col_ptr < vari.COVARI_NUM):
				X_view[rowIdx + row_ptr, col_ptr+1] = float(temp[col_ptr])
				col_ptr = col_ptr + 1
			rowIdx = rowIdx + 1
		row_ptr = row_ptr + int(covariFiles[fileIdx][1])

		f.close()
		os.remove(subfileName)


	### COEFFICIENTS
	COEFCTRL = np.zeros((vari.CTRLBW_NUM, (vari.COVARI_NUM+1)), dtype=np.float64)
	COEFEXP = np.zeros((vari.EXPBW_NUM, (vari.COVARI_NUM+1)), dtype=np.float64)

	if(X_numRows < 50000):
		idx = np.array(list(range(X_numRows)))
	else:
		idx = np.random.choice(np.array(list(range(X_numRows))), 50000, replace=False)

	cdef double [:] Y_view = np.zeros(X_numRows, dtype=np.float64)

	cdef int ptr
	cdef int rcIdx

	for rep in range(vari.CTRLBW_NUM):
		ptr = 0

		for fileIdx in range(len(covariFiles)):
			subfileName = covariFiles[fileIdx][rep+2]
			f = h5py.File(subfileName, "r")

			rcIdx = 0
			while(rcIdx < f['Y'].shape[0]):
				Y_view[rcIdx+ptr] = float(f['Y'][rcIdx])
				rcIdx = rcIdx + 1

			ptr = ptr + int(f['Y'].shape[0])

			f.close()
			os.remove(subfileName)


		deleteIdx = np.where( (np.array(Y_view) < np.finfo(np.float32).min) | (np.array(Y_view) > np.finfo(np.float32).max))[0]
		if(len(deleteIdx) != 0):
			model = sm.GLM(np.delete(np.array(Y_view).astype(int), deleteIdx), np.delete(np.array(X_view), deleteIdx, axis=0), family=sm.families.Poisson(link=sm.genmod.families.links.log)).fit()
		else:
			model = sm.GLM(np.array(Y_view).astype(int), np.array(X_view), family=sm.families.Poisson(link=sm.genmod.families.links.log)).fit()		

		coef = model.params
		COEFCTRL[rep, ] = coef
		corr = np.corrcoef(model.fittedvalues, np.array(Y_view))[0, 1]
		corr = np.round(corr, 2)		

		## PLOT
		maxi1 = np.nanmax(model.fittedvalues[idx])
		maxi2 = np.nanmax(np.array(Y_view)[idx])
		maxi = max(maxi1, maxi2)

		figName = vari.OUTPUT_DIR + "/ctrl_rep" + str(rep+1) + ".png"
		plt.plot(np.array(Y_view)[idx], model.fittedvalues[idx], color='g', marker='s', alpha=0.01)
		plt.text((maxi-25), 10, corr, ha='center', va='center')
		plt.xlabel("observed")
		plt.ylabel("predicted")
		plt.xlim(0, maxi)
		plt.ylim(0, maxi)
		plt.plot([0, maxi], [0, maxi], 'k-', color='r')
		plt.gca().set_aspect('equal', adjustable='box')
		plt.savefig(figName)
		plt.close()
		plt.clf()

	
	for rep in range(vari.EXPBW_NUM):
		ptr = 0
		for fileIdx in range(len(covariFiles)):
			subfileName = covariFiles[fileIdx][rep+2+vari.CTRLBW_NUM]
			f = h5py.File(subfileName, "r")

			rcIdx = 0
			while(rcIdx < f['Y'].shape[0]):
				Y_view[rcIdx+ptr] = float(f['Y'][rcIdx])
				rcIdx = rcIdx + 1

			ptr = ptr + int(f['Y'].shape[0])

			f.close()
			os.remove(subfileName)

		deleteIdx = np.where( (np.array(Y_view) < np.finfo(np.float32).min) | (np.array(Y_view) > np.finfo(np.float32).max))[0]
		if(len(deleteIdx) != 0):
			model = sm.GLM(np.delete(np.array(Y_view).astype(int), deleteIdx), np.delete(np.array(X_view), deleteIdx, axis=0), family=sm.families.Poisson(link=sm.genmod.families.links.log)).fit()
		else:
			model = sm.GLM(np.array(Y_view).astype(int), np.array(X_view), family=sm.families.Poisson(link=sm.genmod.families.links.log)).fit()

		coef = model.params
		COEFEXP[rep, ] = coef
		corr = np.corrcoef(model.fittedvalues, np.array(Y_view))[0, 1]
		corr = np.round(corr, 2)

		## PLOT
		maxi1 = np.nanmax(model.fittedvalues[idx])
		maxi2 = np.nanmax(np.array(Y_view)[idx])
		maxi = max(maxi1, maxi2)
	
		figName = vari.OUTPUT_DIR + "/exp_rep" + str(rep+1) + ".png"
		plt.plot(np.array(Y_view)[idx], model.fittedvalues[idx], color='g', marker='s', alpha=0.01)
		plt.text((maxi-25), 10, corr, ha='center', va='center')
		plt.xlabel("observed")
		plt.ylabel("predicted")
		plt.xlim(0, maxi)
		plt.ylim(0, maxi)
		plt.plot([0, maxi], [0, maxi], 'k-', color='r')
		plt.gca().set_aspect('equal', adjustable='box')
		plt.savefig(figName)
		plt.close()
		plt.clf()

	return COEFCTRL, COEFEXP


cpdef memoryView(value):
	cdef double [:] array_view
	array_view = value

	return array_view


cpdef makeMatrix_ContinuousFrag_train(binStart, binEnd, nBins):

	result = np.zeros(((vari.FRAGLEN+1), (vari.COVARI_NUM+1)), dtype=np.float64)
	for i in range(vari.FRAGLEN+1):
		pos = binStart + i

		if(pos > binEnd):
			result[i, 0] = np.nan
		else:
			result[i, 0] = pos

	return result


cpdef makeMatrix_ContinuousFrag(binStart, binEnd, nBins):

	result = np.zeros(((vari.FRAGLEN+1), (vari.COVARI_NUM+1)), dtype=np.float64)
	for i in range(vari.FRAGLEN+1):
		pos = binStart + i * vari.BINSIZE

		if(pos > binEnd):
			result[i, 0] = np.nan
		else:
			result[i, 0] = pos

	if(nBins == (vari.FRAGLEN+1)):
		result[vari.FRAGLEN, 0] = binEnd

	return result


cpdef makeMatrix_DiscreteFrag(binStart, binEnd, nBins):
	cdef double [:,:] result_view = np.zeros((nBins, vari.COVARI_NUM), dtype=np.float64)

	return result_view


cpdef find5merProb(mer5):
	base_info = 0
	subtract = -1

	for i in range(5):
		if(mer5[i]=='A' or mer5[i]=='a'):
			base_info = base_info + np.power(4, 4-i) * 0
		elif(mer5[i]=='C' or mer5[i]=='c'):
			base_info = base_info + np.power(4, 4-i) * 1
		elif(mer5[i]=='G' or mer5[i]=='g'):
			base_info = base_info + np.power(4, 4-i) * 2
		elif(mer5[i]=='T' or mer5[i]=='t'):
			base_info = base_info + np.power(4, 4-i) * 3
		
		if(i==0):
			subtract = base_info

	mgw = vari.MGW[base_info][1]
	prot = vari.PROT[base_info][1]

	next_base_info = base_info - subtract

	return next_base_info, mgw, prot 


cpdef edit5merProb(past_mer, oldBase, newBase):
	base_info = past_mer
	base_info = base_info * 4
	subtract = -1

	## newBase
	if(newBase=='A' or newBase=='a'):
		base_info = base_info + 0
	elif(newBase=='C' or newBase=='c'):
		base_info = base_info + 1
	elif(newBase=='G' or newBase=='g'):
		base_info = base_info + 2
	elif(newBase=='T' or newBase=='t'):
		base_info = base_info + 3

	base_info = int(base_info)
	mgw = vari.MGW[base_info][1]
	prot = vari.PROT[base_info][1]

	## subtract oldBase
	if(oldBase=='A' or oldBase=='a'):
		subtract = np.power(4, 4) * 0
	elif(oldBase=='C' or oldBase=='c'):
		subtract = np.power(4, 4) * 1
	elif(oldBase=='G' or oldBase=='g'):
		subtract = np.power(4, 4) * 2
	elif(oldBase=='T' or oldBase=='t'):
		subtract = np.power(4, 4) * 3

	next_base_info = base_info - subtract

	return next_base_info, mgw, prot


cpdef findComple5merProb(mer5):
	base_info = 0
	subtract = -1

	for i in range(5):
		if(mer5[i]=='A' or mer5[i]=='a'):
			base_info = base_info + np.power(4, i) * 3
		elif(mer5[i]=='C' or mer5[i]=='c'):
			base_info = base_info + np.power(4, i) * 2
		elif(mer5[i]=='G' or mer5[i]=='g'):
			base_info = base_info + np.power(4, i) * 1
		elif(mer5[i]=='T' or mer5[i]=='t'):
			base_info = base_info + np.power(4, i) * 0

		if(i==0):
			subtract = base_info

	mgw = vari.MGW[base_info][1]
	prot = vari.PROT[base_info][1]

	next_base_info = base_info - subtract
	
	return next_base_info, mgw, prot


cpdef editComple5merProb(past_mer, oldBase, newBase):
	base_info = past_mer
	base_info = int(base_info / 4)
	subtract = -1

	# newBase
	if(newBase=='A' or newBase=='a'):
		base_info = base_info + np.power(4, 4) * 3
	elif(newBase=='C' or newBase=='c'):
		base_info = base_info + np.power(4, 4) * 2
	elif(newBase=='G' or newBase=='g'):
		base_info = base_info + np.power(4, 4) * 1
	elif(newBase=='T' or newBase=='t'):
		base_info = base_info + np.power(4, 4) * 0

	base_info = int(base_info)
	mgw = vari.MGW[base_info][1]
	prot = vari.PROT[base_info][1]

	## subtract oldBase
	if(oldBase=='A' or oldBase=='a'):
		subtract = 3
	elif(oldBase=='C' or oldBase=='c'):
		subtract = 2
	elif(oldBase=='G' or oldBase=='g'):
		subtract = 1
	elif(oldBase=='T' or oldBase=='t'):
		subtract = 0

	next_base_info = base_info - subtract

	return next_base_info, mgw, prot


cpdef findStartGibbs(seq, seqLen):
	gibbs = 0
	subtract = -1

	for i in range(seqLen-1):
		dimer = seq[i:(i+2)].upper()
		if( 'N' in dimer):
			gibbs = gibbs + vari.N_GIBBS
		else:
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
			gibbs = gibbs + vari.GIBBS[dimer_idx][1]		

		if(i==0):
			subtract = gibbs

	start_gibbs = gibbs - subtract

	return start_gibbs, gibbs


cpdef editStartGibbs(oldDimer, newDimer, past_start_gibbs):
	gibbs = past_start_gibbs
	subtract = -1

	# newDimer
	if( 'N' in newDimer):
		gibbs = gibbs + vari.N_GIBBS
	else:
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
		gibbs = gibbs + vari.GIBBS[dimer_idx][1]

	## remove the old dimer for the next iteration
	if( 'N' in oldDimer):
		subtract = vari.N_GIBBS
	else:
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
		subtract = vari.GIBBS[dimer_idx][1]

	start_gibbs = gibbs - subtract

	return start_gibbs, gibbs


cpdef convertGibbs(gibbs):
	tm = gibbs / (vari.ENTROPY*(vari.FRAGLEN-1))
	tm = (tm - vari.MIN_TM) / (vari.MAX_TM - vari.MIN_TM)

	## anneal
	anneal = ( math.exp(tm) - vari.PARA1 ) * vari.PARA2 
	anneal = np.log(anneal)

	## denature
	tm = tm - 1
	denature = ( math.exp( tm*(-1) ) - vari.PARA1 ) * vari.PARA2
	denature = math.log(denature) 	

	return anneal, denature


cpdef correctReadCount(covariFileName, chromo, analysis_start, analysis_end, lastBin, nBins):
	warnings.filterwarnings('ignore', r'All-NaN slice encountered')
	warnings.filterwarnings('ignore', r'Mean of empty slice')	

	region_start = analysis_start
	if(lastBin == False):
		region_end = analysis_start + vari.BINSIZE * nBins
		lastBin_start = None
		lastBin_end = None
	else: # lastBin == True
		region_end = analysis_start + vari.BINSIZE * (nBins-1)
		lastBin_start = region_end
		lastBin_end = analysis_end

	## OUTPUT FILES	
	subfinalCtrl = [0] * vari.CTRLBW_NUM	
	subfinalCtrlNames = [0] * vari.CTRLBW_NUM
	subfinalExp = [0] * vari.EXPBW_NUM
	subfinalExpNames = [0] * vari.EXPBW_NUM

	for i in range(vari.CTRLBW_NUM):
		subfinalCtrl[i] = tempfile.NamedTemporaryFile(mode="w+t", suffix=".bed", dir=vari.OUTPUT_DIR, delete=False)
		subfinalCtrlNames[i] = subfinalCtrl[i].name
		subfinalCtrl[i].close()
	
	for i in range(vari.EXPBW_NUM):
		subfinalExp[i] = tempfile.NamedTemporaryFile(mode="w+t", suffix=".bed", dir=vari.OUTPUT_DIR, delete=False)
		subfinalExpNames[i] = subfinalExp[i].name
		subfinalExp[i].close()

	###### GET POSITIONS WHERE THE NUMBER OF FRAGMENTS > FILTERVALUE
	selectedIdx, highRC_idx, starts = selectIdx(chromo, region_start, region_end, lastBin_start, lastBin_end, nBins)	

	if(len(selectedIdx) == 0):
		for i in range(vari.CTRLBW_NUM):
			os.remove(subfinalCtrlNames[i])
		for i in range(vari.EXPBW_NUM):
			os.remove(subfinalExpNames[i])	

		os.remove(covariFileName)

		return [ [None] * vari.CTRLBW_NUM, [None] * vari.EXPBW_NUM, chromo ]  


	f = h5py.File(covariFileName, "r")	

	for rep in range(vari.CTRLBW_NUM):
		## observed read counts
		bw = pyBigWig.open(vari.CTRLBW_NAMES[rep])
		if(lastBin == True):
			rcArr = np.array(bw.stats(chromo, region_start, region_end, type="mean", nBins=(nBins-1)))
			rcArr[np.where(rcArr==None)] = float(0)
			rcArr = rcArr.tolist()

			last_value = bw.stats(chromo, lastBin_start, lastBin_end, type="mean", nBins=1)[0]
			if(last_value == None):
				last_value = float(0)
			rcArr.extend([last_value])
			rcArr = np.array(rcArr)
		else:
			rcArr = np.array(bw.stats(chromo, region_start, region_end, type="mean", nBins=nBins))
			rcArr[np.where(rcArr==None)] = float(0)
		rcArr = rcArr / vari.CTRLSCALER[rep]
		bw.close()

		## predicted read counts
		prdvals = np.exp(np.sum(f['X'][0:]* vari.COEFCTRL[rep, 1:], axis=1) + vari.COEFCTRL[rep, 0])
		prdvals[highRC_idx] = np.exp(np.sum(f['X'][0:][highRC_idx] * vari.COEFCTRL_HIGHRC[rep, 1:], axis=1) + vari.COEFCTRL_HIGHRC[rep, 0])

		rcArr = rcArr - prdvals
		rcArr = rcArr[selectedIdx]

		idx = np.where( (rcArr < np.finfo(np.float32).min) | (rcArr > np.finfo(np.float32).max))
		if(len(idx[0]) > 0):
			tempStarts = np.delete(starts, idx)
			rcArr = np.delete(rcArr, idx)
			if(len(rcArr) > 0):
				writeBedFile(subfinalCtrlNames[rep], tempStarts, rcArr, analysis_end)
			else:
				os.remove(subfinalCtrlNames[rep])
				subfinalCtrlNames[rep] = None
		else:
			if(len(rcArr) > 0):
				writeBedFile(subfinalCtrlNames[rep], starts, rcArr, analysis_end)
			else:
				os.remove(subfinalCtrlNames[rep])
				subfinalCtrlNames[rep] = None

	for rep in range(vari.EXPBW_NUM):
		## observed read counts
		bw = pyBigWig.open(vari.EXPBW_NAMES[rep])
		if(lastBin == True):
			rcArr = np.array(bw.stats(chromo, region_start, region_end, type="mean", nBins=(nBins-1)))
			rcArr[np.where(rcArr==None)] = float(0)
			rcArr = rcArr.tolist()

			last_value = bw.stats(chromo, lastBin_start, lastBin_end, type="mean", nBins=1)[0]
			if(last_value == None):
				last_value = float(0)
			rcArr.extend([last_value])
			rcArr = np.array(rcArr)
		else:
			rcArr = np.array(bw.stats(chromo, region_start, region_end, type="mean", nBins=nBins))
			rcArr[np.where(rcArr==None)] = float(0)
		rcArr = rcArr / vari.EXPSCALER[rep]
		bw.close()


		## predicted read counts
		prdvals = np.exp(np.sum(f['X'][0:]* vari.COEFEXP[rep, 1:], axis=1) + vari.COEFEXP[rep, 0])
		prdvals[highRC_idx] = np.exp(np.sum(f['X'][0:][highRC_idx] * vari.COEFEXP_HIGHRC[rep, 1:], axis=1) + vari.COEFEXP_HIGHRC[rep, 0])

		rcArr = rcArr - prdvals
		rcArr = rcArr[selectedIdx]		

		idx = np.where( (rcArr < np.finfo(np.float32).min) | (rcArr > np.finfo(np.float32).max))
		if(len(idx[0]) > 0):
			tempStarts = np.delete(starts, idx)
			rcArr = np.delete(rcArr, idx)
			if(len(rcArr) > 0):
				writeBedFile(subfinalExpNames[rep], tempStarts, rcArr, analysis_end)
			else:
				os.remove(subfinalExpNames[rep])
				subfinalExpNames[rep] = None
		else:
			if(len(rcArr) > 0):
				writeBedFile(subfinalExpNames[rep], starts, rcArr, analysis_end)
			else:
				os.remove(subfinalExpNames[rep])
				subfinalExpNames[rep] = None

	f.close()
	os.remove(covariFileName)

	return_array = [subfinalCtrlNames, subfinalExpNames, chromo]

	return return_array


cpdef selectIdx(chromo, region_start, region_end, lastBin_start, lastBin_end, nBins):

	ctrlRC = []
	for rep in range(vari.CTRLBW_NUM):
		bw = pyBigWig.open(vari.CTRLBW_NAMES[rep])
	
		if(lastBin_start != None):
			temp = np.array(bw.stats(chromo, region_start, region_end, type="mean", nBins=(nBins-1)))
			temp[np.where(temp==None)] = float(0)
			temp = temp.tolist()

			last_value = bw.stats(chromo, lastBin_start, lastBin_end, type="mean", nBins=1)[0]
			if(last_value == None):
				last_value = float(0)
			temp.extend([last_value])
			temp = np.array(temp)
		else:
			temp = np.array(bw.stats(chromo, region_start, region_end, type="mean", nBins=nBins))
			temp[np.where(temp==None)] = float(0)

		bw.close()
		ctrlRC.append(temp.tolist())

		if(rep == 0):
			rc_sum = temp
			highRC_idx = np.where(temp > vari.HIGHRC)[0]

		else:
			rc_sum = rc_sum + temp

	ctrlRC = np.nanmean(ctrlRC, axis=0)
	idx1 = np.where(ctrlRC > 0)[0].tolist()

	expRC = []
	for rep in range(vari.EXPBW_NUM):
		bw = pyBigWig.open(vari.EXPBW_NAMES[rep])

		if(lastBin_start != None):
			temp = np.array(bw.stats(chromo, region_start, region_end, type="mean", nBins=(nBins-1)))
			temp[np.where(temp==None)] = float(0)
			temp = temp.tolist()

			last_value = bw.stats(chromo, lastBin_start, lastBin_end, type="mean", nBins=1)[0]
			if(last_value == None):
				last_value = float(0)
			temp.extend([last_value])
			temp = np.array(temp)
		else:
			temp = np.array(bw.stats(chromo, region_start, region_end, type="mean", nBins=nBins))
			temp[np.where(temp==None)] = float(0)

		bw.close()
		expRC.append(temp.tolist())

		rc_sum = rc_sum + temp

	expRC = np.nanmean(expRC, axis=0)
	idx2 = np.where(expRC > 0)[0].tolist()
	idx3 = np.where(rc_sum > vari.FILTERVALUE)[0].tolist()

	idx_temp = np.intersect1d(idx1, idx2)
	idx = np.intersect1d(idx_temp, idx3)


	if(len(idx) == 0):
		return np.array([]), np.array([]), np.array([])

	if(lastBin_start != None):
		starts = np.arange(region_start, region_end, vari.BINSIZE)
		starts = starts.tolist()
		starts.extend([region_end])
		starts = np.array(starts)
	else:
		starts = np.arange(region_start, region_end, vari.BINSIZE)

	
	starts = starts[idx]

	return idx, highRC_idx, starts
	

cpdef writeBedFile(subfileName, tempStarts, tempSignalvals, analysis_end):
	subfile = open(subfileName, "w")
	
	tempSignalvals = tempSignalvals.astype(int)
	numIdx = len(tempSignalvals)

	idx = 0
	prevStart = tempStarts[idx]
	prevRC = tempSignalvals[idx]
	line = [prevStart, (prevStart + vari.BINSIZE), prevRC]
	if(numIdx == 1):
		subfile.write('\t'.join([str(x) for x in line]) + "\n")
		subfile.close()
		return

	idx = 1
	while(idx < numIdx):
		currStart = tempStarts[idx]
		currRC = tempSignalvals[idx]

		if( (currStart == (prevStart + vari.BINSIZE)) and (currRC == prevRC) ):
			line[1] = currStart + vari.BINSIZE
			prevStart = currStart
			prevRC = currRC
			idx = idx + 1
		else:
			### End a current line
			subfile.write('\t'.join([str(x) for x in line]) + "\n")

			### Start a new line
			line = [currStart, (currStart+vari.BINSIZE), currRC]
			prevStart = currStart
			prevRC = currRC
			idx = idx + 1

		if(idx == numIdx):
			line[1] = analysis_end	
			subfile.write('\t'.join([str(x) for x in line]) + "\n")
			subfile.close()
			break

	return






