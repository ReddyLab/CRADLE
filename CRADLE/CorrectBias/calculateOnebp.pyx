import gc
import math
import os
import random
import sys
import tempfile
import time
import warnings
import h5py
import numpy as np
import scipy
import statsmodels.api as sm
import py2bit
import pyBigWig

from CRADLE.CorrectBias import vari
from CRADLE.correctbiasutils.cython import writeBedFile

cpdef calculateContinuousFrag(chromo, analysisStart, analysisEnd, binStart, binEnd, nBins, lastBin):

	warnings.filterwarnings('ignore', r'All-NaN slice encountered')
	warnings.filterwarnings('ignore', r'Mean of empty slice')


	###### CALCULATE INDEX VARIABLE
	fragStart = binStart + 1 - vari.FRAGLEN
	fragEnd = binEnd + vari.FRAGLEN  # not included
	shearStart = fragStart - 2
	shearEnd = fragEnd + 2 # not included

	faFile = py2bit.open(vari.FA)
	chromoEnd = int(faFile.chroms(chromo))

	if shearStart < 1:
		shearStart = 1
		fragStart = 3
		binStart = max(binStart, fragStart)
		analysisStart = max(analysisStart, fragStart)

		###### EDIT BINSTART/ BINEND
		nBins = int( (analysisEnd - analysisStart) / float(vari.BINSIZE) )
		leftValue = (analysisEnd - analysisStart) % int(vari.BINSIZE)
		if leftValue != 0:
			nBins = nBins + 1
			lastBin = True
			binEnd = int( (analysisStart + (nBins-1) * vari.BINSIZE + analysisEnd) / float(2) )
		else:
			lastBin = False
			binEnd = binStart + (nBins-1) * vari.BINSIZE

		fragEnd = binEnd + vari.FRAGLEN
		shearEnd = fragEnd + 2

	if shearEnd > chromoEnd:
		analysisEndModified = min(analysisEnd, chromoEnd - 2)  # not included

		if analysisEndModified == analysisEnd:
			shearEnd = chromoEnd
			fragEnd = shearEnd - 2
		else:
			analysisEnd = analysisEndModified

			###### EDIT BINSTART/ BINEND
			nBins = int( (analysisEnd - analysisStart) / float(vari.BINSIZE) )
			leftValue = (analysisEnd - analysisStart) % int(vari.BINSIZE)
			if leftValue != 0:
				nBins = nBins + 1
				lastBin = True
				binEnd = int( (analysisStart + (nBins-1) * vari.BINSIZE + analysisEnd) / float(2) )
			else:
				lastBin = False
				binEnd = binStart + (nBins-1) * vari.BINSIZE

			fragEnd = binEnd + vari.FRAGLEN
			shearEnd = fragEnd + 2

			if shearEnd > chromoEnd:
				shearEnd = chromoEnd
				fragEnd = shearEnd - 2

	###### GENERATE A RESULT MATRIX
	result = makeMatrixContinuousFrag(binStart, binEnd, nBins)

	###### GET SEQUENCE
	fa = faFile.sequence(chromo, (shearStart-1), (shearEnd-1))
	faFile.close()

	##### OPEN BIAS FILES
	if vari.MAP == 1:
		mapFile = pyBigWig.open(vari.MAPFILE)
		mapValue = np.array(mapFile.values(chromo, fragStart, fragEnd))

		mapValue[np.where(mapValue == 0)] = np.nan
		mapValue = np.log(mapValue)
		mapValue[np.where(np.isnan(mapValue))] = float(-6)

		mapValueView = memoryView(mapValue)
		mapFile.close()
		del mapFile, mapValue

	if vari.GQUAD == 1:
		gquadFile = [0] * len(vari.GQAUDFILE)
		gquadValue = [0] * len(vari.GQAUDFILE)

		for i in range(len(vari.GQAUDFILE)):
			gquadFile[i] = pyBigWig.open(vari.GQAUDFILE[i])
			gquadValue[i] = gquadFile[i].values(chromo, fragStart, fragEnd)
			gquadFile[i].close()

		gquadValue = np.array(gquadValue)
		gquadValue = np.nanmax(gquadValue, axis=0)
		gquadValue[np.where(gquadValue == 0)] = np.nan
		gquadValue = np.log(gquadValue / float(vari.GQAUD_MAX))

		gquadValue[np.where(np.isnan(gquadValue))] = float(-5)
		gquadView = memoryView(gquadValue)

		del gquadFile, gquadValue


	##### INDEX IN 'fa'
	startIdx = 2  # index in the fasta file (Included in the range)
	endIdx = (fragEnd - vari.FRAGLEN) - shearStart + 1   # index in the fasta file (Not included in the range)

	##### INITIALIZE VARIABLES
	if vari.SHEAR == 1:
		pastMer1 = -1
		pastMer2 = -1
	if vari.PCR == 1:
		pastStartGibbs = -1

	resultStartIdx = -1
	resultEndIdx = -1

	##### STORE COVARI RESULTS
	covariFileTemp = tempfile.NamedTemporaryFile(suffix=".hdf5", dir=vari.OUTPUT_DIR, delete=True)
	covariFileName = covariFileTemp.name
	covariFileTemp.close()

	f = h5py.File(covariFileName, "w")
	covariFile = f.create_dataset("X", (nBins, vari.COVARI_NUM), dtype='f', compression="gzip")

	for idx in range(startIdx, endIdx):
		covariIdx = [0] * vari.COVARI_NUM
		covariIdxPtr = 0

		if vari.SHEAR == 1:
			###  mer1
			mer1 = fa[(idx-2):(idx+3)]
			if 'N' in mer1:
				pastMer1 = -1
				mgwIdx = vari.N_MGW
				protIdx = vari.N_PROT
			else:
				if pastMer1 == -1: # there is no information on pastMer1
					pastMer1, mgwIdx, protIdx = find5merProb(mer1)
				else:
					pastMer1, mgwIdx, protIdx = edit5merProb(pastMer1, mer1[0], mer1[4])

			###  mer2
			fragEndIdx = idx + vari.FRAGLEN
			mer2 = fa[(fragEndIdx-3):(fragEndIdx+2)]
			if 'N' in mer2:
				pastMer2 = -1
				mgwIdx = mgwIdx + vari.N_MGW
				protIdx = protIdx + vari.N_PROT
			else:
				if pastMer2 == -1:
					pastMer2, add1, add2 = findComple5merProb(mer2)
				else:
					pastMer2, add1, add2 = editComple5merProb(pastMer2, mer2[0], mer2[4])
				mgwIdx = mgwIdx + add1
				protIdx = protIdx + add2

			covariIdx[covariIdxPtr] = mgwIdx
			covariIdx[covariIdxPtr+1] = protIdx
			covariIdxPtr = covariIdxPtr + 2


		if vari.PCR == 1:
			faIdx = fa[idx:(idx+vari.FRAGLEN)]
			if pastStartGibbs == -1:
				startGibbs, gibbs = findStartGibbs(faIdx, vari.FRAGLEN)
			else:
				oldDimer = faIdx[0:2].upper()
				newDimer = faIdx[(vari.FRAGLEN-2):vari.FRAGLEN].upper()
				startGibbs, gibbs = editStartGibbs(oldDimer, newDimer, pastStartGibbs)

			annealIdx, denatureIdx = convertGibbs(gibbs)

			covariIdx[covariIdxPtr] = annealIdx
			covariIdx[covariIdxPtr+1] = denatureIdx
			covariIdxPtr = covariIdxPtr + 2

		if vari.MAP == 1:
			map1 = mapValueView[(idx-2)]
			map2 = mapValueView[(idx+vari.FRAGLEN-2-vari.KMER)]
			mapIdx = map1 + map2

			covariIdx[covariIdxPtr] = mapIdx
			covariIdxPtr = covariIdxPtr + 1

		if vari.GQUAD == 1:
			gquadIdx = np.nanmax(np.asarray(gquadView[(idx-2):(idx+vari.FRAGLEN-2)]))

			covariIdx[covariIdxPtr] = gquadIdx
			covariIdxPtr = covariIdxPtr + 1

		### DETERMINE WHICH ROWS TO EDIT IN RESULT MATRIX
		thisFragStart = idx + shearStart
		thisFragEnd = thisFragStart + vari.FRAGLEN

		if resultStartIdx == -1:
			resultStartIdx = 0
			resultEndIdx = 1 # not included
			if not np.isnan(result[resultEndIdx, 0]):
				while result[resultEndIdx, 0] < thisFragEnd:
					resultEndIdx = resultEndIdx + 1
					if resultEndIdx > vari.FRAGLEN:
						resultEndIdx = resultEndIdx - (vari.FRAGLEN+1)
					if np.isnan(result[resultEndIdx, 0]):
						break
			maxBinPos = binStart + vari.FRAGLEN
			numPoppedPos = 0
		else:

			while result[resultStartIdx, 0] < thisFragStart:
				## pop the element
				line = []
				for covariPos in range(vari.COVARI_NUM):
					line.extend([ result[resultStartIdx, (covariPos+1)]  ])
					result[resultStartIdx, (covariPos+1)] = float(0)
				covariFile[numPoppedPos] = line


				numPoppedPos = numPoppedPos + 1
				if maxBinPos >= binEnd:
					result[resultStartIdx, 0] = np.nan
				else:
					result[resultStartIdx, 0] = maxBinPos + 1
					maxBinPos = maxBinPos + 1

				resultStartIdx = resultStartIdx + 1
				if resultStartIdx > vari.FRAGLEN:
					resultStartIdx = resultStartIdx - (vari.FRAGLEN+1)

			if not np.isnan(result[resultEndIdx, 0]):
				while result[resultEndIdx, 0] < thisFragEnd:
					resultEndIdx = resultEndIdx + 1
					if resultEndIdx > vari.FRAGLEN:
						resultEndIdx = resultEndIdx - (vari.FRAGLEN+1)
					if np.isnan(result[resultEndIdx, 0]):
						break

		if resultEndIdx < resultStartIdx:
			for pos in range(resultStartIdx, (vari.FRAGLEN+1)):
				for covariPos in range(vari.COVARI_NUM):
					result[pos, covariPos+1] = result[pos, covariPos+1] + covariIdx[covariPos]
			for pos in range(0, resultEndIdx):
				for covariPos in range(vari.COVARI_NUM):
					result[pos, covariPos+1] = result[pos, covariPos+1] + covariIdx[covariPos]
		else:
			for pos in range(resultStartIdx, resultEndIdx):
				for covariPos in range(vari.COVARI_NUM):
					result[pos, covariPos+1] = result[pos, covariPos+1] + covariIdx[covariPos]

		if idx == (endIdx-1): # the last fragment
			### pop the rest of positions that are not np.nan
			if resultEndIdx < resultStartIdx:
				for pos in range(resultStartIdx, (vari.FRAGLEN+1)):
					line = []
					for covariPos in range(vari.COVARI_NUM):
						line.extend([ result[pos, (covariPos+1)]  ])
					covariFile[numPoppedPos] = line

					numPoppedPos = numPoppedPos + 1

				for pos in range(0, resultEndIdx):
					line = []
					for covariPos in range(vari.COVARI_NUM):
						line.extend([ result[pos, (covariPos+1)]  ])
					covariFile[numPoppedPos] = line

					numPoppedPos = numPoppedPos + 1
			else:
				for pos in range(resultStartIdx, resultEndIdx):
					line = []
					for covariPos in range(vari.COVARI_NUM):
						line.extend([ result[pos, (covariPos+1)]  ])
					covariFile[numPoppedPos] = line

					numPoppedPos = numPoppedPos + 1

	f.close()


	return correctReadCount(covariFileName, chromo, analysisStart, analysisEnd, lastBin, nBins)


cpdef calculateDiscreteFrag(chromo, analysisStart, analysisEnd, binStart, binEnd, nBins, lastBin):
	warnings.filterwarnings('ignore', r'All-NaN slice encountered')
	warnings.filterwarnings('ignore', r'Mean of empty slice')


	###### CALCULATE INDEX VARIABLE
	fragStart = binStart + 1 - vari.FRAGLEN
	fragEnd = binEnd + vari.FRAGLEN  # not included
	shearStart = fragStart - 2
	shearEnd = fragEnd + 2 # not included

	faFile = py2bit.open(vari.FA)
	chromoEnd = int(faFile.chroms(chromo))

	if shearStart < 1:
		shearStart = 1
		fragStart = 3
		binStart = max(binStart, fragStart)
		analysisStart = max(analysisStart, fragStart)

		###### EDIT BINSTART/ BINEND
		nBins = int( (analysisEnd-analysisStart) / float(vari.BINSIZE) )
		leftValue = (analysisEnd - analysisStart) % int(vari.BINSIZE)
		if leftValue != 0:
			nBins = nBins + 1
			lastBin = True
			binEnd = int( (analysisStart + (nBins-1) * vari.BINSIZE + analysisEnd) / float(2) )
		else:
			lastBin = False
			binEnd = binStart + (nBins-1) * vari.BINSIZE

		fragEnd = binEnd + vari.FRAGLEN
		shearEnd = fragEnd + 2

	if shearEnd > chromoEnd:
		analysisEndModified = min(analysisEnd, chromoEnd - 2)  # not included

		if analysisEndModified == analysisEnd:
			shearEnd = chromoEnd
			fragEnd = shearEnd - 2
		else:
			analysisEnd = analysisEndModified

			###### EDIT BINSTART/ BINEND
			nBins = int( (analysisEnd-analysisStart) / float(vari.BINSIZE) )
			leftValue = (analysisEnd - analysisStart) % int(vari.BINSIZE)
			if leftValue != 0:
				nBins = nBins + 1
				lastBin = True
				binEnd = int( (analysisStart + (nBins-1) * vari.BINSIZE + analysisEnd) / float(2) )
			else:
				lastBin = False
				binEnd = binStart + (nBins-1) * vari.BINSIZE

			fragEnd = binEnd + vari.FRAGLEN
			shearEnd = fragEnd + 2

			if shearEnd > chromoEnd:
				shearEnd = chromoEnd
				fragEnd = shearEnd - 2


	###### GENERATE A RESULT MATRIX
	result = makeMatrixDiscreteFrag(binStart, binEnd, nBins)

	###### GET SEQUENCE
	fa = faFile.sequence(chromo, (shearStart-1), (shearEnd-1))
	faFile.close()

        ##### OPEN BIAS FILES
	if vari.MAP == 1:
		mapFile = pyBigWig.open(vari.MAPFILE)
		mapValue = np.array(mapFile.values(chromo, fragStart, fragEnd))

		mapValue[np.where(mapValue == 0)] = np.nan
		mapValue = np.log(mapValue)
		mapValue[np.where(np.isnan(mapValue))] = float(-6)

		mapValueView = memoryView(mapValue)
		mapFile.close()
		del mapFile, mapValue

	if vari.GQUAD == 1:
		gquadFile = [0] * len(vari.GQAUDFILE)
		gquadValue = [0] * len(vari.GQAUDFILE)

		for i in range(len(vari.GQAUDFILE)):
			gquadFile[i] = pyBigWig.open(vari.GQAUDFILE[i])
			gquadValue[i] = gquadFile[i].values(chromo, fragStart, fragEnd)
			gquadFile[i].close()

		gquadValue = np.array(gquadValue)
		gquadValue = np.nanmax(gquadValue, axis=0)
		gquadValue[np.where(gquadValue == 0)] = np.nan
		gquadValue = np.log(gquadValue / float(vari.GQAUD_MAX))

		gquadValue[np.where(np.isnan(gquadValue))] = float(-5)
		gquadView = memoryView(gquadValue)

		del gquadFile, gquadValue


	##### STORE COVARI RESULTS
	covariFileTemp = tempfile.NamedTemporaryFile(suffix=".hdf5", dir=vari.OUTPUT_DIR, delete=True)
	covariFileName = covariFileTemp.name
	covariFileTemp.close()

	f = h5py.File(covariFileName, "w")
	covariFile = f.create_dataset("X", (nBins, vari.COVARI_NUM), dtype='f', compression="gzip")

	resultIdx = 0
	while resultIdx < nBins: # for each bin
		if resultIdx == (nBins-1):
			pos = binEnd
		else:
			pos = binStart + resultIdx * vari.BINSIZE

		thisBinFirstFragStart = pos + 1 - vari.FRAGLEN
		thisBinLastFragStart = pos

		if thisBinFirstFragStart < 3:
			thisBinFirstFragStart = 3
		if (thisBinLastFragStart + vari.FRAGLEN) > (chromoEnd - 2):
			thisBinLastFragStart = chromoEnd - 2 - vari.FRAGLEN

		thisBinNumFrag = thisBinLastFragStart - thisBinFirstFragStart + 1

		thisBinFirstFragStartIdx = thisBinFirstFragStart - shearStart

		##### INITIALIZE VARIABLES
		if vari.SHEAR == 1:
			pastMer1 = -1
			pastMer2 = -1
		if vari.PCR == 1:
			pastStartGibbs = -1

		line = [0.0] * vari.COVARI_NUM
		for binFragIdx in range(thisBinNumFrag):
			idx = thisBinFirstFragStartIdx + binFragIdx

			covariIdx = [0] * vari.COVARI_NUM
			covariIdxPtr = 0

			if vari.SHEAR == 1:
				###  mer1
				mer1 = fa[(idx-2):(idx+3)]
				if 'N' in mer1:
					pastMer1 = -1
					mgwIdx = vari.N_MGW
					protIdx = vari.N_PROT
				else:
					if pastMer1 == -1: # there is no information on pastMer1
						pastMer1, mgwIdx, protIdx = find5merProb(mer1)
					else:
						pastMer1, mgwIdx, protIdx = edit5merProb(pastMer1, mer1[0], mer1[4])

                        	###  mer2
				fragEndIdx = idx + vari.FRAGLEN
				mer2 = fa[(fragEndIdx-3):(fragEndIdx+2)]
				if 'N' in mer2:
					pastMer2 = -1
					mgwIdx = mgwIdx + vari.N_MGW
					protIdx = protIdx + vari.N_PROT
				else:
					if pastMer2 == -1:
						pastMer2, add1, add2 = findComple5merProb(mer2)
					else:
						pastMer2, add1, add2 = editComple5merProb(pastMer2, mer2[0], mer2[4])
					mgwIdx = mgwIdx + add1
					protIdx = protIdx + add2

				covariIdx[covariIdxPtr] = mgwIdx
				covariIdx[covariIdxPtr+1] = protIdx
				covariIdxPtr = covariIdxPtr + 2

			if vari.PCR == 1:
				faIdx = fa[idx:(idx+vari.FRAGLEN)]
				if pastStartGibbs == -1:
					startGibbs, gibbs = findStartGibbs(faIdx, vari.FRAGLEN)
				else:
					oldDimer = faIdx[0:2].upper()
					newDimer = faIdx[(vari.FRAGLEN-2):vari.FRAGLEN].upper()
					startGibbs, gibbs = editStartGibbs(oldDimer, newDimer, pastStartGibbs)

				annealIdx, denatureIdx = convertGibbs(gibbs)

				covariIdx[covariIdxPtr] = annealIdx
				covariIdx[covariIdxPtr+1] = denatureIdx
				covariIdxPtr = covariIdxPtr + 2

			if vari.MAP == 1:
				map1 = mapValueView[(idx-2)]
				map2 = mapValueView[(idx+vari.FRAGLEN-2-vari.KMER)]
				mapIdx = map1 + map2

				covariIdx[covariIdxPtr] = mapIdx
				covariIdxPtr = covariIdxPtr + 1

			if vari.GQUAD == 1:
				gquadIdx = np.nanmax(np.asarray(gquadView[(idx-2):(idx+vari.FRAGLEN-2)]))

				covariIdx[covariIdxPtr] = gquadIdx
				covariIdxPtr = covariIdxPtr + 1


			for covariPos in range(vari.COVARI_NUM):
				line[covariPos] = line[covariPos] + covariIdx[covariPos]

		covariFile[resultIdx] = line

		resultIdx = resultIdx + 1

		if resultIdx == nBins:
			break

	f.close()

	return correctReadCount(covariFileName, chromo, analysisStart, analysisEnd, lastBin, nBins)


cpdef calculateTrainCovariates(args):
	### supress numpy nan-error message
	warnings.filterwarnings('ignore', r'All-NaN slice encountered')
	warnings.filterwarnings('ignore', r'Mean of empty slice')

	chromo = args[0]
	analysisStart = int(args[1])  # Genomic coordinates(starts from 1)
	analysisEnd = int(args[2])  # not included

	binStart = int((analysisStart + analysisStart + vari.BINSIZE) / float(2))
	binEnd = int((analysisEnd - vari.BINSIZE + analysisEnd) / float(2))

	###### CALCULATE INDEX VARIABLE
	fragStart = binStart + 1 - vari.FRAGLEN
	fragEnd = binEnd + vari.FRAGLEN  # not included
	shearStart = fragStart - 2
	shearEnd = fragEnd + 2 # not included

	faFile = py2bit.open(vari.FA)
	chromoEnd = int(faFile.chroms(chromo))

	if shearStart < 1:
		shearStart = 1
		fragStart = 3
		binStart = max(binStart, fragStart)
		analysisStart = max(analysisStart, fragStart)

	if shearEnd > chromoEnd:
		shearEnd = chromoEnd
		fragEnd = shearEnd - 2
		binEnd = min(binEnd, (fragEnd-1))
		analysisEnd = min(analysisEnd, fragEnd)  # not included

	###### GENERATE A RESULT MATRIX
	nBins = binEnd - binStart + 1
	result = makeMatrixContinuousFragTrain(binStart, binEnd, nBins)

	###### GET SEQUENCE
	fa = faFile.sequence(chromo, (shearStart-1), (shearEnd-1))
	faFile.close()

        ##### OPEN BIAS FILES
	if vari.MAP == 1:
		mapFile = pyBigWig.open(vari.MAPFILE)
		mapValue = np.array(mapFile.values(chromo, fragStart, fragEnd))

		mapValue[np.where(mapValue == 0)] = np.nan
		mapValue = np.log(mapValue)
		mapValue[np.where(np.isnan(mapValue))] = float(-6)

		mapValueView = memoryView(mapValue)
		mapFile.close()
		del mapFile, mapValue

	if vari.GQUAD == 1:
		gquadFile = [0] * len(vari.GQAUDFILE)
		gquadValue = [0] * len(vari.GQAUDFILE)

		for i in range(len(vari.GQAUDFILE)):
			gquadFile[i] = pyBigWig.open(vari.GQAUDFILE[i])
			gquadValue[i] = gquadFile[i].values(chromo, fragStart, fragEnd)
			gquadFile[i].close()

		gquadValue = np.array(gquadValue)
		gquadValue = np.nanmax(gquadValue, axis=0)
		gquadValue[np.where(gquadValue == 0)] = np.nan
		gquadValue = np.log(gquadValue / float(vari.GQAUD_MAX))

		gquadValue[np.where(np.isnan(gquadValue))] = float(-5)
		gquadView = memoryView(gquadValue)

		del gquadFile, gquadValue


	##### INDEX IN 'fa'
	startIdx = 2  # index in the fasta file (Included in the range)
	endIdx = (fragEnd - vari.FRAGLEN) - shearStart + 1   # index in the fasta file (Not included in the range)

        ##### INITIALIZE VARIABLES
	if vari.SHEAR == 1:
		pastMer1 = -1
		pastMer2 = -1
	if vari.PCR == 1:
		pastStartGibbs = -1

	resultStartIdx = -1
	resultEndIdx = -1

	##### STORE COVARI RESULTS
	covariFileTemp = tempfile.NamedTemporaryFile(suffix=".hdf5", dir=vari.OUTPUT_DIR, delete=True)
	covariFileName = covariFileTemp.name
	covariFileTemp.close()

	f = h5py.File(covariFileName, "w")
	covariFile = f.create_dataset("X", (nBins, vari.COVARI_NUM), dtype='f', compression="gzip")

	for idx in range(startIdx, endIdx):
		covariIdx = [0] * vari.COVARI_NUM
		covariIdxPtr = 0

		if vari.SHEAR == 1:
			###  mer1
			mer1 = fa[(idx-2):(idx+3)]
			if 'N' in mer1:
				pastMer1 = -1
				mgwIdx = vari.N_MGW
				protIdx = vari.N_PROT
			else:
				if pastMer1 == -1: # there is no information on pastMer1
					pastMer1, mgwIdx, protIdx = find5merProb(mer1)
				else:
					pastMer1, mgwIdx, protIdx = edit5merProb(pastMer1, mer1[0], mer1[4])

			##  mer2
			fragEndIdx = idx + vari.FRAGLEN
			mer2 = fa[(fragEndIdx-3):(fragEndIdx+2)]
			if 'N' in mer2:
				pastMer2 = -1
				mgwIdx = mgwIdx + vari.N_MGW
				protIdx = protIdx + vari.N_PROT
			else:
				if pastMer2 == -1:
					pastMer2, add1, add2 = findComple5merProb(mer2)
				else:
					pastMer2, add1, add2 = editComple5merProb(pastMer2, mer2[0], mer2[4])
				mgwIdx = mgwIdx + add1
				protIdx = protIdx + add2

			covariIdx[covariIdxPtr] = mgwIdx
			covariIdx[covariIdxPtr+1] = protIdx
			covariIdxPtr = covariIdxPtr + 2


		if vari.PCR == 1:
			faIdx = fa[idx:(idx+vari.FRAGLEN)]
			if pastStartGibbs == -1:
				startGibbs, gibbs = findStartGibbs(faIdx, vari.FRAGLEN)
			else:
				oldDimer = faIdx[0:2].upper()
				newDimer = faIdx[(vari.FRAGLEN-2):vari.FRAGLEN].upper()
				startGibbs, gibbs = editStartGibbs(oldDimer, newDimer, pastStartGibbs)

			annealIdx, denatureIdx = convertGibbs(gibbs)

			covariIdx[covariIdxPtr] = annealIdx
			covariIdx[covariIdxPtr+1] = denatureIdx
			covariIdxPtr = covariIdxPtr + 2

		if vari.MAP == 1:
			map1 = mapValueView[(idx-2)]
			map2 = mapValueView[(idx+vari.FRAGLEN-2-vari.KMER)]
			mapIdx = map1 + map2

			covariIdx[covariIdxPtr] = mapIdx
			covariIdxPtr = covariIdxPtr + 1

		if vari.GQUAD == 1:
			gquadIdx = np.nanmax(np.asarray(gquadView[(idx-2):(idx+vari.FRAGLEN-2)]))

			covariIdx[covariIdxPtr] = gquadIdx
			covariIdxPtr = covariIdxPtr + 1


		### DETERMINE WHICH ROWS TO EDIT IN RESULT MATRIX
		thisFragStart = idx + shearStart
		thisFragEnd = thisFragStart + vari.FRAGLEN

		if resultStartIdx == -1:
			resultStartIdx = 0
			resultEndIdx = 1 # not include
			if not np.isnan(result[resultEndIdx, 0]):
				while result[resultEndIdx, 0] < thisFragEnd:
					resultEndIdx = resultEndIdx + 1
					if resultEndIdx > vari.FRAGLEN:
						resultEndIdx = resultEndIdx - (vari.FRAGLEN+1)
					if np.isnan(result[resultEndIdx, 0]):
						break
			maxBinPos = binStart + vari.FRAGLEN
			numPoppedPos = 0
		else:
			while result[resultStartIdx, 0] < thisFragStart:
				## pop the element
				line = []
				for covariPos in range(vari.COVARI_NUM):
					line.extend([ result[resultStartIdx, (covariPos+1)]  ])
					result[resultStartIdx, (covariPos+1)] = float(0)
				covariFile[numPoppedPos] = line

				numPoppedPos = numPoppedPos + 1
				if maxBinPos >= binEnd:
					result[resultStartIdx, 0] = np.nan
				else:
					result[resultStartIdx, 0] = maxBinPos + 1
					maxBinPos = maxBinPos + 1

				resultStartIdx = resultStartIdx + 1
				if resultStartIdx > vari.FRAGLEN:
					resultStartIdx = resultStartIdx - (vari.FRAGLEN+1)


			if not np.isnan(result[resultEndIdx, 0]):
				while result[resultEndIdx, 0] < thisFragEnd:
					resultEndIdx = resultEndIdx + 1
					if resultEndIdx > vari.FRAGLEN:
						resultEndIdx = resultEndIdx - (vari.FRAGLEN+1)
					if np.isnan(result[resultEndIdx, 0]):
						break


		if resultEndIdx < resultStartIdx:
			for pos in range(resultStartIdx, (vari.FRAGLEN+1)):
				for covariPos in range(vari.COVARI_NUM):
					result[pos, covariPos+1] = result[pos, covariPos+1] + covariIdx[covariPos]
			for pos in range(0, resultEndIdx):
				for covariPos in range(vari.COVARI_NUM):
					result[pos, covariPos+1] = result[pos, covariPos+1] + covariIdx[covariPos]
		else:
			for pos in range(resultStartIdx, resultEndIdx):
				for covariPos in range(vari.COVARI_NUM):
					result[pos, covariPos+1] = result[pos, covariPos+1] + covariIdx[covariPos]

		if idx == (endIdx-1): # the last fragment
			### pop the rest of positions that are not np.nan
			if resultEndIdx < resultStartIdx:
				for pos in range(resultStartIdx, (vari.FRAGLEN+1)):
					line = []
					for covariPos in range(vari.COVARI_NUM):
						line.extend([ result[pos, (covariPos+1)]  ])
					covariFile[numPoppedPos] = line

					numPoppedPos = numPoppedPos + 1

				for pos in range(0, resultEndIdx):
					line = []
					for covariPos in range(vari.COVARI_NUM):
						line.extend([ result[pos, (covariPos+1)]  ])
					covariFile[numPoppedPos] = line

					numPoppedPos = numPoppedPos + 1
			else:
				for pos in range(resultStartIdx, resultEndIdx):
					line = []
					for covariPos in range(vari.COVARI_NUM):
						line.extend([ result[pos, (covariPos+1)]  ])
					covariFile[numPoppedPos] = line

					numPoppedPos = numPoppedPos + 1

	f.close()

	returnLine = [covariFileName, numPoppedPos]


	### output read counts
	for rep in range(vari.SAMPLE_NUM):
		rcFileTemp = tempfile.NamedTemporaryFile(suffix=".hdf5", dir=vari.OUTPUT_DIR, delete=True)
		rcFileName = rcFileTemp.name
		rcFileTemp.close()

		f = h5py.File(rcFileName, "w")
		rcFile = f.create_dataset("Y", (nBins, ), dtype='f', compression="gzip")


		returnLine.extend([ rcFileName ])

		if rep < vari.CTRLBW_NUM:
			bw = pyBigWig.open(vari.CTRLBW_NAMES[rep])
		else:
			bw = pyBigWig.open(vari.EXPBW_NAMES[rep-vari.CTRLBW_NUM])

		temp = np.array(bw.values(chromo, analysisStart, analysisEnd))
		numPos = len(temp)


		for binIdx in range(nBins):
			if rep < vari.CTRLBW_NUM:
				if (binIdx + vari.BINSIZE) >= numPos:
					rc = np.nanmean(temp[binIdx:]) / float(vari.CTRLSCALER[rep])
				else:
					rc = np.nanmean(temp[binIdx:(binIdx + vari.BINSIZE)]) / float(vari.CTRLSCALER[rep])
			else:
				if (binIdx + vari.BINSIZE) >= numPos:
					rc = np.nanmean(temp[binIdx:]) / float(vari.EXPSCALER[rep-vari.CTRLBW_NUM])
				else:
					rc = np.nanmean(temp[binIdx:(binIdx + vari.BINSIZE)]) / float(vari.EXPSCALER[rep-vari.CTRLBW_NUM])

			if np.isnan(rc):
				rc = float(0)

			rcFile[binIdx] = rc


		f.close()
		bw.close()

	return returnLine


cpdef calculateTaskCovariates(args):
	### supress numpy nan-error message
	warnings.filterwarnings('ignore', r'All-NaN slice encountered')
	warnings.filterwarnings('ignore', r'Mean of empty slice')

	chromo = args[0]
	analysisStart = int(args[1])  # Genomic coordinates(starts from 1)
	analysisEnd = int(args[2])

	#### DECIDE IF 'calculateContinuousFrag' or 'calculateDiscreteFrag'
	if (analysisStart + vari.BINSIZE) >= analysisEnd:
		firstBinPos = int((analysisStart + analysisEnd) / float(2))
		lastBinPos = firstBinPos
		nBins = 1
		lastBin = False
		result = calculateContinuousFrag(chromo, analysisStart, analysisEnd, firstBinPos, lastBinPos, nBins, lastBin)
		return result

	firstBinPos = int((2*analysisStart + vari.BINSIZE) / float(2))
	if (analysisStart + 2*vari.BINSIZE) > analysisEnd:
		secondBinPos = int((analysisStart + vari.BINSIZE + analysisEnd) / float(2))
		lastBinPos = secondBinPos
		nBins = 2
		lastBin = True
	else:
		secondBinPos = int((2*analysisStart + 3 * vari.BINSIZE) / float(2))
		leftValue = (analysisEnd - analysisStart) % int(vari.BINSIZE)
		nBins = int((analysisEnd - analysisStart) / float(vari.BINSIZE))
		if leftValue != 0:
			nBins = nBins + 1
			lastBin = True
			lastBinPos = int( (analysisStart + (nBins-1) * vari.BINSIZE + analysisEnd) / float(2) )  ## should be included in the analysis
		else:
			lastBin = False
			lastBinPos = firstBinPos + (nBins-1) * vari.BINSIZE ## should be included in the analysis

	if (secondBinPos-firstBinPos) <= vari.FRAGLEN:
		result = calculateContinuousFrag(chromo, analysisStart, analysisEnd, firstBinPos, lastBinPos, nBins, lastBin)
		return result
	else:
		result = calculateDiscreteFrag(chromo, analysisStart, analysisEnd, firstBinPos, lastBinPos, nBins, lastBin)
		return result


cpdef performRegression(covariFiles, scatterplotSamples):
	### Read covariates values (X)
	xNumRows = 0
	for i in range(len(covariFiles)):
		xNumRows = xNumRows + int(covariFiles[i][1])
	xNumCols = vari.COVARI_NUM + 1

	cdef double [:,:] XView = np.ones((xNumRows, xNumCols), dtype=np.float64)

	cdef int rowPtr = 0
	cdef int rowIdx
	cdef int colPtr

	for fileIdx in range(len(covariFiles)):
		subfileName = covariFiles[fileIdx][0]
		f = h5py.File(subfileName, "r")

		rowIdx = 0
		while rowIdx < f['X'].shape[0]:
			temp = f['X'][rowIdx]
			colPtr = 0
			while colPtr < vari.COVARI_NUM:
				XView[rowIdx + rowPtr, colPtr+1] = float(temp[colPtr])
				colPtr = colPtr + 1
			rowIdx = rowIdx + 1
		rowPtr = rowPtr + int(covariFiles[fileIdx][1])

		f.close()
		os.remove(subfileName)


	### COEFFICIENTS
	COEFCTRL = np.zeros((vari.CTRLBW_NUM, (vari.COVARI_NUM+1)), dtype=np.float64)
	COEFEXP = np.zeros((vari.EXPBW_NUM, (vari.COVARI_NUM+1)), dtype=np.float64)

	readCounts = np.zeros(xNumRows, dtype=np.float64)
	cdef double [:] readCountsView = readCounts

	cdef int ptr
	cdef int rcIdx

	ctrlPlotValues = {}
	experiPlotValues = {}

	for rep in range(vari.CTRLBW_NUM):
		ptr = 0

		for fileIdx in range(len(covariFiles)):
			subfileName = covariFiles[fileIdx][rep+2]
			f = h5py.File(subfileName, "r")

			rcIdx = 0
			while rcIdx < f['Y'].shape[0]:
				readCountsView[rcIdx+ptr] = float(f['Y'][rcIdx])
				rcIdx = rcIdx + 1

			ptr = ptr + int(f['Y'].shape[0])

			f.close()
			os.remove(subfileName)


		deleteIdx = np.where( (readCounts < np.finfo(np.float32).min) | (readCounts > np.finfo(np.float32).max))[0]
		if len(deleteIdx) != 0:
			model = sm.GLM(np.delete(readCounts.astype(int), deleteIdx), np.delete(np.array(XView), deleteIdx, axis=0), family=sm.families.Poisson(link=sm.genmod.families.links.log)).fit()
		else:
			model = sm.GLM(readCounts.astype(int), np.array(XView), family=sm.families.Poisson(link=sm.genmod.families.links.log)).fit()

		coef = model.params
		COEFCTRL[rep, ] = coef

		ctrlPlotValues[vari.CTRLBW_NAMES[rep]] = (readCounts[scatterplotSamples], model.fittedvalues[scatterplotSamples])

	for rep in range(vari.EXPBW_NUM):
		ptr = 0
		for fileIdx in range(len(covariFiles)):
			subfileName = covariFiles[fileIdx][rep + 2 + vari.CTRLBW_NUM]
			f = h5py.File(subfileName, "r")

			rcIdx = 0
			while rcIdx < f['Y'].shape[0]:
				readCountsView[rcIdx+ptr] = float(f['Y'][rcIdx])
				rcIdx = rcIdx + 1

			ptr = ptr + int(f['Y'].shape[0])

			f.close()
			os.remove(subfileName)

		deleteIdx = np.where( (readCounts < np.finfo(np.float32).min) | (readCounts > np.finfo(np.float32).max))[0]
		if len(deleteIdx) != 0:
			model = sm.GLM(
				np.delete(readCounts.astype(int), deleteIdx),
				np.delete(np.array(XView), deleteIdx, axis=0),
				family=sm.families.Poisson(link=sm.genmod.families.links.log)
			).fit()
		else:
			model = sm.GLM(
				readCounts.astype(int),
				np.array(XView),
				family=sm.families.Poisson(link=sm.genmod.families.links.log)
			).fit()

		coef = model.params
		COEFEXP[rep, ] = coef

		experiPlotValues[vari.EXPBW_NAMES[rep]] = (readCounts[scatterplotSamples], model.fittedvalues[scatterplotSamples])

	return COEFCTRL, COEFEXP, ctrlPlotValues, experiPlotValues


cpdef memoryView(value):
	cdef double [:] arrayView
	arrayView = value

	return arrayView


cpdef makeMatrixContinuousFragTrain(binStart, binEnd, nBins):

	result = np.zeros(((vari.FRAGLEN+1), (vari.COVARI_NUM+1)), dtype=np.float64)
	for i in range(vari.FRAGLEN+1):
		pos = binStart + i

		if pos > binEnd:
			result[i, 0] = np.nan
		else:
			result[i, 0] = pos

	return result


cpdef makeMatrixContinuousFrag(binStart, binEnd, nBins):

	result = np.zeros(((vari.FRAGLEN+1), (vari.COVARI_NUM+1)), dtype=np.float64)
	for i in range(vari.FRAGLEN+1):
		pos = binStart + i * vari.BINSIZE

		if pos > binEnd:
			result[i, 0] = np.nan
		else:
			result[i, 0] = pos

	if nBins == (vari.FRAGLEN+1):
		result[vari.FRAGLEN, 0] = binEnd

	return result


cpdef makeMatrixDiscreteFrag(binStart, binEnd, nBins):
	cdef double [:,:] resultView = np.zeros((nBins, vari.COVARI_NUM), dtype=np.float64)

	return resultView


cpdef find5merProb(mer5):
	baseInfo = 0
	subtract = -1

	for i in range(5):
		if mer5[i]=='A' or mer5[i]=='a':
			baseInfo = baseInfo + np.power(4, 4-i) * 0
		elif mer5[i]=='C' or mer5[i]=='c':
			baseInfo = baseInfo + np.power(4, 4-i) * 1
		elif mer5[i]=='G' or mer5[i]=='g':
			baseInfo = baseInfo + np.power(4, 4-i) * 2
		elif mer5[i]=='T' or mer5[i]=='t':
			baseInfo = baseInfo + np.power(4, 4-i) * 3

		if i==0:
			subtract = baseInfo

	mgw = vari.MGW[baseInfo][1]
	prot = vari.PROT[baseInfo][1]

	nextBaseInfo = baseInfo - subtract

	return nextBaseInfo, mgw, prot


cpdef edit5merProb(pastMer, oldBase, newBase):
	baseInfo = pastMer
	baseInfo = baseInfo * 4
	subtract = -1

	## newBase
	if newBase=='A' or newBase=='a':
		baseInfo = baseInfo + 0
	elif newBase=='C' or newBase=='c':
		baseInfo = baseInfo + 1
	elif newBase=='G' or newBase=='g':
		baseInfo = baseInfo + 2
	elif newBase=='T' or newBase=='t':
		baseInfo = baseInfo + 3

	baseInfo = int(baseInfo)
	mgw = vari.MGW[baseInfo][1]
	prot = vari.PROT[baseInfo][1]

	## subtract oldBase
	if oldBase=='A' or oldBase=='a':
		subtract = np.power(4, 4) * 0
	elif oldBase=='C' or oldBase=='c':
		subtract = np.power(4, 4) * 1
	elif oldBase=='G' or oldBase=='g':
		subtract = np.power(4, 4) * 2
	elif oldBase=='T' or oldBase=='t':
		subtract = np.power(4, 4) * 3

	nextBaseInfo = baseInfo - subtract

	return nextBaseInfo, mgw, prot


cpdef findComple5merProb(mer5):
	baseInfo = 0
	subtract = -1

	for i in range(5):
		if mer5[i]=='A' or mer5[i]=='a':
			baseInfo = baseInfo + np.power(4, i) * 3
		elif mer5[i]=='C' or mer5[i]=='c':
			baseInfo = baseInfo + np.power(4, i) * 2
		elif mer5[i]=='G' or mer5[i]=='g':
			baseInfo = baseInfo + np.power(4, i) * 1
		elif mer5[i]=='T' or mer5[i]=='t':
			baseInfo = baseInfo + np.power(4, i) * 0

		if i==0:
			subtract = baseInfo

	mgw = vari.MGW[baseInfo][1]
	prot = vari.PROT[baseInfo][1]

	nextBaseInfo = baseInfo - subtract

	return nextBaseInfo, mgw, prot


cpdef editComple5merProb(pastMer, oldBase, newBase):
	baseInfo = pastMer
	baseInfo = int(baseInfo / 4)
	subtract = -1

	# newBase
	if newBase=='A' or newBase=='a':
		baseInfo = baseInfo + np.power(4, 4) * 3
	elif newBase=='C' or newBase=='c':
		baseInfo = baseInfo + np.power(4, 4) * 2
	elif newBase=='G' or newBase=='g':
		baseInfo = baseInfo + np.power(4, 4) * 1
	elif newBase=='T' or newBase=='t':
		baseInfo = baseInfo + np.power(4, 4) * 0

	baseInfo = int(baseInfo)
	mgw = vari.MGW[baseInfo][1]
	prot = vari.PROT[baseInfo][1]

	## subtract oldBase
	if oldBase=='A' or oldBase=='a':
		subtract = 3
	elif oldBase=='C' or oldBase=='c':
		subtract = 2
	elif oldBase=='G' or oldBase=='g':
		subtract = 1
	elif oldBase=='T' or oldBase=='t':
		subtract = 0

	nextBaseInfo = baseInfo - subtract

	return nextBaseInfo, mgw, prot


cpdef findStartGibbs(seq, seqLen):
	gibbs = 0
	subtract = -1

	for i in range(seqLen-1):
		dimer = seq[i:(i+2)].upper()
		if 'N' in dimer:
			gibbs = gibbs + vari.N_GIBBS
		else:
			dimerIdx = 0
			for j in range(2):
				if dimer[j]=='A':
					dimerIdx = dimerIdx + np.power(4, 1-j) * 0
				elif dimer[j]=='C':
					dimerIdx = dimerIdx + np.power(4, 1-j) * 1
				elif dimer[j]=='G':
					dimerIdx = dimerIdx + np.power(4, 1-j) * 2
				elif dimer[j]=='T':
					dimerIdx = dimerIdx + np.power(4, 1-j) * 3
			gibbs = gibbs + vari.GIBBS[dimerIdx][1]

		if i==0:
			subtract = gibbs

	startGibbs = gibbs - subtract

	return startGibbs, gibbs


cpdef editStartGibbs(oldDimer, newDimer, pastStartGibbs):
	gibbs = pastStartGibbs
	subtract = -1

	# newDimer
	if 'N' in newDimer:
		gibbs = gibbs + vari.N_GIBBS
	else:
		dimerIdx = 0
		for j in range(2):
			if newDimer[j]=='A':
				dimerIdx = dimerIdx + np.power(4, 1-j) * 0
			elif newDimer[j]=='C':
				dimerIdx = dimerIdx + np.power(4, 1-j) * 1
			elif newDimer[j]=='G':
				dimerIdx = dimerIdx + np.power(4, 1-j) * 2
			elif newDimer[j]=='T':
				dimerIdx = dimerIdx + np.power(4, 1-j) * 3
		gibbs = gibbs + vari.GIBBS[dimerIdx][1]

	## remove the old dimer for the next iteration
	if 'N' in oldDimer:
		subtract = vari.N_GIBBS
	else:
		dimerIdx = 0
		for j in range(2):
			if oldDimer[j]=='A':
				dimerIdx = dimerIdx + np.power(4, 1-j) * 0
			elif oldDimer[j]=='C':
				dimerIdx = dimerIdx + np.power(4, 1-j) * 1
			elif oldDimer[j]=='G':
				dimerIdx = dimerIdx + np.power(4, 1-j) * 2
			elif oldDimer[j]=='T':
				dimerIdx = dimerIdx + np.power(4, 1-j) * 3
		subtract = vari.GIBBS[dimerIdx][1]

	startGibbs = gibbs - subtract

	return startGibbs, gibbs


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


cpdef correctReadCount(covariFileName, chromo, analysisStart, analysisEnd, lastBin, nBins):
	warnings.filterwarnings('ignore', r'All-NaN slice encountered')
	warnings.filterwarnings('ignore', r'Mean of empty slice')

	regionStart = analysisStart
	if not lastBin:
		regionEnd = analysisStart + vari.BINSIZE * nBins
		lastBinStart = None
		lastBinEnd = None
	else: # lastBin
		regionEnd = analysisStart + vari.BINSIZE * (nBins-1)
		lastBinStart = regionEnd
		lastBinEnd = analysisEnd

	###### GET POSITIONS WHERE THE NUMBER OF FRAGMENTS > FILTERVALUE
	selectedIdx, highRCIdx, starts = selectIdx(chromo, regionStart, regionEnd, vari.CTRLBW_NAMES, vari.EXPBW_NAMES, lastBinStart, lastBinEnd, nBins, vari.BINSIZE, vari.HIGHRC, vari.FILTERVALUE)

	## OUTPUT FILES
	subfinalCtrlNames = [None] * vari.CTRLBW_NUM
	subfinalExperiNames = [None] * vari.EXPBW_NUM

	if len(selectedIdx) == 0:
		return [subfinalCtrlNames, subfinalExperiNames, chromo]

	f = h5py.File(covariFileName, "r")

	for rep in range(vari.CTRLBW_NUM):
		## observed read counts
		bw = pyBigWig.open(vari.CTRLBW_NAMES[rep])
		if lastBin:
			rcArr = np.array(bw.stats(chromo, regionStart, regionEnd, type="mean", nBins=(nBins-1)))
			rcArr[np.where(rcArr==None)] = float(0)
			rcArr = rcArr.tolist()

			last_value = bw.stats(chromo, lastBinStart, lastBinEnd, type="mean", nBins=1)[0]
			if last_value is None:
				last_value = float(0)
			rcArr.extend([last_value])
			rcArr = np.array(rcArr)
		else:
			rcArr = np.array(bw.stats(chromo, regionStart, regionEnd, type="mean", nBins=nBins))
			rcArr[np.where(rcArr==None)] = float(0)
		rcArr = rcArr / vari.CTRLSCALER[rep]
		bw.close()

		## predicted read counts
		prdvals = np.exp(np.sum(f['X'][0:]* vari.COEFCTRL[rep, 1:], axis=1) + vari.COEFCTRL[rep, 0])
		prdvals[highRCIdx] = np.exp(np.sum(f['X'][0:][highRCIdx] * vari.COEFCTRL_HIGHRC[rep, 1:], axis=1) + vari.COEFCTRL_HIGHRC[rep, 0])

		rcArr = rcArr - prdvals
		rcArr = rcArr[selectedIdx]

		idx = np.where( (rcArr < np.finfo(np.float32).min) | (rcArr > np.finfo(np.float32).max))
		tempStarts = np.delete(starts, idx)
		rcArr = np.delete(rcArr, idx)
		if len(rcArr) > 0:
			with tempfile.NamedTemporaryFile(mode="w+t", suffix=".bed", dir=vari.OUTPUT_DIR, delete=False) as subfinalCtrlFile:
				subfinalCtrlNames[rep] = subfinalCtrlFile.name
				writeBedFile(subfinalCtrlFile, tempStarts, rcArr, analysisEnd, vari.BINSIZE)

	for rep in range(vari.EXPBW_NUM):
		## observed read counts
		bw = pyBigWig.open(vari.EXPBW_NAMES[rep])
		if lastBin:
			rcArr = np.array(bw.stats(chromo, regionStart, regionEnd, type="mean", nBins=(nBins-1)))
			rcArr[np.where(rcArr==None)] = float(0)
			rcArr = rcArr.tolist()

			last_value = bw.stats(chromo, lastBinStart, lastBinEnd, type="mean", nBins=1)[0]
			if last_value is None:
				last_value = float(0)
			rcArr.extend([last_value])
			rcArr = np.array(rcArr)
		else:
			rcArr = np.array(bw.stats(chromo, regionStart, regionEnd, type="mean", nBins=nBins))
			rcArr[np.where(rcArr==None)] = float(0)
		rcArr = rcArr / vari.EXPSCALER[rep]
		bw.close()


		## predicted read counts
		prdvals = np.exp(np.sum(f['X'][0:]* vari.COEFEXP[rep, 1:], axis=1) + vari.COEFEXP[rep, 0])
		prdvals[highRCIdx] = np.exp(np.sum(f['X'][0:][highRCIdx] * vari.COEFEXP_HIGHRC[rep, 1:], axis=1) + vari.COEFEXP_HIGHRC[rep, 0])

		rcArr = rcArr - prdvals
		rcArr = rcArr[selectedIdx]

		idx = np.where( (rcArr < np.finfo(np.float32).min) | (rcArr > np.finfo(np.float32).max))
		tempStarts = np.delete(starts, idx)
		rcArr = np.delete(rcArr, idx)
		if len(rcArr) > 0:
			with tempfile.NamedTemporaryFile(mode="w+t", suffix=".bed", dir=vari.OUTPUT_DIR, delete=False) as subfinalExperiFile:
				subfinalExperiNames[rep] = subfinalExperiFile.name
				writeBedFile(subfinalExperiFile, tempStarts, rcArr, analysisEnd, vari.BINSIZE)

	f.close()
	os.remove(covariFileName)

	return [subfinalCtrlNames, subfinalExperiNames, chromo]


cpdef selectIdx(chromo, regionStart, regionEnd, ctrlBWNames, experiBWNames, lastBinStart, lastBinEnd, nBins, binSize, highRC, minFragFilterValue):
	meanMinFragFilterValue = int(np.round(minFragFilterValue / (len(ctrlBWNames) + len(experiBWNames))))

	for rep, bwName in enumerate(ctrlBWNames):
		with pyBigWig.open(bwName) as bw:
			if lastBinStart is not None:
				temp = np.array(bw.stats(chromo, regionStart, regionEnd, type="mean", nBins=(nBins-1)))
				temp[np.where(temp == None)] = 0.0
				temp = temp.tolist()

				last_value = bw.stats(chromo, lastBinStart, lastBinEnd, type="mean", nBins=1)[0]
				if last_value is None:
					last_value = 0.0
				temp.extend([last_value])
				temp = np.array(temp)
			else:
				temp = np.array(bw.stats(chromo, regionStart, regionEnd, type="mean", nBins=nBins))
				temp[np.where(temp == None)] = 0.0

			if rep == 0:
				rc_sum = temp
				highReadCountIdx = np.where(temp > highRC)[0]
				overMeanReadCountIdx = np.where(temp >= meanMinFragFilterValue)[0]

			else:
				rc_sum += temp
				overMeanReadCountIdx_temp = np.where(temp >= meanMinFragFilterValue)[0]
				overMeanReadCountIdx = np.intersect1d(overMeanReadCountIdx, overMeanReadCountIdx_temp)

	for bwName in experiBWNames:
		with pyBigWig.open(bwName) as bw:

			if lastBinStart is not None:
				temp = np.array(bw.stats(chromo, regionStart, regionEnd, type="mean", nBins=(nBins-1)))
				temp[np.where(temp == None)] = 0.0
				temp = temp.tolist()

				last_value = bw.stats(chromo, lastBinStart, lastBinEnd, type="mean", nBins=1)[0]
				if last_value is None:
					last_value = 0.0
				temp.extend([last_value])
				temp = np.array(temp)
			else:
				temp = np.array(bw.stats(chromo, regionStart, regionEnd, type="mean", nBins=nBins))
				temp[np.where(temp == None)] = 0.0

			rc_sum += temp

			overMeanReadCountIdx_temp = np.where(temp >= meanMinFragFilterValue)[0]
			overMeanReadCountIdx = np.intersect1d(overMeanReadCountIdx, overMeanReadCountIdx_temp)

	idx = np.where(rc_sum > minFragFilterValue)[0].tolist()
	idx = np.intersect1d(idx, overMeanReadCountIdx)

	highReadCountIdx = np.intersect1d(highReadCountIdx, idx)

	if len(idx) == 0:
		return np.array([]), np.array([]), np.array([])

	if lastBinStart is not None:
		starts = np.arange(regionStart, regionEnd, binSize)
		starts = starts.tolist()
		starts.extend([regionEnd])
		starts = np.array(starts)
	else:
		starts = np.arange(regionStart, regionEnd, binSize)


	starts = starts[idx]

	return idx, highReadCountIdx, starts
