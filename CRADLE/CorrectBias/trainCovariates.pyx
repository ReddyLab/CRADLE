# cython: language_level=3

import math
import os
import tempfile
import warnings
import h5py
import numpy as np
import py2bit
import pyBigWig

import CRADLE.CorrectBias.covariateUtils as cu

from CRADLE.CorrectBias import vari
from CRADLE.correctbiasutils.cython import writeBedFile
from CRADLE.correctbiasutils import vari as commonVari

cpdef calculateTrainCovariates(region):
	### supress numpy nan-error message
	warnings.filterwarnings('ignore', r'All-NaN slice encountered')
	warnings.filterwarnings('ignore', r'Mean of empty slice')

	chromo = region.chromo
	analysisStart = region.start  # Genomic coordinates(starts from 1)
	analysisEnd = region.end  # not included

	binStart = int((analysisStart + analysisStart + vari.BINSIZE) / float(2))
	binEnd = int((analysisEnd - vari.BINSIZE + analysisEnd) / float(2))

	###### CALCULATE INDEX VARIABLE
	fragStart = binStart + 1 - vari.FRAGLEN
	fragEnd = binEnd + vari.FRAGLEN  # not included
	shearStart = fragStart - 2
	shearEnd = fragEnd + 2 # not included

	genome = py2bit.open(vari.GENOME)
	chromoEnd = int(genome.chroms(chromo))

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
	sequence = genome.sequence(chromo, (shearStart-1), (shearEnd-1))
	genome.close()

	##### OPEN BIAS FILES
	if vari.MAP == 1:
		mapFile = pyBigWig.open(vari.MAPFILE)
		mapValue = np.array(mapFile.values(chromo, fragStart, fragEnd))

		mapValue[np.where(mapValue == 0)] = np.nan
		mapValue = np.log(mapValue)
		mapValue[np.where(np.isnan(mapValue))] = float(-6)

		mapValueView = cu.memoryView(mapValue)
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
		gquadView = cu.memoryView(gquadValue)

		del gquadFile, gquadValue


	##### INDEX IN 'sequence'
	startIdx = 2  # index in the genome sequence file (Included in the range)
	endIdx = (fragEnd - vari.FRAGLEN) - shearStart + 1   # index in the genome sequence file (Not included in the range)

	##### INITIALIZE VARIABLES
	if vari.SHEAR == 1:
		pastMer1 = -1
		pastMer2 = -1
	if vari.PCR == 1:
		pastStartGibbs = -1

	resultStartIdx = -1
	resultEndIdx = -1

	##### STORE COVARI RESULTS
	covariFileTemp = tempfile.NamedTemporaryFile(suffix=".hdf5", dir=commonVari.OUTPUT_DIR, delete=True)
	covariFileName = covariFileTemp.name
	covariFileTemp.close()

	f = h5py.File(covariFileName, "w")
	covariFile = f.create_dataset("covari", (nBins, vari.COVARI_NUM), dtype='f', compression="gzip")

	for idx in range(startIdx, endIdx):
		covariIdx = [0] * vari.COVARI_NUM
		covariIdxPtr = 0

		if vari.SHEAR == 1:
			###  mer1
			mer1 = sequence[(idx-2):(idx+3)]
			if 'N' in mer1:
				pastMer1 = -1
				mgwIdx = vari.N_MGW
				protIdx = vari.N_PROT
			else:
				if pastMer1 == -1: # there is no information on pastMer1
					pastMer1, mgwIdx, protIdx = cu.find5merProb(mer1)
				else:
					pastMer1, mgwIdx, protIdx = cu.edit5merProb(pastMer1, mer1[0], mer1[4])

			##  mer2
			fragEndIdx = idx + vari.FRAGLEN
			mer2 = sequence[(fragEndIdx-3):(fragEndIdx+2)]
			if 'N' in mer2:
				pastMer2 = -1
				mgwIdx = mgwIdx + vari.N_MGW
				protIdx = protIdx + vari.N_PROT
			else:
				if pastMer2 == -1:
					pastMer2, add1, add2 = cu.findComple5merProb(mer2)
				else:
					pastMer2, add1, add2 = cu.editComple5merProb(pastMer2, mer2[0], mer2[4])
				mgwIdx = mgwIdx + add1
				protIdx = protIdx + add2

			covariIdx[covariIdxPtr] = mgwIdx
			covariIdx[covariIdxPtr+1] = protIdx
			covariIdxPtr = covariIdxPtr + 2


		if vari.PCR == 1:
			sequenceIdx = sequence[idx:(idx+vari.FRAGLEN)]
			if pastStartGibbs == -1:
				startGibbs, gibbs = cu.findStartGibbs(sequenceIdx, vari.FRAGLEN)
			else:
				oldDimer = sequenceIdx[0:2].upper()
				newDimer = sequenceIdx[(vari.FRAGLEN-2):vari.FRAGLEN].upper()
				startGibbs, gibbs = cu.editStartGibbs(oldDimer, newDimer, pastStartGibbs)

			annealIdx, denatureIdx = cu.convertGibbs(gibbs)

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
	for rep in range(commonVari.SAMPLE_NUM):
		rcFileTemp = tempfile.NamedTemporaryFile(suffix=".hdf5", dir=commonVari.OUTPUT_DIR, delete=True)
		rcFileName = rcFileTemp.name
		rcFileTemp.close()

		f = h5py.File(rcFileName, "w")
		rcFile = f.create_dataset("Y", (nBins, ), dtype='f', compression="gzip")


		returnLine.extend([ rcFileName ])

		if rep < commonVari.CTRLBW_NUM:
			bw = pyBigWig.open(commonVari.CTRLBW_NAMES[rep])
		else:
			bw = pyBigWig.open(commonVari.EXPBW_NAMES[rep-commonVari.CTRLBW_NUM])

		temp = np.array(bw.values(chromo, analysisStart, analysisEnd))
		numPos = len(temp)


		for binIdx in range(nBins):
			if rep < commonVari.CTRLBW_NUM:
				if (binIdx + vari.BINSIZE) >= numPos:
					rc = np.nanmean(temp[binIdx:]) / float(commonVari.CTRLSCALER[rep])
				else:
					rc = np.nanmean(temp[binIdx:(binIdx + vari.BINSIZE)]) / float(commonVari.CTRLSCALER[rep])
			else:
				if (binIdx + vari.BINSIZE) >= numPos:
					rc = np.nanmean(temp[binIdx:]) / float(commonVari.EXPSCALER[rep-commonVari.CTRLBW_NUM])
				else:
					rc = np.nanmean(temp[binIdx:(binIdx + vari.BINSIZE)]) / float(commonVari.EXPSCALER[rep-commonVari.CTRLBW_NUM])

			if np.isnan(rc):
				rc = float(0)

			rcFile[binIdx] = rc


		f.close()
		bw.close()

	return returnLine

cpdef makeMatrixContinuousFragTrain(binStart, binEnd, nBins):

	result = np.zeros(((vari.FRAGLEN+1), (vari.COVARI_NUM+1)), dtype=np.float64)
	for i in range(vari.FRAGLEN+1):
		pos = binStart + i

		if pos > binEnd:
			result[i, 0] = np.nan
		else:
			result[i, 0] = pos

	return result
