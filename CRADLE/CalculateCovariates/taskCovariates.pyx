# cython: language_level=3

import array
import os.path
import pickle
import tempfile

import h5py
import numpy as np
import py2bit
import pyBigWig

cimport cython
from cpython cimport array

import CRADLE.CalculateCovariates.covariateUtils as cu

from CRADLE.CalculateCovariates import vari
from CRADLE.correctbiasutils import vari as commonVari


cdef int BINSIZE = 1


cpdef calculateBoundaries(chromoEnd, analysisStart, analysisEnd, binStart, binEnd, nBins, fragLen):
	fragStart = binStart + 1 - fragLen
	fragEnd = binEnd + fragLen  # not included
	shearStart = fragStart - 2
	shearEnd = fragEnd + 2 # not included

	if shearStart < 1:
		shearStart = 1
		fragStart = 3
		binStart = max(binStart, fragStart)
		analysisStart = max(analysisStart, fragStart)

		###### EDIT BINSTART/ BINEND
		nBins = analysisEnd - analysisStart
		binEnd = binStart + (nBins - 1)

		fragEnd = binEnd + fragLen
		shearEnd = fragEnd + 2

	if shearEnd > chromoEnd:
		analysisEndModified = min(analysisEnd, chromoEnd - 2)  # not included

		if analysisEndModified == analysisEnd:
			shearEnd = chromoEnd
			fragEnd = shearEnd - 2
		else:
			analysisEnd = analysisEndModified

			###### EDIT BINSTART/ BINEND
			nBins = analysisEnd - analysisStart
			binEnd = binStart + (nBins - 1)

			fragEnd = binEnd + fragLen
			shearEnd = fragEnd + 2

			if shearEnd > chromoEnd:
				shearEnd = chromoEnd
				fragEnd = shearEnd - 2

	return analysisStart, analysisEnd, fragStart, fragEnd, shearStart, shearEnd, binStart, binEnd, nBins


cpdef mapValues(mapFile, chromo, fragStart, fragEnd):
	mapValue = np.array(mapFile.values(chromo, fragStart, fragEnd))

	mapValue[np.where(mapValue == 0)] = np.nan
	mapValue = np.log(mapValue)
	mapValue[np.where(np.isnan(mapValue))] = -6.0

	cdef double [:] mapValueView = mapValue

	return mapValueView


cpdef gquadValues(gquadFiles, chromo, fragStart, fragEnd, gquadMax):
	gquadValue = [0] * len(gquadFiles)

	for i, gquadFile in enumerate(gquadFiles):
		gquadValue[i] = gquadFile.values(chromo, fragStart, fragEnd)

	gquadValue = np.array(gquadValue)
	gquadValue = np.nanmax(gquadValue, axis=0)
	gquadValue[np.where(gquadValue == 0)] = np.nan
	gquadValue = np.log(gquadValue / gquadMax)

	gquadValue[np.where(np.isnan(gquadValue))] = -5.0
	cdef double[:] gquadValueView = gquadValue

	return gquadValueView


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef fragCovariates(int idx, pastMer1, pastMer2, int pastStartGibbs, sequence, mapValues, gquadValues, int fragLen, globalVars):
	cdef int shear = globalVars["shear"]
	cdef int pcr = globalVars["pcr"]
	cdef int map = globalVars["map"]
	cdef int gquad = globalVars["gquad"]
	cdef int covariNum = globalVars["covariNum"]
	cdef double n_mgw = globalVars["n_mgw"]
	cdef double n_prot = globalVars["n_prot"]
	cdef int kmer = globalVars["kmer"]

	cdef int n_gibbs = globalVars["n_gibbs"]
	cdef double gibbs = 0
	cdef double subtract = -1
	cdef double startGibbs = 0
	cdef int i
	cdef char d1, d2
	cdef char[:] seq = array.array("b", sequence)

	covariates = np.zeros(covariNum)
	covariIdxPtr = 0

	if shear == 1:
		###  mer1
		mer1 = sequence[(idx-2):(idx+3)]
		if b'N' in mer1:
			pastMer1 = -1
			mgwIdx = n_mgw
			protIdx = n_prot
		else:
			if pastMer1 == -1: # there is no information on pastMer1
				pastMer1, mgwIdx, protIdx = cu.find5merProb(mer1)
			else:
				pastMer1, mgwIdx, protIdx = cu.edit5merProb(pastMer1, mer1[0], mer1[4])

		###  mer2
		fragEndIdx = idx + fragLen
		mer2 = sequence[(fragEndIdx-3):(fragEndIdx+2)]
		if b'N' in mer2:
			pastMer2 = -1
			mgwIdx = mgwIdx + n_mgw
			protIdx = protIdx + n_prot
		else:
			if pastMer2 == -1:
				pastMer2, add1, add2 = cu.findComple5merProb(mer2)
			else:
				pastMer2, add1, add2 = cu.editComple5merProb(pastMer2, mer2[0], mer2[4])
			mgwIdx = mgwIdx + add1
			protIdx = protIdx + add2

		covariates[covariIdxPtr] = mgwIdx
		covariates[covariIdxPtr+1] = protIdx
	covariIdxPtr += 2

	if pcr == 1:
		if pastStartGibbs == -1:
			# This huge unweildy nonsense replaces a call to cu.findStartGibbs.
			# findStartGibbs was a huge bottleneck to speed so this code inlines
			# and unravels it a bit. Additionally, sequence has been changed elsewhere
			# to an array of bytes (instead of unicode characters). This allows us to
			# treat it like an array of chars for fast tests and access.
			#
			# On the benchmark I'm using it brings the time to calculate covariates
			# down from 12 seconds to 2 seconds. No other change comes close to being as important.
			if seq[idx] == 78 or seq[idx + 1] == 78:
				gibbs += n_gibbs
			else:
				d1 = seq[idx]
				d2 = seq[idx + 1]
				if d1 == 65: # A
					if d2 == 65:
						gibbs += -1.04
					elif d2 == 67:
						gibbs += -2.04
					elif d2 == 71:
						gibbs += -1.29
					elif d2 == 84:
						gibbs += -1.27
				elif d1 == 67: # C
					if d2 == 65:
						gibbs += -0.78
					elif d2 == 67:
						gibbs += -1.97
					elif d2 == 71:
						gibbs += -1.44
					elif d2 == 84:
						gibbs += -1.29
				elif d1 == 71: # G
					if d2 == 65:
						gibbs += -1.66
					elif d2 == 67:
						gibbs += -2.7
					elif d2 == 71:
						gibbs += -1.97
					elif d2 == 84:
						gibbs += -2.04
				elif d1 == 84: # T
					if d2 == 65:
						gibbs += -0.12
					elif d2 == 67:
						gibbs += -1.66
					elif d2 == 71:
						gibbs += -0.78
					elif d2 == 84:
						gibbs += -1.04

			subtract = gibbs

			for i in range(1, fragLen - 1):
				if seq[idx + i] == 78 or seq[idx + i + 1] == 78:
					gibbs += n_gibbs
				else:
					d1 = seq[idx]
					d2 = seq[idx + 1]
					if d1 == 65: # A
						if d2 == 65:
							gibbs += -1.04
						elif d2 == 67:
							gibbs += -2.04
						elif d2 == 71:
							gibbs += -1.29
						elif d2 == 84:
							gibbs += -1.27
					elif d1 == 67: # C
						if d2 == 65:
							gibbs += -0.78
						elif d2 == 67:
							gibbs += -1.97
						elif d2 == 71:
							gibbs += -1.44
						elif d2 == 84:
							gibbs += -1.29
					elif d1 == 71: # G
						if d2 == 65:
							gibbs += -1.66
						elif d2 == 67:
							gibbs += -2.7
						elif d2 == 71:
							gibbs += -1.97
						elif d2 == 84:
							gibbs += -2.04
					elif d1 == 84: # T
						if d2 == 65:
							gibbs += -0.12
						elif d2 == 67:
							gibbs += -1.66
						elif d2 == 71:
							gibbs += -0.78
						elif d2 == 84:
							gibbs += -1.04

			startGibbs = gibbs - subtract
		else:
			sequenceIdx = sequence[idx:idx + fragLen]
			oldDimer = sequenceIdx[0:2]
			newDimer = sequenceIdx[(fragLen-2):fragLen]
			startGibbs, gibbs = cu.editStartGibbs(oldDimer, newDimer, pastStartGibbs, globalVars["n_gibbs"])

		annealIdx, denatureIdx = cu.convertGibbs(gibbs, globalVars["entropy"], globalVars["fragLen"], globalVars["min_tm"], globalVars["max_tm"], globalVars["para1"], globalVars["para2"])

		covariates[covariIdxPtr] = annealIdx
		covariates[covariIdxPtr+1] = denatureIdx
	covariIdxPtr += 2

	if map == 1:
		map1 = mapValues[(idx-2)]
		map2 = mapValues[(idx+fragLen-2-kmer)]
		mapIdx = map1 + map2

		covariates[covariIdxPtr] = mapIdx
	covariIdxPtr += 1

	if gquad == 1:
		covariates[covariIdxPtr] = np.nanmax(np.asarray(gquadValues[(idx-2):(idx+fragLen-2)]))

	return covariates, pastMer1, pastMer2, pastStartGibbs


cpdef calculateContinuousFrag(sequence, mapValueView, gquadValueView, result, shearStart, fragEnd, binStart, binEnd, analysisLength, shear, pcr, covariNum, fragLen, globalVars):
	fraglenPlusOne = fragLen + 1

	##### INITIALIZE VARIABLES
	if shear == 1:
		pastMer1 = -1
		pastMer2 = -1

	if pcr == 1:
		pastStartGibbs = -1

	resultStartIdx = -1
	resultEndIdx = -1

	covariDataSet = np.zeros((analysisLength, covariNum))

	##### INDEX IN 'sequence'
	startIdx = 2  # index in the genome sequence file (Included in the range)
	endIdx = (fragEnd - fragLen) - shearStart + 1   # index in the genome sequence file (Not included in the range)

	for idx in range(startIdx, endIdx):
		covariates, pastMer1, pastMer2, pastStartGibbs = fragCovariates(idx, pastMer1, pastMer2, pastStartGibbs, sequence, mapValueView, gquadValueView, fragLen, globalVars)

		### DETERMINE WHICH ROWS TO EDIT IN RESULT MATRIX
		thisFragStart = idx + shearStart
		thisFragEnd = thisFragStart + fragLen

		if resultStartIdx == -1:
			resultStartIdx = 0
			resultEndIdx = 1 # not included
			maxBinPos = binStart + fragLen
			numPoppedPos = 0
		else:
			while result[resultStartIdx, 0] < thisFragStart:
				## pop the element
				covariDataSet[numPoppedPos, :] = result[resultStartIdx, 1:]
				result[resultStartIdx, 1:] = 0.0

				numPoppedPos += 1
				if maxBinPos >= binEnd:
					result[resultStartIdx, 0] = np.nan
				else:
					result[resultStartIdx, 0] = maxBinPos + 1
					maxBinPos += 1

				resultStartIdx += 1
				if resultStartIdx > fragLen:
					resultStartIdx -= fraglenPlusOne

		while not np.isnan(result[resultEndIdx, 0]) and result[resultEndIdx, 0] < thisFragEnd:
			resultEndIdx += 1
			if resultEndIdx > fragLen:
				resultEndIdx -= fraglenPlusOne

		if resultEndIdx < resultStartIdx:
			result[resultStartIdx:fraglenPlusOne, 1:] += covariates
			result[0:resultEndIdx, 1:] += covariates
		else:
			result[resultStartIdx:resultEndIdx, 1:] += covariates

		if idx == (endIdx-1): # the last fragment
			### pop the rest of positions that are not np.nan
			if resultEndIdx < resultStartIdx:
				posLen = fraglenPlusOne - resultStartIdx
				end = numPoppedPos + posLen
				covariDataSet[numPoppedPos:end, :] = result[resultStartIdx:fraglenPlusOne, 1:]
				numPoppedPos = end

				posLen = resultEndIdx
				end = numPoppedPos + posLen
				covariDataSet[numPoppedPos:end, :] = result[0:resultEndIdx, 1:]
				numPoppedPos = end
			else:
				posLen = resultEndIdx - resultStartIdx
				end = numPoppedPos + posLen
				covariDataSet[numPoppedPos:end, :] = result[resultStartIdx:resultEndIdx, 1:]
				numPoppedPos = end

	return covariDataSet


cpdef calculateDiscreteFrag(chromoEnd, sequence, mapValueView, gquadValueView, shearStart, binStart, binEnd, nBins, shear, pcr, covariNum, fragLen, globalVars):
	covariDataSet = np.zeros((nBins, covariNum))
	for resultIdx in range(nBins): # for each bin
		if resultIdx == (nBins-1):
			pos = binEnd
		else:
			pos = binStart + resultIdx

		thisBinFirstFragStart = pos + 1 - fragLen
		thisBinLastFragStart = pos

		if thisBinFirstFragStart < 3:
			thisBinFirstFragStart = 3
		if (thisBinLastFragStart + fragLen) > (chromoEnd - 2):
			thisBinLastFragStart = chromoEnd - 2 - fragLen

		thisBinNumFrag = thisBinLastFragStart - thisBinFirstFragStart + 1

		thisBinFirstFragStartIdx = thisBinFirstFragStart - shearStart

		##### INITIALIZE VARIABLES
		if shear == 1:
			pastMer1 = -1
			pastMer2 = -1
		if pcr == 1:
			pastStartGibbs = -1

		for binFragIdx in range(thisBinFirstFragStartIdx, thisBinFirstFragStartIdx + thisBinNumFrag):
			covariates, pastMer1, pastMer2, pastStartGibbs = fragCovariates(binFragIdx, pastMer1, pastMer2, pastStartGibbs, sequence, mapValueView, gquadValueView, fragLen, globalVars)

			covariDataSet[resultIdx, :] += covariates

		return covariDataSet


cpdef calculateTaskCovariates(regions, globalVars):
	cdef int fragLen = globalVars["fragLen"]
	cdef int shear = globalVars["shear"]
	cdef int pcr = globalVars["pcr"]
	cdef int map = globalVars["map"]
	cdef int gquad = globalVars["gquad"]
	cdef int covariNum = globalVars["covariNum"]
	cdef double gquadMax = globalVars["gquadMax"]

	outputDir = globalVars["outputDir"]

	genome = py2bit.open(globalVars["genome"])
	mapFile = pyBigWig.open(globalVars["mapFile"])
	gquadFiles = [0] * len(globalVars["gquadFile"])
	for i, file in enumerate(globalVars["gquadFile"]):
		gquadFiles[i] = pyBigWig.open(file)

	outputRegions = []
	chromoEnds = {}
	##### CALCULATE COVARIATE VALUES
	for chromo, analysisStart, analysisEnd in regions:
		chromoEnd = chromoEnds.get(chromo, -1)
		if chromoEnd == -1:
			chromoEnd = int(genome.chroms(chromo))
			chromoEnds[chromo] = chromoEnd
		continuousFrag = False

		#### DECIDE IF 'calculateContinuousFrag' or 'calculateDiscreteFrag'
		#### TODO: What is the logic here? why do these things determine continuousFrag for discreteFrag?
		if (analysisStart + BINSIZE) >= analysisEnd:
			firstBinPos = (analysisStart + analysisEnd) // 2
			lastBinPos = firstBinPos
			nBins = 1
			continuousFrag = True
		else:
			firstBinPos = (2 * analysisStart + BINSIZE) // 2
			if (analysisStart + 2) > analysisEnd:
				secondBinPos = (analysisStart + BINSIZE + analysisEnd) // 2
				lastBinPos = secondBinPos
				nBins = 2
			else:
				secondBinPos = (2 * analysisStart + 3) // 2
				nBins = analysisEnd - analysisStart
				lastBinPos = firstBinPos + (nBins - 1) ## should be included in the analysis

			if secondBinPos - firstBinPos <= fragLen:
				continuousFrag = True

		###### CALCULATE INDEX VARIABLE
		analysisStart, analysisEnd, fragStart, fragEnd, shearStart, shearEnd, binStart, binEnd, nBins = calculateBoundaries(chromoEnd, analysisStart, analysisEnd, firstBinPos, lastBinPos, nBins, fragLen)

		###### GET SEQUENCE
		sequence = genome.sequence(chromo, (shearStart-1), (shearEnd-1)).upper().encode("utf-8")

		##### GET BIASES INFO FROM FILES
		if map == 1:
			mapValueView = mapValues(mapFile, chromo, fragStart, fragEnd)

		if gquad == 1:
			gquadValueView = gquadValues(gquadFiles, chromo, fragStart, fragEnd, gquadMax)

		if continuousFrag:
			###### GENERATE A RESULT MATRIX
			result = makeMatrixContinuousFrag(binStart, binEnd, nBins, fragLen, covariNum)

			covariDataSet = calculateContinuousFrag(sequence, mapValueView, gquadValueView, result, shearStart, fragEnd, binStart, binEnd, analysisEnd - analysisStart, shear, pcr, covariNum, fragLen, globalVars)
		else:
			covariDataSet = calculateDiscreteFrag(chromoEnd, sequence, mapValueView, gquadValueView, shearStart, binStart, binEnd, nBins, shear, pcr, covariNum, fragLen, globalVars)

		with open(os.path.join(outputDir, f"{chromo}_{analysisStart}_{analysisEnd}.pkl"), "wb") as file:
			pickle.dump(covariDataSet, file)

		outputRegions.append((chromo, analysisStart, analysisEnd))

	for file in gquadFiles:
		file.close()
	mapFile.close()
	genome.close()

	return outputRegions


cpdef makeMatrixContinuousFrag(binStart, binEnd, nBins, fragLen, covariNum):
	result = np.zeros(((fragLen+1), (covariNum+1)), dtype=np.float64)
	for i in range(fragLen+1):
		pos = binStart + i

		if pos > binEnd:
			result[i, 0] = np.nan
		else:
			result[i, 0] = pos

	if nBins == (fragLen+1):
		result[fragLen, 0] = binEnd

	return result
