# cython: language_level=3

import tempfile
import h5py
import numpy as np
import py2bit
import pyBigWig

import CRADLE.CalculateCovariates.covariateUtils as cu

from CRADLE.CalculateCovariates import vari
from CRADLE.correctbiasutils import vari as commonVari



# The covariate values stored in the HDF files start at index 0 (0-index, obviously)
# The lowest start point for an analysis region is 3 (1-indexed), so we need to subtract
# 3 from the analysis start and end points to match them up with correct covariate values
# in the HDF files.
COVARIATE_FILE_INDEX_OFFSET = 3

BINSIZE = 1


cpdef calculateBoundaries(chromoEnd, analysisStart, analysisEnd, binStart, binEnd, nBins):
	fragStart = binStart + 1 - vari.FRAGLEN
	fragEnd = binEnd + vari.FRAGLEN  # not included
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
			nBins = analysisEnd - analysisStart
			binEnd = binStart + (nBins - 1)

			fragEnd = binEnd + vari.FRAGLEN
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

	mapValueView = cu.memoryView(mapValue)

	return mapValueView


cpdef gquadValues(gquadFiles, chromo, fragStart, fragEnd):
	gquadValue = [0] * len(gquadFiles)

	for i, gquadFile in enumerate(gquadFiles):
		gquadValue[i] = gquadFile.values(chromo, fragStart, fragEnd)

	gquadValue = np.array(gquadValue)
	gquadValue = np.nanmax(gquadValue, axis=0)
	gquadValue[np.where(gquadValue == 0)] = np.nan
	gquadValue = np.log(gquadValue / vari.GQAUD_MAX)

	gquadValue[np.where(np.isnan(gquadValue))] = -5.0
	gquadValueView = cu.memoryView(gquadValue)

	return gquadValueView


cpdef fragCovariates(idx, pastMer1, pastMer2, pastStartGibbs, sequence, mapValues, gquadValues):
	covariates = np.zeros(vari.COVARI_NUM)
	covariIdxPtr = 0

	if vari.SHEAR == 1:
		###  mer1
		mer1 = sequence[(idx-2):(idx+3)].upper()
		if 'N' in mer1:
			pastMer1 = -1
			mgwIdx = vari.N_MGW
			protIdx = vari.N_PROT
		else:
			if pastMer1 == -1: # there is no information on pastMer1
				pastMer1, mgwIdx, protIdx = cu.find5merProb(mer1)
			else:
				pastMer1, mgwIdx, protIdx = cu.edit5merProb(pastMer1, mer1[0], mer1[4])

		###  mer2
		fragEndIdx = idx + vari.FRAGLEN
		mer2 = sequence[(fragEndIdx-3):(fragEndIdx+2)].upper()
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

		covariates[covariIdxPtr] = mgwIdx
		covariates[covariIdxPtr+1] = protIdx
	covariIdxPtr += 2

	if vari.PCR == 1:
		sequenceIdx = sequence[idx:(idx+vari.FRAGLEN)]
		if pastStartGibbs == -1:
			startGibbs, gibbs = cu.findStartGibbs(sequenceIdx, vari.FRAGLEN)
		else:
			oldDimer = sequenceIdx[0:2].upper()
			newDimer = sequenceIdx[(vari.FRAGLEN-2):vari.FRAGLEN].upper()
			startGibbs, gibbs = cu.editStartGibbs(oldDimer, newDimer, pastStartGibbs)

		annealIdx, denatureIdx = cu.convertGibbs(gibbs)

		covariates[covariIdxPtr] = annealIdx
		covariates[covariIdxPtr+1] = denatureIdx
	covariIdxPtr += 2

	if vari.MAP == 1:
		map1 = mapValues[(idx-2)]
		map2 = mapValues[(idx+vari.FRAGLEN-2-vari.KMER)]
		mapIdx = map1 + map2

		covariates[covariIdxPtr] = mapIdx
	covariIdxPtr += 1

	if vari.GQUAD == 1:
		covariates[covariIdxPtr] = np.nanmax(np.asarray(gquadValues[(idx-2):(idx+vari.FRAGLEN-2)]))

	return covariates, pastMer1, pastMer2, pastStartGibbs


cpdef calculateContinuousFrag(sequence, mapValueView, gquadValueView, covariDataSet, result, analysisStart, shearStart, fragEnd, binStart, binEnd):
	##### INITIALIZE VARIABLES
	if vari.SHEAR == 1:
		pastMer1 = -1
		pastMer2 = -1

	if vari.PCR == 1:
		pastStartGibbs = -1

	resultStartIdx = -1
	resultEndIdx = -1

	##### INDEX IN 'sequence'
	startIdx = 2  # index in the genome sequence file (Included in the range)
	endIdx = (fragEnd - vari.FRAGLEN) - shearStart + 1   # index in the genome sequence file (Not included in the range)


	for idx in range(startIdx, endIdx):
		covariates, pastMer1, pastMer2, pastStartGibbs = fragCovariates(idx, pastMer1, pastMer2, pastStartGibbs, sequence, mapValueView, gquadValueView)

		### DETERMINE WHICH ROWS TO EDIT IN RESULT MATRIX
		thisFragStart = idx + shearStart
		thisFragEnd = thisFragStart + vari.FRAGLEN

		if resultStartIdx == -1:
			resultStartIdx = 0
			resultEndIdx = 1 # not included
			maxBinPos = binStart + vari.FRAGLEN
			numPoppedPos = 0

			while not np.isnan(result[resultEndIdx, 0]) and result[resultEndIdx, 0] < thisFragEnd:
				resultEndIdx += 1
				if resultEndIdx > vari.FRAGLEN:
					resultEndIdx -= (vari.FRAGLEN + 1)
		else:
			while result[resultStartIdx, 0] < thisFragStart:
				## pop the element
				covariDataSet[analysisStart + numPoppedPos - COVARIATE_FILE_INDEX_OFFSET, :] = result[resultStartIdx, 1:]
				result[resultStartIdx, 1:] = 0.0

				numPoppedPos += 1
				if maxBinPos >= binEnd:
					result[resultStartIdx, 0] = np.nan
				else:
					result[resultStartIdx, 0] = maxBinPos + 1
					maxBinPos += 1

				resultStartIdx += 1
				if resultStartIdx > vari.FRAGLEN:
					resultStartIdx -= (vari.FRAGLEN + 1)

			while not np.isnan(result[resultEndIdx, 0]) and result[resultEndIdx, 0] < thisFragEnd:
				resultEndIdx += 1
				if resultEndIdx > vari.FRAGLEN:
					resultEndIdx -= (vari.FRAGLEN + 1)

		if resultEndIdx < resultStartIdx:
			result[resultStartIdx:vari.FRAGLEN+1, 1:] += covariates
			result[0:resultEndIdx, 1:] += covariates
		else:
			result[resultStartIdx:resultEndIdx, 1:] += covariates

		if idx == (endIdx-1): # the last fragment
			### pop the rest of positions that are not np.nan
			if resultEndIdx < resultStartIdx:
				posLen = (vari.FRAGLEN+1) - resultStartIdx
				start = analysisStart + numPoppedPos - COVARIATE_FILE_INDEX_OFFSET
				end = start + posLen
				covariDataSet[start:end, :] = result[resultStartIdx: vari.FRAGLEN+1, 1:]
				numPoppedPos += posLen

				posLen = resultEndIdx
				start = analysisStart + numPoppedPos - COVARIATE_FILE_INDEX_OFFSET
				end = start + posLen
				covariDataSet[start:end, :] = result[0: resultEndIdx, 1:]
				numPoppedPos += posLen
			else:
				posLen = resultEndIdx - resultStartIdx
				start = analysisStart + numPoppedPos - COVARIATE_FILE_INDEX_OFFSET
				end = start + posLen
				covariDataSet[start:end, :] = result[resultStartIdx: resultEndIdx, 1:]
				numPoppedPos += posLen


cpdef calculateDiscreteFrag(chromoEnd, sequence, mapValueView, gquadValueView, covariDataSet, analysisStart, shearStart, binStart, binEnd, nBins):
	for resultIdx in range(nBins): # for each bin
		if resultIdx == (nBins-1):
			pos = binEnd
		else:
			pos = binStart + resultIdx

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

		line = np.zeros(vari.COVARI_NUM)
		for binFragIdx in range(thisBinFirstFragStartIdx, thisBinFirstFragStartIdx + thisBinNumFrag):
			covariates, pastMer1, pastMer2, pastStartGibbs = fragCovariates(binFragIdx, pastMer1, pastMer2, pastStartGibbs, sequence, mapValueView, gquadValueView)

			line += covariates

		covariDataSet[analysisStart + resultIdx - COVARIATE_FILE_INDEX_OFFSET, :] = line


cpdef calculateTaskCovariates(chromo, outputFilename, regions):
	genome = py2bit.open(vari.GENOME)
	chromoEnd = int(genome.chroms(chromo))
	mapFile = pyBigWig.open(vari.MAPFILE)
	gquadFiles = [0] * len(vari.GQAUDFILE)
	for i, file in enumerate(vari.GQAUDFILE):
		gquadFiles[i] = pyBigWig.open(file)


	##### CREATE COVARIATE FILE
	f = h5py.File(outputFilename, "w")
	covariDataSet = f.create_dataset("covari", (chromoEnd, vari.COVARI_NUM), dtype='f', compression="gzip")

	##### CALCULATE COVARIATE VALUES
	for _chromo, analysisStart, analysisEnd in regions:
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

			if secondBinPos - firstBinPos <= vari.FRAGLEN:
				continuousFrag = True

		###### CALCULATE INDEX VARIABLE
		analysisStart, analysisEnd, fragStart, fragEnd, shearStart, shearEnd, binStart, binEnd, nBins = calculateBoundaries(chromoEnd, analysisStart, analysisEnd, firstBinPos, lastBinPos, nBins)

		###### GET SEQUENCE
		sequence = genome.sequence(chromo, (shearStart-1), (shearEnd-1))

		##### GET BIASES INFO FROM FILES
		if vari.MAP == 1:
			mapValueView = mapValues(mapFile, chromo, fragStart, fragEnd)

		if vari.GQUAD == 1:
			gquadValueView = gquadValues(gquadFiles, chromo, fragStart, fragEnd)

		if continuousFrag:
			###### GENERATE A RESULT MATRIX
			result = makeMatrixContinuousFrag(binStart, binEnd, nBins)

			calculateContinuousFrag(sequence, mapValueView, gquadValueView, covariDataSet, result, analysisStart, shearStart, fragEnd, binStart, binEnd)
		else:
			calculateDiscreteFrag(chromoEnd, sequence, mapValueView, gquadValueView, covariDataSet, analysisStart, shearStart, binStart, binEnd, nBins)

	for file in gquadFiles:
		file.close()
	mapFile.close()
	f.close()
	genome.close()


cpdef makeMatrixContinuousFrag(binStart, binEnd, nBins):
	result = np.zeros(((vari.FRAGLEN+1), (vari.COVARI_NUM+1)), dtype=np.float64)
	for i in range(vari.FRAGLEN+1):
		pos = binStart + i

		if pos > binEnd:
			result[i, 0] = np.nan
		else:
			result[i, 0] = pos

	if nBins == (vari.FRAGLEN+1):
		result[vari.FRAGLEN, 0] = binEnd

	return result
