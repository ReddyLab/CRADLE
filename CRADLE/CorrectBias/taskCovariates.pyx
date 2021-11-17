# cython: language_level=3

import tempfile
import warnings
import h5py
import numpy as np
import py2bit
import pyBigWig

import CRADLE.CorrectBias.covariateUtils as cu

from CRADLE.CorrectBias import vari
from CRADLE.CorrectBias.readCounts import correctReadCounts
from CRADLE.correctbiasutils import vari as commonVari

cpdef calculateContinuousFrag(chromo, analysisStart, analysisEnd, binStart, binEnd, nBins, lastBin):
	warnings.filterwarnings('ignore', r'All-NaN slice encountered')
	warnings.filterwarnings('ignore', r'Mean of empty slice')


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

			###  mer2
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


	return correctReadCounts(covariFileName, chromo, analysisStart, analysisEnd, lastBin, nBins)


cpdef calculateDiscreteFrag(chromo, analysisStart, analysisEnd, binStart, binEnd, nBins, lastBin):
	warnings.filterwarnings('ignore', r'All-NaN slice encountered')
	warnings.filterwarnings('ignore', r'Mean of empty slice')


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


	##### STORE COVARI RESULTS
	covariFileTemp = tempfile.NamedTemporaryFile(suffix=".hdf5", dir=commonVari.OUTPUT_DIR, delete=True)
	covariFileName = covariFileTemp.name
	covariFileTemp.close()

	f = h5py.File(covariFileName, "w")
	covariFile = f.create_dataset("covari", (nBins, vari.COVARI_NUM), dtype='f', compression="gzip")

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

				###  mer2
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


			for covariPos in range(vari.COVARI_NUM):
				line[covariPos] = line[covariPos] + covariIdx[covariPos]

		covariFile[resultIdx] = line

		resultIdx = resultIdx + 1

		if resultIdx == nBins:
			break

	f.close()

	return correctReadCounts(covariFileName, chromo, analysisStart, analysisEnd, lastBin, nBins)


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
