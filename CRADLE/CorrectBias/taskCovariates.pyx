# cython: language_level=3

import tempfile
import warnings
import h5py
import numpy as np
import py2bit
import pyBigWig

import CRADLE.CorrectBias.covariateUtils as cu

from CRADLE.CorrectBias.readCounts import correctReadCounts

cpdef calculateContinuousFrag(chromo, analysisStart, analysisEnd, binStart, binEnd, nBins, lastBin, globalVars):
	warnings.filterwarnings('ignore', r'All-NaN slice encountered')
	warnings.filterwarnings('ignore', r'Mean of empty slice')


	###### CALCULATE INDEX VARIABLE
	fragStart = binStart + 1 - globalVars["fragLen"]
	fragEnd = binEnd + globalVars["fragLen"]  # not included
	shearStart = fragStart - 2
	shearEnd = fragEnd + 2 # not included

	genome = py2bit.open(globalVars["genome"])
	chromoEnd = int(genome.chroms(chromo))

	if shearStart < 1:
		shearStart = 1
		fragStart = 3
		binStart = max(binStart, fragStart)
		analysisStart = max(analysisStart, fragStart)

		###### EDIT BINSTART/ BINEND
		nBins = int( (analysisEnd - analysisStart) / float(globalVars["binSize"]) )
		leftValue = (analysisEnd - analysisStart) % int(globalVars["binSize"])
		if leftValue != 0:
			nBins = nBins + 1
			lastBin = True
			binEnd = int( (analysisStart + (nBins-1) * globalVars["binSize"] + analysisEnd) / float(2) )
		else:
			lastBin = False
			binEnd = binStart + (nBins-1) * globalVars["binSize"]

		fragEnd = binEnd + globalVars["fragLen"]
		shearEnd = fragEnd + 2

	if shearEnd > chromoEnd:
		analysisEndModified = min(analysisEnd, chromoEnd - 2)  # not included

		if analysisEndModified == analysisEnd:
			shearEnd = chromoEnd
			fragEnd = shearEnd - 2
		else:
			analysisEnd = analysisEndModified

			###### EDIT BINSTART/ BINEND
			nBins = int( (analysisEnd - analysisStart) / float(globalVars["binSize"]) )
			leftValue = (analysisEnd - analysisStart) % int(globalVars["binSize"])
			if leftValue != 0:
				nBins = nBins + 1
				lastBin = True
				binEnd = int( (analysisStart + (nBins-1) * globalVars["binSize"] + analysisEnd) / float(2) )
			else:
				lastBin = False
				binEnd = binStart + (nBins-1) * globalVars["binSize"]

			fragEnd = binEnd + globalVars["fragLen"]
			shearEnd = fragEnd + 2

			if shearEnd > chromoEnd:
				shearEnd = chromoEnd
				fragEnd = shearEnd - 2

	###### GENERATE A RESULT MATRIX
	result = makeMatrixContinuousFrag(binStart, binEnd, nBins, globalVars)

	###### GET SEQUENCE
	sequence = genome.sequence(chromo, (shearStart-1), (shearEnd-1))
	genome.close()

	##### OPEN BIAS FILES
	if globalVars["map"] == 1:
		mapFile = pyBigWig.open(globalVars["mapFile"])
		mapValue = np.array(mapFile.values(chromo, fragStart, fragEnd))

		mapValue[np.where(mapValue == 0)] = np.nan
		mapValue = np.log(mapValue)
		mapValue[np.where(np.isnan(mapValue))] = float(-6)

		mapValueView = cu.memoryView(mapValue)
		mapFile.close()
		del mapFile, mapValue

	if globalVars["gquad"] == 1:
		gquadFile = [0] * len(globalVars["gquadFile"])
		gquadValue = [0] * len(globalVars["gquadFile"])

		for i in range(len(globalVars["gquadFile"])):
			gquadFile[i] = pyBigWig.open(globalVars["gquadFile"][i])
			gquadValue[i] = gquadFile[i].values(chromo, fragStart, fragEnd)
			gquadFile[i].close()

		gquadValue = np.array(gquadValue)
		gquadValue = np.nanmax(gquadValue, axis=0)
		gquadValue[np.where(gquadValue == 0)] = np.nan
		gquadValue = np.log(gquadValue / float(globalVars["gquadMax"]))

		gquadValue[np.where(np.isnan(gquadValue))] = float(-5)
		gquadView = cu.memoryView(gquadValue)

		del gquadFile, gquadValue


	##### INDEX IN 'sequence'
	startIdx = 2  # index in the genome sequence file (Included in the range)
	endIdx = (fragEnd - globalVars["fragLen"]) - shearStart + 1   # index in the genome sequence file (Not included in the range)

	##### INITIALIZE VARIABLES
	if globalVars["shear"] == 1:
		pastMer1 = -1
		pastMer2 = -1
	if globalVars["pcr"] == 1:
		pastStartGibbs = -1

	resultStartIdx = -1
	resultEndIdx = -1

	##### STORE COVARI RESULTS
	covariFileTemp = tempfile.NamedTemporaryFile(suffix=".hdf5", dir=globalVars["outputDir"], delete=True)
	covariFileName = covariFileTemp.name
	covariFileTemp.close()

	f = h5py.File(covariFileName, "w")
	covariFile = f.create_dataset("covari", (nBins, globalVars["covariNum"]), dtype='f', compression="gzip")

	for idx in range(startIdx, endIdx):
		covariIdx = [0] * globalVars["covariNum"]
		covariIdxPtr = 0

		if globalVars["shear"] == 1:
			###  mer1
			mer1 = sequence[(idx-2):(idx+3)]
			if 'N' in mer1:
				pastMer1 = -1
				mgwIdx = globalVars["n_mgw"]
				protIdx = globalVars["n_prot"]
			else:
				if pastMer1 == -1: # there is no information on pastMer1
					pastMer1, mgwIdx, protIdx = cu.find5merProb(mer1, globalVars)
				else:
					pastMer1, mgwIdx, protIdx = cu.edit5merProb(pastMer1, mer1[0], mer1[4], globalVars)

			###  mer2
			fragEndIdx = idx + globalVars["fragLen"]
			mer2 = sequence[(fragEndIdx-3):(fragEndIdx+2)]
			if 'N' in mer2:
				pastMer2 = -1
				mgwIdx = mgwIdx + globalVars["n_mgw"]
				protIdx = protIdx + globalVars["n_prot"]
			else:
				if pastMer2 == -1:
					pastMer2, add1, add2 = cu.findComple5merProb(mer2, globalVars)
				else:
					pastMer2, add1, add2 = cu.editComple5merProb(pastMer2, mer2[0], mer2[4], globalVars)
				mgwIdx = mgwIdx + add1
				protIdx = protIdx + add2

			covariIdx[covariIdxPtr] = mgwIdx
			covariIdx[covariIdxPtr+1] = protIdx
			covariIdxPtr = covariIdxPtr + 2


		if globalVars["pcr"] == 1:
			sequenceIdx = sequence[idx:(idx+globalVars["fragLen"])]
			if pastStartGibbs == -1:
				startGibbs, gibbs = cu.findStartGibbs(sequenceIdx, globalVars["fragLen"], globalVars)
			else:
				oldDimer = sequenceIdx[0:2].upper()
				newDimer = sequenceIdx[(globalVars["fragLen"]-2):globalVars["fragLen"]].upper()
				startGibbs, gibbs = cu.editStartGibbs(oldDimer, newDimer, pastStartGibbs, globalVars)

			annealIdx, denatureIdx = cu.convertGibbs(gibbs, globalVars)

			covariIdx[covariIdxPtr] = annealIdx
			covariIdx[covariIdxPtr+1] = denatureIdx
			covariIdxPtr = covariIdxPtr + 2

		if globalVars["map"] == 1:
			map1 = mapValueView[(idx-2)]
			map2 = mapValueView[(idx+globalVars["fragLen"]-2-globalVars["kmer"])]
			mapIdx = map1 + map2

			covariIdx[covariIdxPtr] = mapIdx
			covariIdxPtr = covariIdxPtr + 1

		if globalVars["gquad"] == 1:
			gquadIdx = np.nanmax(np.asarray(gquadView[(idx-2):(idx+globalVars["fragLen"]-2)]))

			covariIdx[covariIdxPtr] = gquadIdx
			covariIdxPtr = covariIdxPtr + 1

		### DETERMINE WHICH ROWS TO EDIT IN RESULT MATRIX
		thisFragStart = idx + shearStart
		thisFragEnd = thisFragStart + globalVars["fragLen"]

		if resultStartIdx == -1:
			resultStartIdx = 0
			resultEndIdx = 1 # not included
			if not np.isnan(result[resultEndIdx, 0]):
				while result[resultEndIdx, 0] < thisFragEnd:
					resultEndIdx = resultEndIdx + 1
					if resultEndIdx > globalVars["fragLen"]:
						resultEndIdx = resultEndIdx - (globalVars["fragLen"]+1)
					if np.isnan(result[resultEndIdx, 0]):
						break
			maxBinPos = binStart + globalVars["fragLen"]
			numPoppedPos = 0
		else:

			while result[resultStartIdx, 0] < thisFragStart:
				## pop the element
				line = []
				for covariPos in range(globalVars["covariNum"]):
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
				if resultStartIdx > globalVars["fragLen"]:
					resultStartIdx = resultStartIdx - (globalVars["fragLen"]+1)

			if not np.isnan(result[resultEndIdx, 0]):
				while result[resultEndIdx, 0] < thisFragEnd:
					resultEndIdx = resultEndIdx + 1
					if resultEndIdx > globalVars["fragLen"]:
						resultEndIdx = resultEndIdx - (globalVars["fragLen"]+1)
					if np.isnan(result[resultEndIdx, 0]):
						break

		if resultEndIdx < resultStartIdx:
			for pos in range(resultStartIdx, (globalVars["fragLen"]+1)):
				for covariPos in range(globalVars["covariNum"]):
					result[pos, covariPos+1] = result[pos, covariPos+1] + covariIdx[covariPos]
			for pos in range(0, resultEndIdx):
				for covariPos in range(globalVars["covariNum"]):
					result[pos, covariPos+1] = result[pos, covariPos+1] + covariIdx[covariPos]
		else:
			for pos in range(resultStartIdx, resultEndIdx):
				for covariPos in range(globalVars["covariNum"]):
					result[pos, covariPos+1] = result[pos, covariPos+1] + covariIdx[covariPos]

		if idx == (endIdx-1): # the last fragment
			### pop the rest of positions that are not np.nan
			if resultEndIdx < resultStartIdx:
				for pos in range(resultStartIdx, (globalVars["fragLen"]+1)):
					line = []
					for covariPos in range(globalVars["covariNum"]):
						line.extend([ result[pos, (covariPos+1)]  ])
					covariFile[numPoppedPos] = line

					numPoppedPos = numPoppedPos + 1

				for pos in range(0, resultEndIdx):
					line = []
					for covariPos in range(globalVars["covariNum"]):
						line.extend([ result[pos, (covariPos+1)]  ])
					covariFile[numPoppedPos] = line

					numPoppedPos = numPoppedPos + 1
			else:
				for pos in range(resultStartIdx, resultEndIdx):
					line = []
					for covariPos in range(globalVars["covariNum"]):
						line.extend([ result[pos, (covariPos+1)]  ])
					covariFile[numPoppedPos] = line

					numPoppedPos = numPoppedPos + 1

	f.close()


	return correctReadCounts(covariFileName, chromo, analysisStart, analysisEnd, lastBin, nBins, globalVars)


cpdef calculateDiscreteFrag(chromo, analysisStart, analysisEnd, binStart, binEnd, nBins, lastBin, globalVars):
	warnings.filterwarnings('ignore', r'All-NaN slice encountered')
	warnings.filterwarnings('ignore', r'Mean of empty slice')


	###### CALCULATE INDEX VARIABLE
	fragStart = binStart + 1 - globalVars["fragLen"]
	fragEnd = binEnd + globalVars["fragLen"]  # not included
	shearStart = fragStart - 2
	shearEnd = fragEnd + 2 # not included

	genome = py2bit.open(globalVars["genome"])
	chromoEnd = int(genome.chroms(chromo))

	if shearStart < 1:
		shearStart = 1
		fragStart = 3
		binStart = max(binStart, fragStart)
		analysisStart = max(analysisStart, fragStart)

		###### EDIT BINSTART/ BINEND
		nBins = int( (analysisEnd-analysisStart) / float(globalVars["binSize"]) )
		leftValue = (analysisEnd - analysisStart) % int(globalVars["binSize"])
		if leftValue != 0:
			nBins = nBins + 1
			lastBin = True
			binEnd = int( (analysisStart + (nBins-1) * globalVars["binSize"] + analysisEnd) / float(2) )
		else:
			lastBin = False
			binEnd = binStart + (nBins-1) * globalVars["binSize"]

		fragEnd = binEnd + globalVars["fragLen"]
		shearEnd = fragEnd + 2

	if shearEnd > chromoEnd:
		analysisEndModified = min(analysisEnd, chromoEnd - 2)  # not included

		if analysisEndModified == analysisEnd:
			shearEnd = chromoEnd
			fragEnd = shearEnd - 2
		else:
			analysisEnd = analysisEndModified

			###### EDIT BINSTART/ BINEND
			nBins = int( (analysisEnd-analysisStart) / float(globalVars["binSize"]) )
			leftValue = (analysisEnd - analysisStart) % int(globalVars["binSize"])
			if leftValue != 0:
				nBins = nBins + 1
				lastBin = True
				binEnd = int( (analysisStart + (nBins-1) * globalVars["binSize"] + analysisEnd) / float(2) )
			else:
				lastBin = False
				binEnd = binStart + (nBins-1) * globalVars["binSize"]

			fragEnd = binEnd + globalVars["fragLen"]
			shearEnd = fragEnd + 2

			if shearEnd > chromoEnd:
				shearEnd = chromoEnd
				fragEnd = shearEnd - 2


	###### GET SEQUENCE
	sequence = genome.sequence(chromo, (shearStart-1), (shearEnd-1))
	genome.close()

	##### OPEN BIAS FILES
	if globalVars["map"] == 1:
		mapFile = pyBigWig.open(globalVars["mapFile"])
		mapValue = np.array(mapFile.values(chromo, fragStart, fragEnd))

		mapValue[np.where(mapValue == 0)] = np.nan
		mapValue = np.log(mapValue)
		mapValue[np.where(np.isnan(mapValue))] = float(-6)

		mapValueView = cu.memoryView(mapValue)
		mapFile.close()
		del mapFile, mapValue

	if globalVars["gquad"] == 1:
		gquadFile = [0] * len(globalVars["gquadFile"])
		gquadValue = [0] * len(globalVars["gquadFile"])

		for i in range(len(globalVars["gquadFile"])):
			gquadFile[i] = pyBigWig.open(globalVars["gquadFile"][i])
			gquadValue[i] = gquadFile[i].values(chromo, fragStart, fragEnd)
			gquadFile[i].close()

		gquadValue = np.array(gquadValue)
		gquadValue = np.nanmax(gquadValue, axis=0)
		gquadValue[np.where(gquadValue == 0)] = np.nan
		gquadValue = np.log(gquadValue / float(globalVars["gquadMax"]))

		gquadValue[np.where(np.isnan(gquadValue))] = float(-5)
		gquadView = cu.memoryView(gquadValue)

		del gquadFile, gquadValue


	##### STORE COVARI RESULTS
	covariFileTemp = tempfile.NamedTemporaryFile(suffix=".hdf5", dir=globalVars["outputDir"], delete=True)
	covariFileName = covariFileTemp.name
	covariFileTemp.close()

	f = h5py.File(covariFileName, "w")
	covariFile = f.create_dataset("covari", (nBins, globalVars["covariNum"]), dtype='f', compression="gzip")

	resultIdx = 0
	while resultIdx < nBins: # for each bin
		if resultIdx == (nBins-1):
			pos = binEnd
		else:
			pos = binStart + resultIdx * globalVars["binSize"]

		thisBinFirstFragStart = pos + 1 - globalVars["fragLen"]
		thisBinLastFragStart = pos

		if thisBinFirstFragStart < 3:
			thisBinFirstFragStart = 3
		if (thisBinLastFragStart + globalVars["fragLen"]) > (chromoEnd - 2):
			thisBinLastFragStart = chromoEnd - 2 - globalVars["fragLen"]

		thisBinNumFrag = thisBinLastFragStart - thisBinFirstFragStart + 1

		thisBinFirstFragStartIdx = thisBinFirstFragStart - shearStart

		##### INITIALIZE VARIABLES
		if globalVars["shear"] == 1:
			pastMer1 = -1
			pastMer2 = -1
		if globalVars["pcr"] == 1:
			pastStartGibbs = -1

		line = [0.0] * globalVars["covariNum"]
		for binFragIdx in range(thisBinNumFrag):
			idx = thisBinFirstFragStartIdx + binFragIdx

			covariIdx = [0] * globalVars["covariNum"]
			covariIdxPtr = 0

			if globalVars["shear"] == 1:
				###  mer1
				mer1 = sequence[(idx-2):(idx+3)]
				if 'N' in mer1:
					pastMer1 = -1
					mgwIdx = globalVars["n_mgw"]
					protIdx = globalVars["n_prot"]
				else:
					if pastMer1 == -1: # there is no information on pastMer1
						pastMer1, mgwIdx, protIdx = cu.find5merProb(mer1, globalVars)
					else:
						pastMer1, mgwIdx, protIdx = cu.edit5merProb(pastMer1, mer1[0], mer1[4], globalVars)

				###  mer2
				fragEndIdx = idx + globalVars["fragLen"]
				mer2 = sequence[(fragEndIdx-3):(fragEndIdx+2)]
				if 'N' in mer2:
					pastMer2 = -1
					mgwIdx = mgwIdx + globalVars["n_mgw"]
					protIdx = protIdx + globalVars["n_prot"]
				else:
					if pastMer2 == -1:
						pastMer2, add1, add2 = cu.findComple5merProb(mer2, globalVars)
					else:
						pastMer2, add1, add2 = cu.editComple5merProb(pastMer2, mer2[0], mer2[4], globalVars)
					mgwIdx = mgwIdx + add1
					protIdx = protIdx + add2

				covariIdx[covariIdxPtr] = mgwIdx
				covariIdx[covariIdxPtr+1] = protIdx
				covariIdxPtr = covariIdxPtr + 2

			if globalVars["pcr"] == 1:
				sequenceIdx = sequence[idx:(idx+globalVars["fragLen"])]
				if pastStartGibbs == -1:
					startGibbs, gibbs = cu.findStartGibbs(sequenceIdx, globalVars["fragLen"], globalVars)
				else:
					oldDimer = sequenceIdx[0:2].upper()
					newDimer = sequenceIdx[(globalVars["fragLen"]-2):globalVars["fragLen"]].upper()
					startGibbs, gibbs = cu.editStartGibbs(oldDimer, newDimer, pastStartGibbs, globalVars)

				annealIdx, denatureIdx = cu.convertGibbs(gibbs, globalVars)

				covariIdx[covariIdxPtr] = annealIdx
				covariIdx[covariIdxPtr+1] = denatureIdx
				covariIdxPtr = covariIdxPtr + 2

			if globalVars["map"] == 1:
				map1 = mapValueView[(idx-2)]
				map2 = mapValueView[(idx+globalVars["fragLen"]-2-globalVars["kmer"])]
				mapIdx = map1 + map2

				covariIdx[covariIdxPtr] = mapIdx
				covariIdxPtr = covariIdxPtr + 1

			if globalVars["gquad"] == 1:
				gquadIdx = np.nanmax(np.asarray(gquadView[(idx-2):(idx+globalVars["fragLen"]-2)]))

				covariIdx[covariIdxPtr] = gquadIdx
				covariIdxPtr = covariIdxPtr + 1


			for covariPos in range(globalVars["covariNum"]):
				line[covariPos] = line[covariPos] + covariIdx[covariPos]

		covariFile[resultIdx] = line

		resultIdx = resultIdx + 1

		if resultIdx == nBins:
			break

	f.close()

	return correctReadCounts(covariFileName, chromo, analysisStart, analysisEnd, lastBin, nBins, globalVars)


cpdef calculateTaskCovariates(chromo, analysisStart, analysisEnd, globalVars):
	### supress numpy nan-error message
	warnings.filterwarnings('ignore', r'All-NaN slice encountered')
	warnings.filterwarnings('ignore', r'Mean of empty slice')

	#### DECIDE IF 'calculateContinuousFrag' or 'calculateDiscreteFrag'
	if (analysisStart + globalVars["binSize"]) >= analysisEnd:
		firstBinPos = int((analysisStart + analysisEnd) / float(2))
		lastBinPos = firstBinPos
		nBins = 1
		lastBin = False
		result = calculateContinuousFrag(chromo, analysisStart, analysisEnd, firstBinPos, lastBinPos, nBins, lastBin, globalVars)
		return result

	firstBinPos = int((2*analysisStart + globalVars["binSize"]) / float(2))
	if (analysisStart + 2*globalVars["binSize"]) > analysisEnd:
		secondBinPos = int((analysisStart + globalVars["binSize"] + analysisEnd) / float(2))
		lastBinPos = secondBinPos
		nBins = 2
		lastBin = True
	else:
		secondBinPos = int((2*analysisStart + 3 * globalVars["binSize"]) / float(2))
		leftValue = (analysisEnd - analysisStart) % int(globalVars["binSize"])
		nBins = int((analysisEnd - analysisStart) / float(globalVars["binSize"]))
		if leftValue != 0:
			nBins = nBins + 1
			lastBin = True
			lastBinPos = int( (analysisStart + (nBins-1) * globalVars["binSize"] + analysisEnd) / float(2) )  ## should be included in the analysis
		else:
			lastBin = False
			lastBinPos = firstBinPos + (nBins-1) * globalVars["binSize"] ## should be included in the analysis

	if (secondBinPos-firstBinPos) <= globalVars["fragLen"]:
		result = calculateContinuousFrag(chromo, analysisStart, analysisEnd, firstBinPos, lastBinPos, nBins, lastBin, globalVars)
		return result
	else:
		result = calculateDiscreteFrag(chromo, analysisStart, analysisEnd, firstBinPos, lastBinPos, nBins, lastBin, globalVars)
		return result


cpdef makeMatrixContinuousFrag(binStart, binEnd, nBins, globalVars):

	result = np.zeros(((globalVars["fragLen"]+1), (globalVars["covariNum"]+1)), dtype=np.float64)
	for i in range(globalVars["fragLen"]+1):
		pos = binStart + i * globalVars["binSize"]

		if pos > binEnd:
			result[i, 0] = np.nan
		else:
			result[i, 0] = pos

	if nBins == (globalVars["fragLen"]+1):
		result[globalVars["fragLen"], 0] = binEnd

	return result
