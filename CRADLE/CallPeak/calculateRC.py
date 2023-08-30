import os
import tempfile
import warnings
import numpy as np
import pyBigWig
import scipy.stats
import statsmodels.sandbox.stats.multicomp

def getVarianceAndRegionCutoff(region, globalVars):
	warnings.filterwarnings('ignore', r'All-NaN slice encountered')
	warnings.filterwarnings('ignore', r'Mean of empty slice')
	warnings.filterwarnings('ignore', r'Degrees of freedom <= 0 for slice')

	regionChromo = region[0]
	regionStart = region[1]
	regionEnd = region[2]

	numBin = max(1, int((regionEnd - regionStart) / globalVars["binSize1"]))

	totalRC = []
	sampleRC = []
	for filename in globalVars["ctrlbwNames"]:
		with pyBigWig.open(filename) as bwFile:
			temp = np.array(bwFile.stats(regionChromo, regionStart, regionEnd,  type="mean", nBins=numBin))
			temp[np.where(temp == None)] = np.nan
			tempList = temp.tolist()
			totalRC.append(tempList)
			sampleRC.append(tempList)

	ctrlMean = np.nanmean(np.array(sampleRC), axis=0)

	sampleRC = []
	for filename in globalVars["expbwNames"]:
		with pyBigWig.open(filename) as bwFile:
			temp = np.array(bwFile.stats(regionChromo, regionStart, regionEnd,  type="mean", nBins=numBin))
			temp[np.where(temp == None)] = np.nan
			tempList = temp.tolist()
			totalRC.append(tempList)
			sampleRC.append(tempList)

	expMean = np.nanmean(np.array(sampleRC), axis=0)

	var = np.nanvar(np.array(totalRC), axis=0)
	var = np.array(var)
	idx = np.where(np.isnan(var) == True)
	var = np.delete(var, idx)

	diff = np.array(np.absolute(expMean - ctrlMean))
	idx = np.where(np.isnan(diff) == True)
	diff = np.delete(diff, idx)

	if len(var) == 0:
		var = None
	else:
		var = var.tolist()

	if len(diff) == 0:
		diff = None
	else:
		diff = diff.tolist()

	return var, diff


def statTest(definedRegion, globalVars):
	region = defineRegion(definedRegion, globalVars)

	if region is not None:
		return doWindowApproach(region, globalVars)

	return None


def defineRegion(region, globalVars):
	warnings.filterwarnings('ignore', r'All-NaN slice encountered')
	warnings.filterwarnings('ignore', r'Mean of empty slice')

	analysisChromo = region[0]
	analysisStart = region[1]
	analysisEnd = region[2]

	#### Number of bins
	binNum = (analysisEnd - analysisStart) // globalVars["binSize1"]
	regionStart = analysisStart

	if binNum == 0:
		regionEnd = analysisEnd
		lastBinExist = False
		binNum = 1
	else:
		if (analysisStart + binNum * globalVars["binSize1"]) < analysisEnd:
			lastBinExist = True   # exist!
			regionEnd = regionStart + binNum * globalVars["binSize1"]
			lastBinStart = regionEnd
			lastBinEnd = analysisEnd
		else:
			lastBinExist = False
			regionEnd = analysisEnd

	#### ctrlMean
	sampleRC1 = []

	for filename in globalVars["ctrlbwNames"]:
		with pyBigWig.open(filename) as bwFile:
			temp = np.array(bwFile.stats(analysisChromo, regionStart, regionEnd,  type="mean", nBins=binNum))
			temp[np.where(temp == None)] = np.nan
			temp = temp.tolist()

			if lastBinExist:
				lastValue = bwFile.stats(analysisChromo, lastBinStart, lastBinEnd, type="mean", nBins=1)[0]
				if lastValue is None:
					lastValue = np.nan
				temp.append(lastValue)

			sampleRC1.append(temp)

	ctrlMean = np.nanmean( np.array(sampleRC1), axis=0)
	del sampleRC1

	#### expMean
	sampleRC2 = []
	for filename in globalVars["expbwNames"]:
		with pyBigWig.open(filename) as bwFile:
			temp = np.array(bwFile.stats(analysisChromo, regionStart, regionEnd,  type="mean", nBins=binNum))
			temp[np.where(temp == None)] = np.nan
			temp = temp.tolist()

			if lastBinExist:
				lastValue = bwFile.stats(analysisChromo, lastBinStart, lastBinEnd, type="mean", nBins=1)[0]
				if lastValue is None:
					lastValue = np.nan
				temp.append(lastValue)

			sampleRC2.append(temp)

	expMean = np.nanmean( np.array(sampleRC2), axis=0)
	del sampleRC2

	#### diff
	diff = np.array(expMean - ctrlMean, dtype=np.float64)

	idx = 0
	pastGroupType = -2
	numRegion = 0
	definedRegion = []

	if lastBinExist:
		binNum = binNum + 1

	while idx < binNum:
		if np.isnan(diff[idx]):
			if pastGroupType == -2:
				idx += 1
				continue

			definedRegion.append(regionVector)
			numRegion = numRegion + 1
			idx = idx + 1
			pastGroupType = -2
			continue

		if np.absolute(diff[idx]) > globalVars["regionCutoff"]:
			if diff[idx] > 0:
				currGroupType = 1   # enriched
			else:
				currGroupType = -1   # repressed
		else:
			currGroupType = 0

		if pastGroupType == -2: # the first region
			regionVector = [
				analysisChromo,
				(analysisStart + globalVars["binSize1"] * idx),
				(analysisStart + globalVars["binSize1"] * idx + globalVars["binSize1"]),
				currGroupType
			]
			if idx == (binNum - 1):
				regionVector[2] = analysisEnd
				definedRegion.append(regionVector)
				numRegion += 1
				break

			pastGroupType = currGroupType
			idx += 1
			continue

		if currGroupType != pastGroupType:
			## End a previous region
			regionVector[2] = analysisStart + globalVars["binSize1"] * idx + globalVars["binSize1"] - globalVars["binSize1"]
			definedRegion.append(regionVector)
			numRegion += 1

			## Start a new reigon
			regionVector = [
				analysisChromo,
				(analysisStart + globalVars["binSize1"] * idx),
				(analysisStart + globalVars["binSize1"] * idx + globalVars["binSize1"]),
				currGroupType
			]
		else:
			regionVector[2] = analysisStart + globalVars["binSize1"] * idx + globalVars["binSize1"]

		if idx == (binNum-1):
			regionVector[2] = analysisEnd
			definedRegion.append(regionVector)
			numRegion += 1
			break

		pastGroupType = currGroupType
		idx += 1


	### variance check
	ctrlBW = [pyBigWig.open(filename) for filename in globalVars["ctrlbwNames"]]
	expBW = [pyBigWig.open(filename) for filename in globalVars["expbwNames"]]

	if len(definedRegion) == 0:
		return None

	definedRegion = restrictRegionLen(definedRegion, globalVars)

	regions = []
	for chromo, start, end, groupType in definedRegion:
		readCounts = []
		for ctrlBWFile in ctrlBW:
			rcTemp = np.nanmean(ctrlBWFile.values(chromo, start, end))
			readCounts.append(rcTemp)

		for expBWFile in expBW:
			rcTemp = np.nanmean(expBWFile.values(chromo, start, end))
			readCounts.append(rcTemp)

		regionVar = np.nanvar(readCounts)

		if np.isnan(regionVar):
			continue

		thetaIdx = np.max(np.where(globalVars["filterCutoffs"] < regionVar))
		theta = globalVars["filterCutoffsThetas"][thetaIdx]

		regions.append((chromo, start, end, groupType, theta))

	if len(regions) == 0:
		return None

	for file in ctrlBW:
		file.close()
	for file in expBW:
		file.close()

	return regions


def restrictRegionLen(definedRegion, globalVars):
	definedRegionNew = []

	maxRegionLen = globalVars["binSize1"] * 3

	for region in definedRegion:
		start = region[1]
		end = region[2]
		groupType = region[3]
		regionLen = end - start

		if regionLen > maxRegionLen:
			newNumBin = int(np.ceil(regionLen / maxRegionLen))
			newRegionSize = int(regionLen / newNumBin)

			for newBinIdx in range(newNumBin):
				newStart = start + newBinIdx * newRegionSize
				newEnd = newStart + newRegionSize
				if newBinIdx == (newNumBin - 1):
					newEnd = end
				definedRegionNew.append((region[0], newStart, newEnd, groupType))
		else:
			definedRegionNew.append(region)

	return definedRegionNew


def doWindowApproach(regions, globalVars):
	warnings.simplefilter("ignore", category=RuntimeWarning)

	subfile = tempfile.NamedTemporaryFile(mode="w+t", dir=globalVars["outputDir"], delete=False)
	simesP = []
	writtenRegionNum = 0

	for regionChromo, regionStart, regionEnd, _regionGroupType, regionTheta in regions:
		totalRC = []
		for filename in globalVars["ctrlbwNames"]:
			with pyBigWig.open(filename) as bwFile:
				temp = bwFile.values(regionChromo, regionStart, regionEnd)
				totalRC.append(temp)

		for filename in globalVars["expbwNames"]:
			with pyBigWig.open(filename) as bwFile:
				temp = bwFile.values(regionChromo, regionStart, regionEnd)
				totalRC.append(temp)

		totalRC = np.array(totalRC)
		binStartIdx = 0
		analysisEndIdx = regionEnd - regionStart

		windowPvalue = []
		windowEnrich = []

		while (binStartIdx + globalVars["binSize2"]) <= analysisEndIdx:
			if (binStartIdx + globalVars["shiftSize2"] + globalVars["binSize2"]) > analysisEndIdx:
				binEndIdx = analysisEndIdx
			else:
				binEndIdx = binStartIdx + globalVars["binSize2"]

			readCounts = np.nanmean(totalRC[:,binStartIdx:binEndIdx], axis=1)

			if len(np.where(np.isnan(readCounts))[0]) > 0:
				windowPvalue.append(np.nan)
				windowEnrich.append(np.nan)
				if (binEndIdx == analysisEndIdx) and (len(windowPvalue) != 0):
					#### calculate a Simes' p value for the reigon
					windowPvalue = np.array(windowPvalue)
					windowPvalueWoNan = windowPvalue[np.isnan(windowPvalue) == False]
					if len(windowPvalueWoNan) == 0:
						break
					rankPvalue = scipy.stats.rankdata(windowPvalueWoNan)
					numWindow = len(windowPvalueWoNan)
					pMerged = np.min((windowPvalueWoNan * numWindow) / rankPvalue)
					simesP.append(pMerged)

					regionInfo = [regionChromo, regionStart, regionEnd, regionTheta, pMerged]
					subfile.write('\t'.join([str(x) for x in regionInfo]) + "\t")
					subfile.write(','.join([str(x) for x in windowPvalue]) + "\t")
					subfile.write(','.join([str(x) for x in windowEnrich]) + "\n")
					writtenRegionNum += 1

				binStartIdx += globalVars["shiftSize2"]
				continue

			readCounts = readCounts.tolist()
			if readCounts == globalVars["allZero"]:
				windowPvalue.append(np.nan)
				windowEnrich.append(np.nan)

				if (binEndIdx == analysisEndIdx) and (len(windowPvalue) != 0):
					#### calculate a Simes' p value for the reigon
					windowPvalue = np.array(windowPvalue)
					windowPvalueWoNan = windowPvalue[np.isnan(windowPvalue) == False]
					if len(windowPvalueWoNan) == 0:
						break
					rankPvalue = scipy.stats.rankdata(windowPvalueWoNan)
					numWindow = len(windowPvalueWoNan)
					pMerged = np.min((windowPvalueWoNan * numWindow) / rankPvalue)
					simesP.append(pMerged)

					regionInfo = [regionChromo, regionStart, regionEnd, regionTheta, pMerged]
					subfile.write('\t'.join([str(x) for x in regionInfo]) + "\t")
					subfile.write(','.join([str(x) for x in windowPvalue]) + "\t")
					subfile.write(','.join([str(x) for x in windowEnrich]) + "\n")
					writtenRegionNum += 1

				binStartIdx += globalVars["shiftSize2"]
				continue

			windowInfo = doStatTesting(readCounts, globalVars)
			pvalue = float(windowInfo[1])
			enrich = int(windowInfo[0])
			windowPvalue.append(pvalue)
			windowEnrich.append(enrich)

			if (binEndIdx == analysisEndIdx) and (len(windowPvalue) != 0):
				#### calculate a Simes' p value for the reigon
				windowPvalue = np.array(windowPvalue)
				windowPvalueWoNan = windowPvalue[np.isnan(windowPvalue) == False]
				if len(windowPvalueWoNan) == 0:
					break
				rankPvalue = scipy.stats.rankdata(windowPvalueWoNan)
				numWindow = len(windowPvalueWoNan)
				pMerged = np.min((windowPvalueWoNan * numWindow) / rankPvalue)
				simesP.append(pMerged)

				regionInfo = [regionChromo, regionStart, regionEnd, regionTheta, pMerged]
				subfile.write('\t'.join([str(x) for x in regionInfo]) + "\t")
				subfile.write(','.join([str(x) for x in windowPvalue]) + "\t")
				subfile.write(','.join([str(x) for x in windowEnrich]) + "\n")
				writtenRegionNum += 1

			binStartIdx += globalVars["shiftSize2"]

	subfile.close()

	if writtenRegionNum == 0:
		os.remove(subfile.name)
		return None

	return subfile.name


def doStatTesting(rc, globalVars):
	ctrlbwNum = globalVars["ctrlbwNum"]
	ctrlRC = rc[:ctrlbwNum]
	expRC = rc[ctrlbwNum:]

	ctrlVar = np.nanvar(ctrlRC)
	expVar = np.nanvar(expRC)

	if (ctrlVar == 0) and (expVar == 0):
		statistics = float(np.nanmean(expRC) - np.nanmean(ctrlRC))

		pvalue = scipy.stats.norm.cdf(statistics, loc=0, scale=globalVars["nullStd"])

		if pvalue > 0.5:
			pvalue = (1 - pvalue) * 2
			enrich = 1
		else:
			pvalue = pvalue * 2
			enrich = -1

	elif (ctrlVar == 0) and (expVar != 0):
		loc = np.nanmean(ctrlRC)
		tResult = scipy.stats.ttest_1samp(expRC, popmean=loc)

		if tResult.statistic > 0:
			enrich = 1
		else:
			enrich = -1

		pvalue = tResult.pvalue

	elif (ctrlVar != 0) and (expVar == 0):
		loc = np.nanmean(expRC)
		tResult = scipy.stats.ttest_1samp(ctrlRC, popmean=loc)

		if tResult.statistic > 0:
			enrich = -1
		else:
			enrich = 1
		pvalue = tResult.pvalue

	else:
		welchResult = scipy.stats.ttest_ind(ctrlRC, expRC, equal_var=globalVars["equalVar"])

		if welchResult.statistic > 0:
			enrich = -1
		else:
			enrich = 1

		pvalue = welchResult.pvalue

	windowInfo = [enrich, pvalue]

	return windowInfo


def selectTheta(resultTTestFiles, globalVars):
	alpha = globalVars["fdr"]

	totalRegionNumArray = []
	selectRegionNumArray = []

	for theta in globalVars["filterCutoffsThetas"]:
		pValueSimes = []

		for ttestFile in resultTTestFiles:
			with open(ttestFile) as ttestResults:
				for region in ttestResults:
					regionLine = region.split()
					regionTheta = int(regionLine[3])
					regionPvalue = float(regionLine[4])

					if np.isnan(regionPvalue):
						continue

					if regionTheta >= theta:
						pValueSimes.append(regionPvalue)

		totalRegionNum = len(pValueSimes)
		pValueGroupBh = statsmodels.sandbox.stats.multicomp.multipletests(pValueSimes, alpha=alpha, method='fdr_bh')
		selectRegionNum = len(np.where(pValueGroupBh[0])[0])

		totalRegionNumArray.append(totalRegionNum)
		selectRegionNumArray.append(selectRegionNum)

	selectRegionNumArray = np.array(selectRegionNumArray)
	maxNum = np.max(selectRegionNumArray)
	idx = np.where(selectRegionNumArray == maxNum)
	idx = idx[0][0]

	adjFDR = ( globalVars["fdr"] * selectRegionNumArray[idx] ) / totalRegionNumArray[idx]

	return globalVars["filterCutoffsThetas"][idx], adjFDR, selectRegionNumArray[idx], totalRegionNumArray[idx]


def doFDRprocedure(inputFilename, selectRegionIdx, globalVars):
	inputStream = open(inputFilename)
	inputFile = inputStream.readlines()

	subfile = tempfile.NamedTemporaryFile(mode="w+t", dir=globalVars["outputDir"], delete=False)

	### open bw files to store diff value
	ctrlBWFiles = [pyBigWig.open(filename) for filename in globalVars["ctrlbwNames"]]
	expBWFiles = [pyBigWig.open(filename) for filename in globalVars["expbwNames"]]

	for regionIdx in selectRegionIdx:
		regionInfo = inputFile[regionIdx].split()
		regionChromo = regionInfo[0]
		regionStart = int(regionInfo[1])
		regionEnd = int(regionInfo[2])
		windowPvalue = list(map(float, regionInfo[5].split(",")))
		windowEnrich = list(map(float, regionInfo[6].split(",")))
		windowNum = len(windowPvalue)

		windowPvalue = np.array(windowPvalue)
		pValueRegionBh = np.array([0.0] * len(windowPvalue))
		nanIdx = np.where(np.isnan(windowPvalue) == True)
		pValueRegionBh[nanIdx] = np.nan
		qValueRegionBh = np.array([np.nan] * len(windowPvalue))

		nonNanIdx = np.where(np.isnan(windowPvalue) == False)
		windowPvalueTemp = windowPvalue[nonNanIdx]


		bhResultTemp = statsmodels.sandbox.stats.multicomp.multipletests(windowPvalueTemp, alpha=globalVars["adjFDR"], method='fdr_bh')
		pValueRegionBhTemp = bhResultTemp[0]
		qValueRegionBhTemp = bhResultTemp[1]
		del bhResultTemp
		for nonIdx in range(len(nonNanIdx[0])):
			pValueRegionBh[nonNanIdx[0][nonIdx]] = pValueRegionBhTemp[nonIdx]
			qValueRegionBh[nonNanIdx[0][nonIdx]] = qValueRegionBhTemp[nonIdx]
		del pValueRegionBhTemp, qValueRegionBhTemp
		pValueRegionBh = np.array(pValueRegionBh)
		qValueRegionBh = np.array(qValueRegionBh)
		selectWindowIdx = np.where(pValueRegionBh == True)[0]

		if len(selectWindowIdx) == 0:
			continue

		### merge if the windows are overlapping and 'enrich' are the same
		idx = selectWindowIdx[0]
		pastStart = regionStart + idx * globalVars["shiftSize2"]
		pastEnd = pastStart + globalVars["binSize2"]
		pastPvalue = windowPvalue[idx]
		pastQvalue = qValueRegionBh[idx]
		pastEnrich = windowEnrich[idx]
		pastPvalueSets = [pastPvalue]
		pastQvalueSets = [pastQvalue]

		selectWindowVector = [regionChromo, pastStart, pastEnd, pastEnrich]

		lastIdx = selectWindowIdx[len(selectWindowIdx)-1]

		if lastIdx == selectWindowIdx[0]:
			selectWindowVector.append(pastPvalue)
			selectWindowVector.append(pastQvalue)

			if lastIdx == (windowNum-1):
				pastEnd = regionEnd
				selectWindowVector[2] = pastEnd

			ctrlRC = []
			for ctrlBWFile in ctrlBWFiles:
				ctrlRC.append(ctrlBWFile.values(selectWindowVector[0], selectWindowVector[1], selectWindowVector[2]))
			ctrlRC = np.array(ctrlRC)
			ctrlRCPosMean = np.mean(ctrlRC, axis=0)

			expRC = []
			for expBWFile in expBWFiles:
				expRC.append(expBWFile.values(selectWindowVector[0], selectWindowVector[1], selectWindowVector[2]))
			expRC = np.array(expRC)
			expRCPosMean = np.mean(expRC, axis=0)

			diffPos = expRCPosMean - ctrlRCPosMean
			diffPos = np.array(diffPos)
			diffPosNanNum = len(np.where(np.isnan(diffPos) == True)[0])
			if diffPosNanNum == len(diffPos):
				continue

			subPeakStarts, subPeakEnds, subPeakDiffs = truncateNan(selectWindowVector[1], selectWindowVector[2], diffPos)
			writePeak(selectWindowVector, subPeakStarts, subPeakEnds, subPeakDiffs, subfile)

			continue


		for idx in selectWindowIdx[1:]:
			currStart = regionStart + idx * globalVars["shiftSize2"]
			currEnd = currStart + globalVars["binSize2"]
			currPvalue = windowPvalue[idx]
			currEnrich = windowEnrich[idx]
			currQvalue = qValueRegionBh[idx]

			if (pastStart <= currStart <= pastEnd) and (pastEnrich == currEnrich):
				selectWindowVector[2] = currEnd
				pastPvalueSets.append(currPvalue)
				pastQvalueSets.append(currQvalue)

			else:
				### End a previous region
				selectWindowVector[2] = pastEnd
				selectWindowVector.append(np.min(pastPvalueSets))
				selectWindowVector.append(np.min(pastQvalueSets))

				ctrlRC = []
				for ctrlBWFile in ctrlBWFiles:
					ctrlRC.append(ctrlBWFile.values(selectWindowVector[0], selectWindowVector[1], selectWindowVector[2]))
				ctrlRC = np.array(ctrlRC)
				ctrlRCPosMean = np.mean(ctrlRC, axis=0)

				expRC = []
				for expBWFile in expBWFiles:
					expRC.append(expBWFile.values(selectWindowVector[0], selectWindowVector[1], selectWindowVector[2]))
				expRC = np.array(expRC)
				expRCPosMean = np.mean(expRC, axis=0)

				diffPos = expRCPosMean - ctrlRCPosMean
				diffPos = np.array(diffPos)
				diffPosNanNum = len(np.where(np.isnan(diffPos) == True)[0])
				if diffPosNanNum == len(diffPos):
					## stark a new region
					pastPvalueSets = [currPvalue]
					pastQvalueSets = [currQvalue]
					selectWindowVector = [regionChromo, currStart, currEnd, currEnrich]

					pastStart = currStart
					pastEnd = currEnd
					pastEnrich = currEnrich
					continue

				subPeakStarts, subPeakEnds, subPeakDiffs = truncateNan(selectWindowVector[1], selectWindowVector[2], diffPos)
				writePeak(selectWindowVector, subPeakStarts, subPeakEnds, subPeakDiffs, subfile)

				### Start a new region
				pastPvalueSets = [currPvalue]
				pastQvalueSets = [currQvalue]
				selectWindowVector = [regionChromo, currStart, currEnd, currEnrich]

			if idx == lastIdx:
				if lastIdx == (windowNum - 1):
					pastEnd = regionEnd
					selectWindowVector[2] = pastEnd

				selectWindowVector.append(np.min(pastPvalueSets))
				selectWindowVector.append(np.min(pastQvalueSets))

				ctrlRC = []
				for ctrlBWFile in ctrlBWFiles:
					ctrlRC.append(ctrlBWFile.values(selectWindowVector[0], selectWindowVector[1], selectWindowVector[2]))
				ctrlRC = np.array(ctrlRC)
				ctrlRCPosMean = np.mean(ctrlRC, axis=0)

				expRC = []
				for expBWFile in expBWFiles:
					expRC.append(expBWFile.values(selectWindowVector[0], selectWindowVector[1], selectWindowVector[2]))
				expRC = np.array(expRC)
				expRCPosMean = np.mean(expRC, axis=0)

				diffPos = expRCPosMean - ctrlRCPosMean
				diffPos = np.array(diffPos)
				diffPosNanNum = len(np.where(np.isnan(diffPos) == True)[0])
				if diffPosNanNum == len(diffPos):
					break

				subPeakStarts, subPeakEnds, subPeakDiffs = truncateNan(selectWindowVector[1], selectWindowVector[2], diffPos)
				writePeak(selectWindowVector, subPeakStarts, subPeakEnds, subPeakDiffs, subfile)

				break

			pastStart = currStart
			pastEnd = currEnd
			pastEnrich = currEnrich

	subfile.close()

	for file in ctrlBWFiles:
		file.close()
	for file in expBWFiles:
		file.close()

	os.remove(inputFilename)

	return subfile.name


def testSubPeak(subpeakDiff, binEnrichType):
	if subpeakDiff == 0:
		return False
	if (binEnrichType == 1) and (subpeakDiff < 0):
		return False
	if (binEnrichType == -1) and (subpeakDiff > 0):
		return False

	return True


def truncateNan(peakStart, peakEnd, diffPos):
	idx = np.where(np.isnan(diffPos) == False)[0]

	if len(idx) == len(diffPos):
		peakDiff = int(np.round(np.mean(diffPos)))
		return [peakStart], [peakEnd], [peakDiff]
	else:
		filteredIdx = []
		nanIdx = np.where(np.isnan(diffPos) == True)[0]

		prevPosIdx = nanIdx[0]
		nanPosStartIdx = prevPosIdx
		i = 1
		while i < len(nanIdx):
			currPosIdx = nanIdx[i]

			if currPosIdx == (prevPosIdx+1):

				if i == (len(nanIdx)-1):
					strechLen = currPosIdx - nanPosStartIdx
					if strechLen >= 20:   ## Save it to the filter out list
						filteredIdx.extend(list(range(nanPosStartIdx, (currPosIdx+1))))
						break
					if nanPosStartIdx == 0:  ## if np.nan stretch  exists in the beginning of the peak region
						filteredIdx.extend(list(range(nanPosStartIdx, (currPosIdx+1))))
						break
					if currPosIdx == (len(diffPos)-1):   ### if np.nan strech exists in the end of the peak region
						filteredIdx.extend(list(range(nanPosStartIdx, (currPosIdx+1))))
						break

				prevPosIdx = currPosIdx
			else:
				#### End a subfiltered
				strechLen = prevPosIdx - nanPosStartIdx
				if strechLen >= 20:   ## Save it to the filter out list
					filteredIdx.extend(list(range(nanPosStartIdx, (prevPosIdx+1))))
				if nanPosStartIdx == 0:
					filteredIdx.extend(list(range(nanPosStartIdx, (prevPosIdx+1))))

				if (i == (len(nanIdx)-1)) and (currPosIdx == (len(diffPos)-1)):
					filteredIdx.append(currPosIdx)
					break

				prevPosIdx = currPosIdx
				nanPosStartIdx = currPosIdx

			i += 1

		##### Get subpeak regions
		if len(filteredIdx) > 0:
			totalPosIdx = np.array(list(range(len(diffPos))))
			totalPosIdx = np.delete(totalPosIdx, filteredIdx)

			subPeakStarts = []
			subPeakEnds = []
			subPeakDiffs = []

			prevPosIdx = totalPosIdx[0]
			subPeakStartIdx = prevPosIdx
			i = 1
			while i < len(totalPosIdx):
				currPosIdx = totalPosIdx[i]

				if i == (len(totalPosIdx)-1):
					subPeakEndIdx = currPosIdx + 1
					subPeakDiff = int(np.round(np.nanmean(diffPos[subPeakStartIdx:subPeakEndIdx])))

					subPeakStarts.append(subPeakStartIdx + peakStart)
					subPeakEnds.append(subPeakEndIdx + peakStart)
					subPeakDiffs.append(subPeakDiff )
					break

				if currPosIdx == (prevPosIdx+1):
					prevPosIdx = currPosIdx
					i = i + 1
				else:
					#### End a region
					subPeakEndIdx = prevPosIdx + 1
					subPeakDiff = int(np.round(np.nanmean(diffPos[subPeakStartIdx:subPeakEndIdx])))
					subPeakStarts.append(subPeakStartIdx + peakStart)
					subPeakEnds.append(subPeakEndIdx + peakStart)
					subPeakDiffs.append(subPeakDiff)

					### Start a new region
					subPeakStartIdx = currPosIdx
					prevPosIdx = currPosIdx
					i = i + 1

			return subPeakStarts, subPeakEnds, subPeakDiffs

		else:
			peakDiff = int(np.round(np.nanmean(diffPos)))
			return [peakStart], [peakEnd], [peakDiff]


def writePeak(selectWindowVector, subPeakStarts, subPeakEnds, subPeakDiffs, subfile):
	windowVector = list(selectWindowVector)
	windowVector[3] = int(windowVector[3])
	for start, end, diff in zip(subPeakStarts, subPeakEnds, subPeakDiffs):
		testResult = testSubPeak(diff, selectWindowVector[3])

		if testResult:
			temp = windowVector.copy()
			temp[1] = start
			temp[2] = end
			temp.append(diff)
			subfile.write('\t'.join([str(x) for x in temp]) + "\n")
