# cython: language_level=3

import os
import tempfile
import warnings
import numpy as np
import pyBigWig
import scipy.stats
import statsmodels.sandbox.stats.multicomp

from CRADLE.CallPeak import vari

cpdef getVariance(region):
	warnings.filterwarnings('ignore', r'All-NaN slice encountered')
	warnings.filterwarnings('ignore', r'Mean of empty slice')
	warnings.filterwarnings('ignore', r'Degrees of freedom <= 0 for slice')

	regionChromo = region[0]
	regionStart = int(region[1])
	regionEnd = int(region[2])

	numBin = int((regionEnd - regionStart) / vari.BINSIZE1)
	if numBin == 0:
		numBin = 1

	totalRC = []
	for rep in range(vari.CTRLBW_NUM):
		bw = pyBigWig.open(vari.CTRLBW_NAMES[rep])
		temp = np.array(bw.stats(regionChromo, regionStart, regionEnd,  type="mean", nBins=numBin))
		temp[np.where(temp == None)] = np.nan
		totalRC.append(temp.tolist())
		bw.close()

	for rep in range(vari.EXPBW_NUM):
		bw = pyBigWig.open(vari.EXPBW_NAMES[rep])
		temp = np.array(bw.stats(regionChromo, regionStart, regionEnd,  type="mean", nBins=numBin))
		temp[np.where(temp == None)] = np.nan
		totalRC.append(temp.tolist())
		bw.close()

	var = np.nanvar(np.array(totalRC), axis=0)
	var = np.array(var)
	idx = np.where(np.isnan(var) == True)
	var = np.delete(var, idx)

	if len(var) > 0:
		return var.tolist()
	else:
		return None

cpdef getRegionCutoff(region):
	warnings.filterwarnings('ignore', r'All-NaN slice encountered')
	warnings.filterwarnings('ignore', r'Mean of empty slice')

	regionChromo = region[0]
	regionStart = int(region[1])
	regionEnd = int(region[2])
	numBin = int((regionEnd - regionStart) / vari.BINSIZE1)
	if numBin == 0:
		numBin = 1

	sampleRC = []
	for rep in range(vari.CTRLBW_NUM):
		bw = pyBigWig.open(vari.CTRLBW_NAMES[rep])
		temp = np.array(bw.stats(regionChromo, regionStart, regionEnd,  type="mean", nBins=numBin))
		temp[np.where(temp == None)] = np.nan
		sampleRC.append(temp.tolist())
		bw.close()

	ctrlMean = np.nanmean(np.array(sampleRC), axis=0)

	sampleRC = []
	for rep in range(vari.EXPBW_NUM):
		bw = pyBigWig.open(vari.EXPBW_NAMES[rep])
		temp = np.array(bw.stats(regionChromo, regionStart, regionEnd,  type="mean", nBins=numBin))
		temp[np.where(temp == None)] = np.nan
		sampleRC.append(temp.tolist())
		bw.close()

	expMean = np.nanmean(np.array(sampleRC), axis=0)
	del sampleRC

	diff = np.array(np.absolute(expMean - ctrlMean))
	idx = np.where(np.isnan(diff) == True)
	diff = np.delete(diff, idx)

	if len(diff) > 0:
		return diff.tolist()
	else:
		return None


cpdef defineRegion(region):
	warnings.filterwarnings('ignore', r'All-NaN slice encountered')
	warnings.filterwarnings('ignore', r'Mean of empty slice')

	analysisChromo = region[0]
	analysisStart = int(region[1])
	analysisEnd = int(region[2])

	#### Number of bin
	binNum = int( (analysisEnd - analysisStart) / vari.BINSIZE1 )
	regionStart = analysisStart
	if binNum == 0:
		regionEnd = analysisEnd
		lastBinExist = False
		binNum = 1
	else:
		if (analysisStart + binNum * vari.BINSIZE1) < analysisEnd:
			lastBinExist = True   # exist!
			regionEnd = regionStart + binNum * vari.BINSIZE1
			lastBinStart = regionEnd
			lastBinEnd = analysisEnd
		else:
			lastBinExist = False
			regionEnd = analysisEnd

	#### ctrlMean
	sampleRC1 = []
	for rep in range(vari.CTRLBW_NUM):
		bw = pyBigWig.open(vari.CTRLBW_NAMES[rep])
		temp = np.array(bw.stats(analysisChromo, regionStart, regionEnd,  type="mean", nBins=binNum))
		temp[np.where(temp == None)] = np.nan
		temp = temp.tolist()

		if lastBinExist:
			lastValue = bw.stats(analysisChromo, lastBinStart, lastBinEnd, type="mean", nBins=1)[0]
			if lastValue is None:
				lastValue = np.nan
			temp.extend([lastValue])

		sampleRC1.append(temp)
		bw.close()

	ctrlMean = np.nanmean( np.array(sampleRC1), axis=0)
	del sampleRC1

	#### expMean
	sampleRC2 = []
	for rep in range(vari.EXPBW_NUM):
		bw = pyBigWig.open(vari.EXPBW_NAMES[rep])
		temp = np.array(bw.stats(analysisChromo, regionStart, regionEnd,  type="mean", nBins=binNum))
		temp[np.where(temp == None)] = np.nan
		temp = temp.tolist()

		if lastBinExist:
			lastValue = bw.stats(analysisChromo, lastBinStart, lastBinEnd, type="mean", nBins=1)[0]
			if lastValue is None:
				lastValue = np.nan
			temp.extend([lastValue])

		sampleRC2.append(temp)
		bw.close()

	expMean = np.nanmean( np.array(sampleRC2), axis=0)
	del sampleRC2

	#### diff
	diff = np.array(expMean - ctrlMean, dtype=np.float64)
	cdef double [:] diffView = diff
	del diff

	cdef int idx = 0
	pastGroupType = -2
	numRegion = 0
	definedRegion = []

	if lastBinExist:
		binNum = binNum + 1

	while idx < binNum:
		if np.isnan(diffView[idx]):
			if pastGroupType == -2:
				idx = idx + 1
				continue
			else:
				definedRegion.append(regionVector)
				numRegion = numRegion + 1
				idx = idx + 1
				pastGroupType = -2
				continue

		if np.absolute(diffView[idx]) > vari.REGION_CUTOFF:
			if diffView[idx] > 0:
				currGroupType = 1   # enriched
			else:
				currGroupType = -1   # repressed
		else:
			currGroupType = 0

		if pastGroupType == -2: # the first region
			regionVector = [
				analysisChromo,
				(analysisStart + vari.BINSIZE1 * idx),
				(analysisStart + vari.BINSIZE1 * idx + vari.BINSIZE1),
				currGroupType
			]
			if idx == (binNum - 1):
				regionVector[2] = analysisEnd
				definedRegion.append(regionVector)
				numRegion = numRegion + 1
				break
			else:
				pastGroupType = currGroupType
				idx = idx + 1
				continue

		if currGroupType != pastGroupType:
			## End a previous region
			regionVector[2] = analysisStart + vari.BINSIZE1 * idx + vari.BINSIZE1 - vari.BINSIZE1
			definedRegion.append(regionVector)
			numRegion = numRegion + 1

			## Start a new reigon
			regionVector = [
				analysisChromo,
				(analysisStart + vari.BINSIZE1 * idx),
				(analysisStart + vari.BINSIZE1 * idx + vari.BINSIZE1),
				currGroupType
			]
		else:
			regionVector[2] = analysisStart + vari.BINSIZE1 * idx + vari.BINSIZE1

		if idx == (binNum-1):
			regionVector[2] = analysisEnd
			definedRegion.append(regionVector)
			numRegion = numRegion + 1
			break

		pastGroupType = currGroupType
		idx = idx + 1


	### variance check
	CTRLBW = [0] * vari.CTRLBW_NUM
	EXPBW = [0] * vari.EXPBW_NUM

	for rep in range(vari.CTRLBW_NUM):
		CTRLBW[rep] = pyBigWig.open(vari.CTRLBW_NAMES[rep])
	for rep in range(vari.EXPBW_NUM):
		EXPBW[rep] = pyBigWig.open(vari.EXPBW_NAMES[rep])

	if len(definedRegion) == 0:
		return None

	definedRegion = restrictRegionLen(definedRegion)

	deleteIdx = []
	for regionIdx in range(len(definedRegion)):
		chromo = definedRegion[regionIdx][0]
		start = int(definedRegion[regionIdx][1])
		end = int(definedRegion[regionIdx][2])

		rc = []
		for rep in range(vari.CTRLBW_NUM):
			rcTemp = np.nanmean(CTRLBW[rep].values(chromo, start, end))
			rc.extend([rcTemp])

		for rep in range(vari.EXPBW_NUM):
			rcTemp = np.nanmean(EXPBW[rep].values(chromo, start, end))
			rc.extend([rcTemp])

		regionVar = np.nanvar(rc)

		if np.isnan(regionVar) == True:
			deleteIdx.extend([regionIdx])
			continue

		thetaIdx = np.max(np.where(vari.FILTER_CUTOFFS < regionVar))
		theta = vari.FILTER_CUTOFFS_THETAS[thetaIdx]

		definedRegion[regionIdx].extend([ theta ])

	definedRegion = np.delete(np.array(definedRegion), deleteIdx, axis=0)
	definedRegion = definedRegion.tolist()


	if len(definedRegion) == 0:
		return None

	subfile = tempfile.NamedTemporaryFile(mode="w+t", dir=vari.OUTPUT_DIR, delete=False)
	for line in definedRegion:
		subfile.write('\t'.join([str(x) for x in line]) + "\n")
	subfile.close()

	for rep in range(vari.CTRLBW_NUM):
		CTRLBW[rep].close()
	for rep in range(vari.EXPBW_NUM):
		EXPBW[rep].close()

	return subfile.name

cpdef restrictRegionLen(definedRegion):
	definedRegion_new = []

	maxRegionLen = vari.BINSIZE1 * 3

	for regionIdx in range(len(definedRegion)):
		start = int(definedRegion[regionIdx][1])
		end = int(definedRegion[regionIdx][2])
		groupType = int(definedRegion[regionIdx][3])
		regionLen = end - start

		if( regionLen > maxRegionLen ):
			newNumBin = int(np.ceil(regionLen / maxRegionLen))
			newRegionSize = int(regionLen / newNumBin)
			chromo = definedRegion[regionIdx][0]

			for newBinIdx in range(newNumBin):
				newStart = start + newBinIdx * newRegionSize
				newEnd = newStart + newRegionSize
				if(newBinIdx == ((newNumBin)-1)):
					newEnd = end
				definedRegion_new.append([chromo, newStart, newEnd, groupType])
		else:
			definedRegion_new.append(definedRegion[regionIdx])

	return definedRegion_new

cpdef doWindowApproach(arg):
	warnings.simplefilter("ignore", category=RuntimeWarning)

	inputFilename = arg
	inputStream = open(inputFilename)
	inputFile = inputStream.readlines()

	subfile = tempfile.NamedTemporaryFile(mode="w+t", dir=vari.OUTPUT_DIR, delete=False)
	simesP = []
	writtenRegionNum = 0

	for regionNum in range(len(inputFile)):
		temp = inputFile[regionNum].split()
		regionChromo = temp[0]
		regionStart = int(temp[1])
		regionEnd = int(temp[2])
		regionTheta = int(temp[4])

		totalRC = []
		for rep in range(vari.CTRLBW_NUM):
			bw = pyBigWig.open(vari.CTRLBW_NAMES[rep])
			temp = bw.values(regionChromo, regionStart, regionEnd)
			totalRC.append(temp)
			bw.close()

		for rep in range(vari.EXPBW_NUM):
			bw = pyBigWig.open(vari.EXPBW_NAMES[rep])
			temp = bw.values(regionChromo, regionStart, regionEnd)
			totalRC.append(temp)
			bw.close()

		totalRC = np.array(totalRC)
		binStartIdx = 0
		analysisEndIdx = regionEnd - regionStart

		windowPvalue = []
		windowEnrich = []

		while (binStartIdx + vari.BINSIZE2) <= analysisEndIdx:
			if (binStartIdx + vari.SHIFTSIZE2 + vari.BINSIZE2) > analysisEndIdx:
				binEndIdx = analysisEndIdx
			else:
				binEndIdx = binStartIdx + vari.BINSIZE2

			rc = np.nanmean(totalRC[:,binStartIdx:binEndIdx], axis=1)

			if len(np.where(np.isnan(rc))[0]) > 0:
				windowPvalue.extend([np.nan])
				windowEnrich.extend([np.nan])
				if (binEndIdx == analysisEndIdx) and (len(windowPvalue) != 0):
					#### calculate a Simes' p value for the reigon
					windowPvalue = np.array(windowPvalue)
					windowPvalueWoNan = windowPvalue[np.isnan(windowPvalue) == False]
					if len(windowPvalueWoNan) == 0:
						break
					rankPvalue = scipy.stats.rankdata(windowPvalueWoNan)
					numWindow = len(windowPvalueWoNan)
					pMerged = np.min((windowPvalueWoNan * numWindow) / rankPvalue)
					simesP.extend([pMerged])

					regionInfo = [regionChromo, regionStart, regionEnd, regionTheta, pMerged]
					subfile.write('\t'.join([str(x) for x in regionInfo]) + "\t")
					subfile.write(','.join([str(x) for x in windowPvalue]) + "\t")
					subfile.write(','.join([str(x) for x in windowEnrich]) + "\n")
					writtenRegionNum = writtenRegionNum + 1

				binStartIdx = binStartIdx + vari.SHIFTSIZE2
				continue

			rc = rc.tolist()
			if rc == vari.ALL_ZERO:
				windowPvalue.extend([np.nan])
				windowEnrich.extend([np.nan])

				if (binEndIdx == analysisEndIdx) and (len(windowPvalue) != 0):
					#### calculate a Simes' p value for the reigon
					windowPvalue = np.array(windowPvalue)
					windowPvalueWoNan = windowPvalue[np.isnan(windowPvalue) == False]
					if len(windowPvalueWoNan) == 0:
						break
					rankPvalue = scipy.stats.rankdata(windowPvalueWoNan)
					numWindow = len(windowPvalueWoNan)
					pMerged = np.min((windowPvalueWoNan * numWindow) / rankPvalue)
					simesP.extend([pMerged])

					regionInfo = [regionChromo, regionStart, regionEnd, regionTheta, pMerged]
					subfile.write('\t'.join([str(x) for x in regionInfo]) + "\t")
					subfile.write(','.join([str(x) for x in windowPvalue]) + "\t")
					subfile.write(','.join([str(x) for x in windowEnrich]) + "\n")
					writtenRegionNum = writtenRegionNum + 1

				binStartIdx = binStartIdx + vari.SHIFTSIZE2
				continue

			windowInfo = doStatTesting(rc)
			pvalue = float(windowInfo[1])
			enrich = int(windowInfo[0])
			windowPvalue.extend([pvalue])
			windowEnrich.extend([enrich])

			if (binEndIdx == analysisEndIdx) and (len(windowPvalue) != 0):
				#### calculate a Simes' p value for the reigon
				windowPvalue = np.array(windowPvalue)
				windowPvalueWoNan = windowPvalue[np.isnan(windowPvalue) == False]
				if len(windowPvalueWoNan) == 0:
					break
				rankPvalue = scipy.stats.rankdata(windowPvalueWoNan)
				numWindow = len(windowPvalueWoNan)
				pMerged = np.min((windowPvalueWoNan * numWindow) / rankPvalue)
				simesP.extend([pMerged])

				regionInfo = [regionChromo, regionStart, regionEnd, regionTheta, pMerged]
				subfile.write('\t'.join([str(x) for x in regionInfo]) + "\t")
				subfile.write(','.join([str(x) for x in windowPvalue]) + "\t")
				subfile.write(','.join([str(x) for x in windowEnrich]) + "\n")
				writtenRegionNum = writtenRegionNum + 1

			binStartIdx = binStartIdx + vari.SHIFTSIZE2

	subfile.close()

	os.remove(inputFilename)

	if writtenRegionNum == 0:
		os.remove(subfile.name)
		return None
	else:
		return subfile.name

cpdef doStatTesting(rc):
	ctrlRC = []
	expRC = []

	for rep in range(vari.CTRLBW_NUM):
		rc[rep] = float(rc[rep])
		ctrlRC.extend([rc[rep]])

	for rep in range(vari.EXPBW_NUM):
		rc[rep + vari.CTRLBW_NUM] = float(rc[rep+vari.CTRLBW_NUM])
		expRC.extend([rc[rep + vari.CTRLBW_NUM] ])

	ctrlVar = np.nanvar(ctrlRC)
	expVar = np.nanvar(expRC)


	if (ctrlVar == 0) and (expVar == 0):
		statistics = float(np.nanmean(expRC) - np.nanmean(ctrlRC))

		pvalue = scipy.stats.norm.cdf(statistics, loc=0, scale=vari.NULL_STD)

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
		welchResult = scipy.stats.ttest_ind(ctrlRC, expRC, equal_var=vari.EQUALVAR)

		if welchResult.statistic > 0:
			enrich = -1
		else:
			enrich = 1

		pvalue = welchResult.pvalue


	windowInfo = [enrich, pvalue]

	return windowInfo

cpdef doFDRprocedure(args):
	inputFilename = args[0]
	selectRegionIdx = args[1]

	inputStream = open(inputFilename)
	inputFile = inputStream.readlines()

	subfile = tempfile.NamedTemporaryFile(mode="w+t", dir=vari.OUTPUT_DIR, delete=False)

	### open bw files to store diff value
	ctrlBW = [0] * vari.CTRLBW_NUM
	expBW = [0] * vari.EXPBW_NUM

	for i in range(vari.CTRLBW_NUM):
		ctrlBW[i] = pyBigWig.open(vari.CTRLBW_NAMES[i])
	for i in range(vari.EXPBW_NUM):
		expBW[i] = pyBigWig.open(vari.EXPBW_NAMES[i])

	for regionIdx in selectRegionIdx:
		regionInfo = inputFile[regionIdx].split()
		regionChromo = regionInfo[0]
		regionStart = int(regionInfo[1])
		regionEnd = int(regionInfo[2])
		windowPvalue = list(map(float, regionInfo[5].split(",")))
		windowEnrich = list(map(float, regionInfo[6].split(",")))
		windowNum = len(windowPvalue)

		windowPvalue = np.array(windowPvalue)
		PValueRegionBh = np.array([0.0] * len(windowPvalue))
		nanIdx = np.where(np.isnan(windowPvalue) == True)
		PValueRegionBh[nanIdx] = np.nan
		QValueRegionBh = np.array([np.nan] * len(windowPvalue))

		nonNanIdx = np.where(np.isnan(windowPvalue) == False)
		windowPvalueTemp = windowPvalue[nonNanIdx]


		bhResultTemp = statsmodels.sandbox.stats.multicomp.multipletests(windowPvalueTemp, alpha=vari.ADJ_FDR, method='fdr_bh')
		PValueRegionBhTemp = bhResultTemp[0]
		QValueRegionBhTemp = bhResultTemp[1]
		del bhResultTemp
		for nonIdx in range(len(nonNanIdx[0])):
			PValueRegionBh[nonNanIdx[0][nonIdx]] = PValueRegionBhTemp[nonIdx]
			QValueRegionBh[nonNanIdx[0][nonIdx]] = QValueRegionBhTemp[nonIdx]
		del PValueRegionBhTemp, QValueRegionBhTemp
		PValueRegionBh = np.array(PValueRegionBh)
		QValueRegionBh = np.array(QValueRegionBh)
		selectWindowIdx = np.where(PValueRegionBh == True)[0]

		if len(selectWindowIdx) == 0:
			continue

		### merge if the windows are overlapping and 'enrich' are the same
		idx = selectWindowIdx[0]
		pastStart = regionStart + idx * vari.SHIFTSIZE2
		pastEnd = pastStart + vari.BINSIZE2
		pastPvalue = windowPvalue[idx]
		pastQvalue = QValueRegionBh[idx]
		pastEnrich = windowEnrich[idx]
		pastPvalueSets = [pastPvalue]
		pastQvalueSets = [pastQvalue]

		selectWindowVector = [regionChromo, pastStart, pastEnd, pastEnrich]

		lastIdx = selectWindowIdx[len(selectWindowIdx)-1]

		if lastIdx == selectWindowIdx[0]:
			selectWindowVector.extend([ pastPvalue ])
			selectWindowVector.extend([ pastQvalue ])

			if lastIdx == (windowNum-1):
				pastEnd = regionEnd
				selectWindowVector[2] = pastEnd

			ctrlRC = []
			for rep in range(vari.CTRLBW_NUM):
				ctrlRC.append(ctrlBW[rep].values(selectWindowVector[0], selectWindowVector[1], selectWindowVector[2]))
			ctrlRC = np.array(ctrlRC)
			ctrlRCPosMean = np.mean(ctrlRC, axis=0)

			expRC = []
			for rep in range(vari.EXPBW_NUM):
				expRC.append(expBW[rep].values(selectWindowVector[0], selectWindowVector[1], selectWindowVector[2]))
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
			currStart = regionStart + idx * vari.SHIFTSIZE2
			currEnd = currStart + vari.BINSIZE2
			currPvalue = windowPvalue[idx]
			currEnrich = windowEnrich[idx]
			currQvalue = QValueRegionBh[idx]

			if (currStart >= pastStart) and (currStart <= pastEnd) and (pastEnrich == currEnrich):
				selectWindowVector[2] = currEnd
				pastPvalueSets.extend([currPvalue])
				pastQvalueSets.extend([currQvalue])

			else:
				### End a previous region
				selectWindowVector[2] = pastEnd
				selectWindowVector.extend([ np.min(pastPvalueSets) ])
				selectWindowVector.extend([ np.min(pastQvalueSets) ])

				ctrlRC = []
				for rep in range(vari.CTRLBW_NUM):
					ctrlRC.append(ctrlBW[rep].values(selectWindowVector[0], selectWindowVector[1], selectWindowVector[2]))
				ctrlRC = np.array(ctrlRC)
				ctrlRCPosMean = np.mean(ctrlRC, axis=0)

				expRC = []
				for rep in range(vari.EXPBW_NUM):
					expRC.append(expBW[rep].values(selectWindowVector[0], selectWindowVector[1], selectWindowVector[2]))
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

				selectWindowVector.extend([ np.min(pastPvalueSets) ])
				selectWindowVector.extend([ np.min(pastQvalueSets) ])

				ctrlRC = []
				for rep in range(vari.CTRLBW_NUM):
					ctrlRC.append(ctrlBW[rep].values(selectWindowVector[0], selectWindowVector[1], selectWindowVector[2]))
				ctrlRC = np.array(ctrlRC)
				ctrlRCPosMean = np.mean(ctrlRC, axis=0)

				expRC = []
				for rep in range(vari.EXPBW_NUM):
					expRC.append(expBW[rep].values(selectWindowVector[0], selectWindowVector[1], selectWindowVector[2]))
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

	for rep in range(vari.CTRLBW_NUM):
		ctrlBW[rep].close()
	for rep in range(vari.EXPBW_NUM):
		expBW[rep].close()

	os.remove(inputFilename)

	return subfile.name

cpdef testSubPeak(subpeakDiff, binEnrichType):
	diff = int(subpeakDiff)

	if diff == 0:
		return False
	if (binEnrichType == 1) and (diff < 0):
		return False
	if (binEnrichType == -1) and (diff > 0):
		return False

	return True

cpdef truncateNan(peakStart, peakEnd, diffPos):
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
				i = i + 1
			else:
				#### End a subfiltered
				strechLen = prevPosIdx - nanPosStartIdx
				if strechLen >= 20:   ## Save it to the filter out list
					filteredIdx.extend(list(range(nanPosStartIdx, (prevPosIdx+1))))
				if nanPosStartIdx == 0:
					filteredIdx.extend(list(range(nanPosStartIdx, (prevPosIdx+1))))

				if (i == (len(nanIdx)-1)) and (currPosIdx == (len(diffPos)-1)):
					filteredIdx.extend([currPosIdx])
					break

				prevPosIdx = currPosIdx
				nanPosStartIdx = currPosIdx
				i = i + 1

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

					subPeakStarts.extend([ subPeakStartIdx + peakStart ])
					subPeakEnds.extend([ subPeakEndIdx + peakStart ])
					subPeakDiffs.extend([ subPeakDiff  ])
					break

				if currPosIdx == (prevPosIdx+1):
					prevPosIdx = currPosIdx
					i = i + 1
				else:
					#### End a region
					subPeakEndIdx = prevPosIdx + 1
					subPeakDiff = int(np.round(np.nanmean(diffPos[subPeakStartIdx:subPeakEndIdx])))
					subPeakStarts.extend([ subPeakStartIdx + peakStart ])
					subPeakEnds.extend([ subPeakEndIdx + peakStart ])
					subPeakDiffs.extend([ subPeakDiff  ])

					### Start a new region
					subPeakStartIdx = currPosIdx
					prevPosIdx = currPosIdx
					i = i + 1

			return subPeakStarts, subPeakEnds, subPeakDiffs

		else:
			peakDiff = int(np.round(np.nanmean(diffPos)))
			return [peakStart], [peakEnd], [peakDiff]

cpdef selectTheta(metaDataName):
	inputFilename = metaDataName
	inputStream = open(inputFilename)
	inputFile = inputStream.readlines()

	totalRegionNumArray = []
	selectRegionNumArray = []

	for thetaIdx in range(len(vari.FILTER_CUTOFFS_THETAS)):
		theta = int(vari.FILTER_CUTOFFS_THETAS[thetaIdx])

		PValueSimes = []

		for subFileIdx in range(len(inputFile)):
			subfileName = inputFile[subFileIdx].split()[0]
			subfileStream = open(subfileName)
			subfileContents = subfileStream.readlines()

			for regionIdx in range(len(subfileContents)):
				line = subfileContents[regionIdx].split()
				regionTheta = int(line[3])
				regionPvalue = float(line[4])

				if np.isnan(regionPvalue):
					continue

				if regionTheta >= theta:
					PValueSimes.extend([ regionPvalue ])

		totalRegionNum =  len(PValueSimes)
		PValueGroupBh = statsmodels.sandbox.stats.multicomp.multipletests(PValueSimes, alpha=vari.FDR, method='fdr_bh')
		selectRegionNum = len(np.where(PValueGroupBh[0])[0])

		totalRegionNumArray.extend([totalRegionNum])
		selectRegionNumArray.extend([selectRegionNum])

	selectRegionNumArray = np.array(selectRegionNumArray)
	maxNum = np.max(selectRegionNumArray)
	idx = np.where(selectRegionNumArray == maxNum)
	idx = idx[0][0]

	return [vari.FILTER_CUTOFFS_THETAS[idx], selectRegionNumArray[idx], totalRegionNumArray[idx]]

cpdef writePeak(selectWindowVector, subPeakStarts, subPeakEnds, subPeakDiffs, subfile):
	for subPeakNum in range(len(subPeakStarts)):
		testResult = testSubPeak(subPeakDiffs[subPeakNum], selectWindowVector[3])

		if testResult:
			temp = list(selectWindowVector)
			temp[1] = subPeakStarts[subPeakNum]
			temp[2] = subPeakEnds[subPeakNum]
			temp[3] = int(temp[3])
			temp.extend([ subPeakDiffs[subPeakNum]  ])
			subfile.write('\t'.join([str(x) for x in temp]) + "\n")