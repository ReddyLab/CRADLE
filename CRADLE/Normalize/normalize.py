import multiprocessing
import os.path
import time
from functools import reduce
from random import sample

import numpy as np
import pyBigWig
import statsmodels.api as sm

from CRADLE.correctbiasutils import vari as commonVari
from CRADLE.correctbiasutils.cython import coalesceSections  # type: ignore

TRAINBIN_SIZE = 1000


def getRegions(region):
	mergedRegions, regionOverlaps = mergeRegions(region)

	if len(regionOverlaps) == 0:
		nonOverlappingRegions = mergedRegions
	else:
		nonOverlappingRegions = getNonOverlappingRegions(mergedRegions, regionOverlaps)

	combinedRegions = []

	for region in regionOverlaps:
		combinedRegions.append([*region, True])

	for region in nonOverlappingRegions:
		combinedRegions.append([*region, False])

	combinedRegions.sort(key=lambda x: x[1])
	combinedRegions.sort(key=lambda x: x[0])

	return combinedRegions, nonOverlappingRegions


def mergeRegions(regionBed):
	with open(regionBed, encoding="utf-8") as inputStream:
		inputFileContents = inputStream.readlines()

	regions = []
	for line in inputFileContents:
		temp = line.split()  # chr start end
		temp[1] = int(temp[1])
		temp[2] = int(temp[2])
		regions.append(temp)

	if len(regions) <= 1:
		return regions, []

	regions.sort(key=lambda x: x[1])
	regions.sort(key=lambda x: x[0])

	regionOverlaps = []

	pastChromo = regions[0][0]
	pastStart = regions[0][1]
	pastEnd = regions[0][2]

	mergedRegions = [[pastChromo, pastStart, pastEnd]]

	for region in regions[1:]:
		currChromo, currStart, currEnd = region

		if (currChromo == pastChromo) and (currStart >= pastStart) and (currStart <= pastEnd):
			if currStart != pastEnd:
				regionOverlaps.append([currChromo, currStart, min(currEnd, pastEnd)])
			# else: the overlapping position is 0-length

			newEnd = max(currEnd, pastEnd)
			mergedRegions[-1][2] = newEnd
			pastEnd = newEnd
		else:
			mergedRegions.append([currChromo, currStart, currEnd])
			pastEnd = currEnd

		pastChromo = currChromo
		pastStart = currStart

	return mergedRegions, regionOverlaps


def getNonOverlappingRegions(mergedRegions, regionOverlaps):
	nonOverlappingRegions = []
	for region in mergedRegions:
		regionChromo, regionStart, regionEnd = region
		chromoRegionOverlaps = list(filter(lambda region: region[0] == regionChromo, regionOverlaps))  # pylint: disable=W0640

		overlapThis = []
		## overlap Case 1 : Overlap regions that completely cover the region.
		overlapThis = list(filter(
			lambda region: region[1] <= regionStart and region[2] >= regionEnd,  # pylint: disable=W0640
			chromoRegionOverlaps
		))
		if len(overlapThis) > 0:
			continue

		## overlap Case 2: Overlap regions that only end inside the region
		overlapThis.extend(
			filter(
				lambda region: region[1] < regionStart and region[2] > regionStart and region[2] <= regionEnd,  # pylint: disable=W0640
				chromoRegionOverlaps
			)
		)

		## overlap Case 3: Overlap regions that only start inside the region
		overlapThis.extend(
			filter(
				lambda region: region[1] >= regionStart and region[1] < regionEnd and region[2] > regionEnd,  # pylint: disable=W0640
				chromoRegionOverlaps
			)
		)

		## overlap Case 4: Overlap regions that start and end inside the region
		overlapThis.extend(
			filter(
				lambda region: region[1] > regionStart and region[2] < regionEnd,  # pylint: disable=W0640
				chromoRegionOverlaps
			)
		)

		if len(overlapThis) == 0:
			nonOverlappingRegions.append(region)
			continue

		overlapThis.sort(key=lambda x: x[1])

		currStart = regionStart
		for pos, overlap in enumerate(overlapThis):
			_chr, overlapStart, overlapEnd = overlap

			if overlapStart < currStart:
				# Overlap case 2
				currStart = max(overlapEnd, currStart)
			else:
				nonOverlappingRegions.append([regionChromo, currStart, overlapStart])
				if overlapEnd < regionEnd:
					# Overlap case 4
					currStart = overlapEnd
				else:
					# Overlap case 3
					break  # the rest of the region is covered by overlaps

			# Last overlap, handle remaining target region
			if (pos == (len(overlapThis)-1)) and (overlapEnd < regionEnd):
				nonOverlappingRegions.append([regionChromo, overlapEnd, regionEnd])

	return nonOverlappingRegions


def selectTrainSet(nonOverlapRegions):
	trainRegionNum = 1_000_000
	totalRegionLen = reduce(lambda acc, region: acc + (region[2] - region[1]), nonOverlapRegions, 0)

	if totalRegionLen < trainRegionNum:
		return nonOverlapRegions

	trainRegionNum = trainRegionNum / TRAINBIN_SIZE

	trainSetMeta = []
	for region in nonOverlapRegions:
		chromo, start, end = region

		trainNumToSelect = int(trainRegionNum * ((end - start) / totalRegionLen))
		trainSetMeta.append([chromo, start, end, trainNumToSelect])

	numProcess = min(len(trainSetMeta), commonVari.NUMPROCESS)
	task = np.array_split(trainSetMeta, numProcess)
	pool = multiprocessing.Pool(numProcess)
	trainSetResults = pool.map_async(getTrainSet, task).get()
	pool.close()
	pool.join()

	trainSet = []
	for result in trainSetResults:
		trainSet.extend(result)

	return trainSet


def getTrainSet(subregions):
	subTrainSet = []

	for region in subregions:
		chromo, start, end, trainNumToSelect = region

		if trainNumToSelect == 0:
			continue

		selectedStarts = sample(
			list(np.arange(start, end, TRAINBIN_SIZE)), trainNumToSelect
		)
		for startIdx in selectedStarts:
			subTrainSet.append([chromo, startIdx])

	return subTrainSet


def getScaler(trainSets):
	with pyBigWig.open(commonVari.CTRLBW_NAMES[0]) as ob1:
		ob1Values = []
		for trainSet in trainSets:
			regionChromo = trainSet[0]
			regionStart = trainSet[1]
			regionEnd = regionStart + TRAINBIN_SIZE

			temp = np.array(ob1.values(regionChromo, regionStart, regionEnd))
			temp[np.isnan(temp) == True] = 0
			ob1Values.extend(list(temp))

	task = []
	sampleNum = commonVari.CTRLBW_NUM + commonVari.EXPBW_NUM
	for i in range(1, sampleNum):
		task.append([i, trainSets, ob1Values])

	###### OBTAIN A SCALER FOR EACH SAMPLE
	numProcess = min(len(task), commonVari.NUMPROCESS)
	pool = multiprocessing.Pool(numProcess)
	scalerResult = pool.starmap_async(getScalerForEachSample, task).get()
	pool.close()
	pool.join()

	return scalerResult


def getScalerForEachSample(taskNum, trainSets, ob1Values):
	if taskNum < commonVari.CTRLBW_NUM:
		ob2 = pyBigWig.open(commonVari.CTRLBW_NAMES[taskNum])
	else:
		ob2 = pyBigWig.open(commonVari.EXPBW_NAMES[taskNum - commonVari.CTRLBW_NUM])

	ob2Values = []
	for trainSet in trainSets:
		regionChromo = trainSet[0]
		regionStart = trainSet[1]
		regionEnd = regionStart + TRAINBIN_SIZE

		temp = np.array(ob2.values(regionChromo, regionStart, regionEnd))
		temp[np.isnan(temp) == True] = 0
		ob2Values.extend(list(temp))
	ob2.close()

	model = sm.OLS(ob2Values, ob1Values).fit()
	scaler = model.params[0]

	return scaler


def getRegionScalers(combinedRegions, scalerSample):
	ctrlBW = [0] * commonVari.CTRLBW_NUM
	for repIdx in range(commonVari.CTRLBW_NUM):
		ctrlBW[repIdx] = pyBigWig.open(commonVari.CTRLBW_NAMES[repIdx])

	rcPerOnebp = [0] * len(combinedRegions)
	maxRCPerOnebp = 0
	for regionIdx, region in enumerate(combinedRegions):
		chromo, start, end, isOverlap = region

		rcArr = []
		for repIdx in range(commonVari.CTRLBW_NUM):
			temp = (
				np.array(ctrlBW[repIdx].values(chromo, start, end))
				/ scalerSample[repIdx]
			)
			rcArr.append(list(temp))

		sumRC = np.sum(np.nanmean(rcArr, axis=0))
		rcPerOnebp[regionIdx] = sumRC / (end - start)

		if not isOverlap and (rcPerOnebp[regionIdx] > maxRCPerOnebp):
			maxRCPerOnebp = rcPerOnebp[regionIdx]

	rcPerOnebp = np.array(rcPerOnebp)
	regionScalers = maxRCPerOnebp / rcPerOnebp

	return regionScalers


def getResultBWHeader(combinedRegions, ctrlBWName):
	chromoInData = np.array(combinedRegions)[:,0]

	chromoInDataUnique = set()
	resultBWHeader = []
	with pyBigWig.open(ctrlBWName) as bw:
		for chromo in chromoInData:
			if chromo in chromoInDataUnique:
				continue
			chromoInDataUnique.add(chromo)
			chromoSize = bw.chroms(chromo)
			resultBWHeader.append((chromo, chromoSize))

	return resultBWHeader


def generateNormalizedBWs(outputDir, bwHeader, combinedRegions, scaler, scalerRegions, observedBWName):
	normObBWName = ".".join(observedBWName.rsplit("/", 1)[-1].split(".")[:-1])
	normObBWName = os.path.join(outputDir, normObBWName + "_normalized.bw")

	normObBW = pyBigWig.open(normObBWName, "w")
	normObBW.addHeader(bwHeader)

	obBW = pyBigWig.open(observedBWName)
	for regionIdx, region in enumerate(combinedRegions):
		chromo, start, end, _overlap = region
		regionScalers = scalerRegions[regionIdx]

		starts = np.arange(start, end, dtype=np.long)
		values = np.array(obBW.values(chromo, start, end))

		idx = np.where((np.isnan(values) == False) & (values > 0))[0]
		starts = starts[idx]
		values = values[idx]
		values = (values / scaler) * regionScalers

		if len(starts) == 0:
			continue

		coalescedSectionCount, startEntries, endEntries, valueEntries = coalesceSections(starts, values)

		normObBW.addEntries([chromo] * coalescedSectionCount, startEntries, ends=endEntries, values=valueEntries)

	normObBW.close()
	obBW.close()

	return normObBWName


def run(args):
	startTime = time.perf_counter()
	###### INITIALIZE PARAMETERS
	print("======  INITIALIZING PARAMETERS .... \n")
	commonVari.setGlobalVariables(args)
	combinedRegions, nonOverlapRegions = getRegions(args.r)

	## 1) Get training set
	trainSet = selectTrainSet(nonOverlapRegions)

	## 2) Normlize samples relative to the first sample of ctrlbw.
	scalerSample = [1]
	scalerSample.extend(getScaler(trainSet))

	print("### Scalers in samples:")
	print(f"   - ctrlbw: {scalerSample[:commonVari.CTRLBW_NUM]}")
	print(f"   - expbw: {scalerSample[commonVari.EXPBW_NUM:]}")

	## 3) Esimate scalers to normalize regions in a sample
	regionScalers = getRegionScalers(combinedRegions, scalerSample)

	## 4) Normalize a sample for different regions
	resultBWHeader = getResultBWHeader(combinedRegions, commonVari.CTRLBW_NAMES[0])
	jobList = []
	for repIdx in range(commonVari.CTRLBW_NUM):
		jobList.append(
			[commonVari.OUTPUT_DIR, resultBWHeader, combinedRegions, scalerSample[repIdx], regionScalers, commonVari.CTRLBW_NAMES[repIdx]]
		)
	for repIdx in range(commonVari.EXPBW_NUM):
		jobList.append(
			[
				commonVari.OUTPUT_DIR,
				resultBWHeader,
				combinedRegions,
				scalerSample[commonVari.CTRLBW_NUM + repIdx],
				regionScalers,
				commonVari.EXPBW_NAMES[repIdx],
			]
		)

	pool = multiprocessing.Pool(min(len(jobList), commonVari.NUMPROCESS))
	normFileNames = pool.starmap_async(generateNormalizedBWs, jobList).get()
	pool.close()
	pool.join()

	print("Normalizing is completed!")
	print("\n")
	print("Nomralized observed bigwig file names: ")
	print(normFileNames)
	print("\n")
	print(f"-- RUNNING TIME: {((time.perf_counter() - startTime) / 3600)} hour(s)")
