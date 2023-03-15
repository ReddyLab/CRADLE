import multiprocessing
import os.path
import time
from random import sample

import numpy as np
import pyBigWig
import statsmodels.api as sm

from CRADLE.correctbiasutils import vari as commonVari
from CRADLE.correctbiasutils.cython import arraySplit, coalesceSections  # type: ignore

TRAINBIN_SIZE = 1000

RegionDtype = np.dtype([("chromo", "U7"), ("start", "i4"), ("end", "i4"), ("overlap", bool)])


def getRegions(region):
	mergedRegions, regionOverlaps = mergeRegions(region)

	if len(regionOverlaps) == 0:
		nonOverlappingRegions = mergedRegions
	else:
		nonOverlappingRegions = getNonOverlappingRegions(mergedRegions, regionOverlaps)

	combinedRegions = np.concatenate((regionOverlaps, nonOverlappingRegions))
	combinedRegions = combinedRegions[np.lexsort((combinedRegions[:]["start"], combinedRegions[:]["chromo"]))]

	return combinedRegions, nonOverlappingRegions


def mergeRegions(regionBed):
	with open(regionBed, encoding="utf-8") as inputStream:
		inputFileContents = inputStream.readlines()

	regions = []
	for line in inputFileContents:
		temp = line.split()  # chr start end
		regions.append((temp[0], int(temp[1]), int(temp[2]), False))

	regions = np.array(regions, dtype=RegionDtype)
	if len(regions) <= 1:
		return regions, np.array([], dtype=RegionDtype)

	regions = regions[np.lexsort((regions[:]["start"], regions[:]["chromo"]))]

	regionOverlaps = []

	pastChromo, pastStart, pastEnd, _overlap = regions[0]

	mergedRegions = [[pastChromo, pastStart, pastEnd, False]]

	for region in regions[1:]:
		currChromo, currStart, currEnd, _overlap = region

		if (currChromo == pastChromo) and (currStart >= pastStart) and (currStart <= pastEnd):
			if currStart != pastEnd:
				regionOverlaps.append([currChromo, currStart, min(currEnd, pastEnd), True])
			# else: the overlapping position is 0-length

			newEnd = max(currEnd, pastEnd)
			mergedRegions[-1][2] = newEnd
			pastEnd = newEnd
		else:
			mergedRegions.append([currChromo, currStart, currEnd, False])
			pastEnd = currEnd

		pastChromo = currChromo
		pastStart = currStart

	mergedRegions = np.array([tuple(region) for region in mergedRegions], dtype=RegionDtype)
	regionOverlaps = np.array([tuple(region) for region in regionOverlaps], dtype=RegionDtype)
	return mergedRegions, regionOverlaps


def getNonOverlappingRegions(mergedRegions, regionOverlaps):
	nonOverlappingRegions = []
	for region in mergedRegions:
		regionChromo, regionStart, regionEnd, _overlap = region
		chromoRegionOverlaps = np.extract(regionOverlaps[:]["chromo"] == regionChromo, regionOverlaps)  # pylint: disable=W0640

		## overlap Case 1 : Overlap regions that completely cover the region.
		overlapThis = np.extract(
			(chromoRegionOverlaps[:]["start"] <= regionStart) &
			(chromoRegionOverlaps[:]["end"] >= regionEnd),
			chromoRegionOverlaps
		)
		if len(overlapThis) > 0:
			continue

		## overlap Case 2: Overlap regions that only end inside the region
		overlapThis = np.concatenate(
			(overlapThis, np.extract(
				(chromoRegionOverlaps[:]["start"] < regionStart) &
				(chromoRegionOverlaps[:]["end"] > regionStart) &
				(chromoRegionOverlaps[:]["end"] <= regionEnd),
				chromoRegionOverlaps
			))
		)

		## overlap Case 3: Overlap regions that only start inside the region
		overlapThis = np.concatenate(
			(overlapThis, np.extract(
				(chromoRegionOverlaps[:]["start"] >= regionStart) &
				(chromoRegionOverlaps[:]["start"] < regionEnd) &
				(chromoRegionOverlaps[:]["end"] > regionEnd),
				chromoRegionOverlaps
			))
		)

		## overlap Case 4: Overlap regions that start and end inside the region
		overlapThis = np.concatenate(
			(overlapThis, np.extract(
				(chromoRegionOverlaps[:]["start"] > regionStart) &
				(chromoRegionOverlaps[:]["end"] < regionEnd),
				chromoRegionOverlaps
			))
		)

		if len(overlapThis) == 0:
			nonOverlappingRegions.append(region)
			continue

		overlapThis = overlapThis[overlapThis[:]["start"].argsort()]

		currStart = regionStart
		for pos, overlap in enumerate(overlapThis):
			_chr, overlapStart, overlapEnd, _overlap = overlap

			if overlapStart < currStart:
				# Overlap case 2
				currStart = max(overlapEnd, currStart)
			else:
				nonOverlappingRegions.append((regionChromo, currStart, overlapStart, False))
				if overlapEnd < regionEnd:
					# Overlap case 4
					currStart = overlapEnd
				else:
					# Overlap case 3
					break  # the rest of the region is covered by overlaps

			# Last overlap, handle remaining target region
			if (pos == (len(overlapThis)-1)) and (overlapEnd < regionEnd):
				nonOverlappingRegions.append((regionChromo, overlapEnd, regionEnd, False))

	return np.array(nonOverlappingRegions, dtype=RegionDtype)


def selectTrainSet(nonOverlapRegions):
	trainRegionNum = 1_000_000
	totalRegionLen = np.sum(nonOverlapRegions[:]["end"] - nonOverlapRegions[:]["start"])

	if totalRegionLen < trainRegionNum:
		return nonOverlapRegions

	trainRegionNum = trainRegionNum / TRAINBIN_SIZE

	trainSetMeta = []
	for region in nonOverlapRegions:
		chromo, start, end, _overlap = region

		trainNumToSelect = int(trainRegionNum * ((end - start) / totalRegionLen))
		trainSetMeta.append((chromo, start, end, trainNumToSelect))

	numProcess = min(len(trainSetMeta), commonVari.NUMPROCESS)
	task = arraySplit(trainSetMeta, numProcess)
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

	for region in filter(lambda r: r[3] != 0, subregions):
		chromo, start, end, trainNumToSelect = region

		selectedStarts = sample(
			list(np.arange(start, end, TRAINBIN_SIZE)), trainNumToSelect
		)
		for startIdx in selectedStarts:
			subTrainSet.append((chromo, startIdx))

	return subTrainSet


def getScaler(trainSets):
	# Passing ob1Values in to getScalerForEachSample via starmap_async is
	# surprisingly slow and ~quadruples the running time of the whole utility,
	# so its stays a global
	global ob1Values
	with pyBigWig.open(commonVari.CTRLBW_NAMES[0]) as ob1:
		ob1Values = []
		for trainSet in trainSets:
			regionChromo, regionStart = trainSet
			regionEnd = regionStart + TRAINBIN_SIZE

			temp = np.array(ob1.values(regionChromo, regionStart, regionEnd))
			temp[np.isnan(temp) == True] = 0
			ob1Values.extend(list(temp))

	sampleNum = commonVari.CTRLBW_NUM + commonVari.EXPBW_NUM
	task = [(i, trainSets) for i in range(1, sampleNum)]
	###### OBTAIN A SCALER FOR EACH SAMPLE
	numProcess = min(len(task), commonVari.NUMPROCESS)
	pool = multiprocessing.Pool(numProcess)
	scalerResult = pool.starmap_async(getScalerForEachSample, task).get()
	pool.close()
	pool.join()

	return scalerResult


def getScalerForEachSample(taskNum, trainSets):
	if taskNum < commonVari.CTRLBW_NUM:
		ob2 = pyBigWig.open(commonVari.CTRLBW_NAMES[taskNum])
	else:
		ob2 = pyBigWig.open(commonVari.EXPBW_NAMES[taskNum - commonVari.CTRLBW_NUM])

	ob2Values = []
	for trainSet in trainSets:
		regionChromo, regionStart = trainSet
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

	rcPerOnebp = np.zeros(len(combinedRegions))
	maxRCPerOnebp = 0
	for regionIdx, region in enumerate(combinedRegions):
		chromo, start, end, isOverlap = region

		rcArr = [
			np.array(ctrlBW[repIdx].values(chromo, start, end)) / scalerSample[repIdx]
			for repIdx in range(commonVari.CTRLBW_NUM)
		]

		sumRC = np.sum(np.nanmean(rcArr, axis=0))
		rcPerOnebp[regionIdx] = sumRC / (end - start)

		if not isOverlap and (rcPerOnebp[regionIdx] > maxRCPerOnebp):
			maxRCPerOnebp = rcPerOnebp[regionIdx]

	regionScalers = maxRCPerOnebp / rcPerOnebp

	return regionScalers


def getResultBWHeader(combinedRegions, ctrlBWName):
	chromoInData = combinedRegions[:]["chromo"]

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

		starts = np.arange(start, end, dtype=np.int_)
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
	jobList = [[commonVari.OUTPUT_DIR, resultBWHeader, combinedRegions, scalerSample[repIdx], regionScalers, commonVari.CTRLBW_NAMES[repIdx]]
				for repIdx in range(commonVari.CTRLBW_NUM)]

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
