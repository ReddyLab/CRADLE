import multiprocessing
import random
import time

import numpy as np
import pyBigWig
import statsmodels.api as sm

from copy import deepcopy

from CRADLE.correctbiasutils import vari as commonVari

TRAINBIN_SIZE = 1000

def setVariables(args):
	setRegions(args.r)

def setRegions(region):
	global REGION
	global overlapREGION
	global nonOverlapRegion
	global REGION_combined

	REGION, overlapREGION = mergeRegions(region)

	if len(overlapREGION) == 0:
		nonOverlapRegion = REGION
	else:
		nonOverlapRegion = excludeOverlapRegion()

	REGION_combined = []
	for i in range(len(overlapREGION)):
		temp = list(deepcopy(overlapREGION[i]))
		temp.extend([True])
		REGION_combined.append(temp)

	for i in range(len(nonOverlapRegion)):
		temp = list(deepcopy(nonOverlapRegion[i]))
		temp.extend([False])
		REGION_combined.append(temp)
	REGION_combined = np.array(REGION_combined)
	REGION_combined = REGION_combined[np.lexsort(( REGION_combined[:,1].astype(int), REGION_combined[:,0])  ) ]

def mergeRegions(region):
	inputFilename = region
	inputStream = open(inputFilename)
	inputFileContents = inputStream.readlines()

	region = []
	for i in range(len(inputFileContents)):
		temp = inputFileContents[i].split()
		temp[1] = int(temp[1])
		temp[2] = int(temp[2])
		region.append(temp)
	inputStream.close()


	region_overlapped = []
	if len(region) > 1:
		region = np.array(region)
		region = region[np.lexsort(( region[:,1].astype(int), region[:,0])  ) ]
		region = region.tolist()

		regionMerged = []

		pos = 0
		pastChromo = region[pos][0]
		pastStart = int(region[pos][1])
		pastEnd = int(region[pos][2])
		regionMerged.append([ pastChromo, pastStart, pastEnd])
		resultIdx = 0

		pos = 1
		while pos < len(region):
			currChromo = region[pos][0]
			currStart = int(region[pos][1])
			currEnd = int(region[pos][2])

			if (currChromo == pastChromo) and (currStart >= pastStart) and (currStart <= pastEnd):
				region_overlapped.append([currChromo, currStart, pastEnd])
				maxEnd = np.max([currEnd, pastEnd])
				regionMerged[resultIdx][2] = maxEnd
				pos = pos + 1
				pastChromo = currChromo
				pastStart = currStart
				pastEnd = maxEnd
			else:
				regionMerged.append([currChromo, currStart, currEnd])
				resultIdx = resultIdx + 1
				pos = pos + 1
				pastChromo = currChromo
				pastStart = currStart
				pastEnd = currEnd
		return np.array(regionMerged), np.array(region_overlapped)
	else:
		return np.array(region), np.array(region_overlapped)

def excludeOverlapRegion():
	regionWoOverlap = []
	for region in REGION:
		regionChromo = region[0]
		regionStart = int(region[1])
		regionEnd = int(region[2])

		overlapThis = []
		## overlap Case 1 : A blacklist region completely covers the region.
		idx = np.where(
			(overlapREGION[:,0] == regionChromo) &
			(overlapREGION[:,1].astype(int) <= regionStart) &
			(overlapREGION[:,2].astype(int) >= regionEnd)
			)[0]
		if len(idx) > 0:
			continue

		## overlap Case 2
		idx = np.where(
			(overlapREGION[:,0] == regionChromo) &
			(overlapREGION[:,2].astype(int) > regionStart) &
			(overlapREGION[:,2].astype(int) <= regionEnd)
			)[0]
		if len(idx) > 0:
			overlapThis.extend( overlapREGION[idx].tolist() )

		## overlap Case 3
		idx = np.where(
			(overlapREGION[:,0] == regionChromo) &
			(overlapREGION[:,1].astype(int) >= regionStart) &
			(overlapREGION[:,1].astype(int) < regionEnd)
			)[0]
		if len(idx) > 0:
			overlapThis.extend( overlapREGION[idx].tolist() )

		if len(overlapThis) == 0:
			regionWoOverlap.append(region)
			continue

		overlapThis = np.array(overlapThis)
		overlapThis = overlapThis[overlapThis[:,1].astype(int).argsort()]
		overlapThis = np.unique(overlapThis, axis=0)
		overlapThis = overlapThis[overlapThis[:,1].astype(int).argsort()]

		currStart = regionStart
		for pos in range(len(overlapThis)):
			overlapStart = int(overlapThis[pos][1])
			overlapEnd = int(overlapThis[pos][2])

			if overlapStart <= regionStart:
				currStart = overlapEnd
			else:
				if currStart == overlapStart:
					currStart = overlapEnd
					continue

				regionWoOverlap.append([ regionChromo, currStart, overlapStart ])
				currStart = overlapEnd

			if (pos == (len(overlapThis)-1)) and (overlapEnd < regionEnd):
				if overlapEnd == regionEnd:
					break
				regionWoOverlap.append([ regionChromo, overlapEnd, regionEnd ])

	return np.array(regionWoOverlap)

def selectTrainSet():
	trainRegionNum = np.power(10, 6)
	totalRegionLen = np.sum(nonOverlapRegion[:,2].astype(int) - nonOverlapRegion[:,1].astype(int))
	if(totalRegionLen < trainRegionNum):
		return nonOverlapRegion

	trainRegionNum = trainRegionNum / float(TRAINBIN_SIZE)

	totalRegionLen = np.sum(nonOverlapRegion[:,2].astype(int) - nonOverlapRegion[:,1].astype(int))
	trainSeteMeta = []
	for regionIdx in range(len(nonOverlapRegion)):
		chromo = nonOverlapRegion[regionIdx][0]
		start = int(nonOverlapRegion[regionIdx][1])
		end = int(nonOverlapRegion[regionIdx][2])

		trainNumToSelect = int(trainRegionNum * ((end - start) / totalRegionLen) )
		trainSeteMeta.append([chromo, start, end, trainNumToSelect])

	numProcess = len(trainSeteMeta)
	if( numProcess > commonVari.NUMPROCESS):
		numProcess = commonVari.NUMPROCESS
	task = np.array_split(trainSeteMeta, numProcess)
	pool = multiprocessing.Pool(numProcess)
	result = pool.map_async(getTrainSet, task).get()
	pool.close()
	pool.join()

	trainSet = []
	for i in range(len(result)):
		trainSet.extend(result[i])

	return trainSet

def getTrainSet(subregions):
	subTrainSet = []

	for regionIdx in range(len(subregions)):
		chromo = subregions[regionIdx][0]
		start = int(subregions[regionIdx][1])
		end = int(subregions[regionIdx][2])
		trainNumToSelect = int(subregions[regionIdx][3])

		if(trainNumToSelect == 0):
			continue

		selectedStarts = random.sample(list(np.arange(start, end, TRAINBIN_SIZE)), trainNumToSelect)
		for i in selectedStarts:
			subTrainSet.append([chromo, i])

	return subTrainSet

def getScaler(trainSet):
	global ob1Values
	ob1 = pyBigWig.open(commonVari.CTRLBW_NAMES[0])

	ob1Values = []
	for i in range(len(trainSet)):
		regionChromo = trainSet[i][0]
		regionStart = int(trainSet[i][1])
		regionEnd = regionStart + TRAINBIN_SIZE

		temp = np.array(ob1.values(regionChromo, regionStart, regionEnd))
		temp[np.isnan(temp) == True] = 0
		ob1Values.extend(list(temp))
	ob1.close()

	task = []
	sampleNum = len(commonVari.CTRLBW_NAMES) + len(commonVari.EXPBW_NAMES)
	for i in range(1, sampleNum):
		task.append([i, trainSet])

	###### OBTAIN A SCALER FOR EACH SAMPLE
	numProcess = len(task)
	if(numProcess > commonVari.NUMPROCESS):
		numProcess = commonVari.NUMPROCESS
	pool = multiprocessing.Pool(numProcess)
	scalerResult = pool.map_async(getScalerForEachSample, task).get()
	pool.close()
	pool.join()

	del ob1Values

	return scalerResult

def getScalerForEachSample(args):
	taskNum = int(args[0])
	trainSet = args[1]

	if taskNum < commonVari.CTRLBW_NUM:
		ob2 = pyBigWig.open(commonVari.CTRLBW_NAMES[taskNum])
	else:
		ob2 = pyBigWig.open(commonVari.EXPBW_NAMES[taskNum - commonVari.CTRLBW_NUM])

	ob2Values = []
	for i in range(len(trainSet)):
		regionChromo = trainSet[i][0]
		regionStart = int(trainSet[i][1])
		regionEnd = regionStart + TRAINBIN_SIZE

		temp = np.array(ob2.values(regionChromo, regionStart, regionEnd))
		temp[np.isnan(temp) == True] = 0
		ob2Values.extend(list(temp))
	ob2.close()

	model = sm.OLS(ob2Values, ob1Values).fit()
	scaler = model.params[0]

	return scaler

def getScalerRegion():
	ctrlBW = [0] * commonVari.CTRLBW_NUM
	for repIdx in range(commonVari.CTRLBW_NUM):
		ctrlBW[repIdx] = pyBigWig.open(commonVari.CTRLBW_NAMES[repIdx])


	rcPerOnebp = [0] * len(REGION_combined)
	maxRCPerOnebp = 0
	for regionIdx in range(len(REGION_combined)):
		chromo = REGION_combined[regionIdx][0]
		start = int(REGION_combined[regionIdx][1])
		end = int(REGION_combined[regionIdx][2])
		I_overlap = REGION_combined[regionIdx][3]

		rcArr = []
		for repIdx in range(commonVari.CTRLBW_NUM):
			temp = np.array(ctrlBW[repIdx].values(chromo, start, end)) / SCALER_SAMPLE[repIdx]
			rcArr.append(list(temp))

		sumRC = np.sum(np.nanmean(rcArr, axis=0))
		rcPerOnebp[regionIdx] = sumRC / (end-start)

		if( (I_overlap == 'False') and (rcPerOnebp[regionIdx] > maxRCPerOnebp)):
			maxRCPerOnebp = rcPerOnebp[regionIdx]

	rcPerOnebp = np.array(rcPerOnebp)
	scaler_diffBacs = maxRCPerOnebp / rcPerOnebp

	return scaler_diffBacs

def getResultBWHeader():
	chromoInData = np.array(REGION_combined)[:,0]

	chromoInDataUnique = []
	bw = pyBigWig.open(commonVari.CTRLBW_NAMES[0])

	resultBWHeader = []
	for i in range(len(chromoInData)):
		chromo = chromoInData[i]
		if chromo in chromoInDataUnique:
			continue
		chromoInDataUnique.extend([chromo])
		chromoSize = bw.chroms(chromo)
		resultBWHeader.append( (chromo, chromoSize) )

	return resultBWHeader

def generateNormalizedBWs(args):
	bwHeader = args[0]
	scaler = float(args[1])
	observedBWName = args[2]

	normObBWName = '.'.join( observedBWName.rsplit('/', 1)[-1].split(".")[:-1])
	normObBWName = commonVari.OUTPUT_DIR + "/" + normObBWName + "_normalized.bw"
	normObBW = pyBigWig.open(normObBWName, "w")
	normObBW.addHeader(bwHeader)

	obBW = pyBigWig.open(observedBWName)
	for regionIdx in range(len(REGION_combined)):
		chromo = REGION_combined[regionIdx][0]
		start = int(REGION_combined[regionIdx][1])
		end = int(REGION_combined[regionIdx][2])
		I_overlap = REGION_combined[regionIdx][3]
		scaler_diffBacs = SCALER_REGION[regionIdx]


		starts = np.array(range(start, end))
		values = np.array(obBW.values(chromo, start, end))

		idx = np.where( (np.isnan(values) == False) & (values > 0))[0]
		starts = starts[idx]
		values = values[idx]
		values = (values / scaler) * scaler_diffBacs

		if len(starts) == 0:
			continue

		## merge positions with the same values
		values = values.astype(int)
		numIdx = len(values)

		idx = 0
		prevStart = starts[idx]
		prevRC = values[idx]
		line = [prevStart, (prevStart+1), prevRC]

		if numIdx == 1:
			normObBW.addEntries([chromo], [int(prevStart)], ends=[int(prevStart+1)], values=[float(prevRC)])
		else:
			idx = 1
			while idx < numIdx:
				currStart = starts[idx]
				currRC = values[idx]

				if (currStart == (prevStart + 1)) and (currRC == prevRC):
					line[1] = currStart + 1
					prevStart = currStart
					prevRC = currRC
					idx = idx + 1
				else:
					### End a current line
					normObBW.addEntries([chromo], [int(line[0])], ends=[int(line[1])], values=[float(line[2])])

					### Start a new line
					line = [currStart, (currStart+1), currRC]
					prevStart = currStart
					prevRC = currRC
					idx = idx + 1

				if idx == numIdx:
					normObBW.addEntries([chromo], [int(line[0])], ends=[int(line[1])], values=[float(line[2])])
					break

	normObBW.close()
	obBW.close()

	return normObBWName

def run(args):
	startTime = time.time()
	###### INITIALIZE PARAMETERS
	print("======  INITIALIZING PARAMETERS .... \n")
	commonVari.setGlobalVariables(args)
	setVariables(args)

	## 1) Get training set
	trainSet = selectTrainSet()

	## 2) Normlize samples relative to the first sample of ctrlbw.
	global SCALER_SAMPLE
	SCALER_SAMPLE = [1]
	SCALER_SAMPLE.extend(getScaler(trainSet))

	print("### Scalers in samples:")
	print("   - ctrlbw: %s" % SCALER_SAMPLE[:commonVari.CTRLBW_NUM] )
	print("   - expbw: %s" % SCALER_SAMPLE[commonVari.EXPBW_NUM:] )


	## 3) Esimate scalers to normalize regions in a sample
	global SCALER_REGION
	SCALER_REGION = getScalerRegion()

	## 4) Normalize a sample for different regions
	resultBWHeader = getResultBWHeader()
	jobList = []
	for repIdx in range(commonVari.CTRLBW_NUM):
		jobList.append([resultBWHeader, SCALER_SAMPLE[repIdx], commonVari.CTRLBW_NAMES[repIdx]])
	for repIdx in range(commonVari.EXPBW_NUM):
		jobList.append([resultBWHeader, SCALER_SAMPLE[commonVari.CTRLBW_NUM + repIdx], commonVari.EXPBW_NAMES[repIdx]])

	if(commonVari.NUMPROCESS < len(jobList)):
		pool = multiprocessing.Pool(commonVari.NUMPROCESS)
	else:
		pool = multiprocessing.Pool(len(jobList))

	normFileNames = pool.map_async(generateNormalizedBWs, jobList).get()
	pool.close()
	pool.join()

	print("Normalizing is completed!")
	print("\n")
	print("Nomralized observed bigwig file names: ")
	print(normFileNames)
	print("\n")
	print("-- RUNNING TIME: %s hour(s)" % ((time.time()-startTime)/3600) )


