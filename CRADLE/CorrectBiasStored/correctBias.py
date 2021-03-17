import gc
import linecache
import math
import multiprocessing
import os
import tempfile
import time
import numpy as np
import py2bit
import pyBigWig
import statsmodels.api as sm

from . import vari
from . import calculateOneBP
from .calculateOneBP import TrainingRegion, TrainingSet

TRAINING_BIN_SIZE = 1000

def getCandidateTrainSet(rcPercentile):
	global highRC

	trainRegionNum = math.pow(10, 6)
	trainRegionNum = trainRegionNum / float(TRAINING_BIN_SIZE)

	totalBinNum = 0
	for region in vari.REGION:
		regionStart = int(region[1])
		regionEnd = int(region[2])

		numBin = int( (regionEnd - regionStart) / TRAINING_BIN_SIZE)
		if numBin == 0:
			numBin = 1
		totalBinNum = totalBinNum + numBin

	if totalBinNum < trainRegionNum:
		trainRegionNum = totalBinNum

	trainRegionNum1 = int(np.round(trainRegionNum * 0.5 / 5))
	trainRegionNum2 = int(np.round(trainRegionNum * 0.5 / 9))
	trainSetMeta = []

	ctrlBW = pyBigWig.open(vari.CTRLBW_NAMES[0])

	meanRC = []
	for region in vari.REGION:
		regionChromo = region[0]
		regionStart = int(region[1])
		regionEnd = int(region[2])

		numBin = int( (regionEnd - regionStart) / TRAINING_BIN_SIZE )
		if numBin == 0:
			numBin = 1
		temp = np.array(ctrlBW.stats(regionChromo, regionStart, regionEnd, nBins=numBin, type="mean"))
		temp = temp[np.where(temp != None)]
		temp = temp[np.where(temp > 0)]

		meanRC.extend(temp.tolist())

	ctrlBW.close()
	meanRC = np.array(meanRC)
	del temp, ctrlBW

	for i in range(5):
		rc1 = int(np.percentile(meanRC, int(rcPercentile[i])))
		rc2 = int(np.percentile(meanRC, int(rcPercentile[i+1])))
		temp = [rc1, rc2, trainRegionNum1]
		trainSetMeta.append(temp)  ## RC criteria1(down), RC criteria2(up), # of bases, candidate regions

	for i in range(6):
		rc1 = int(np.percentile(meanRC, int(rcPercentile[i+5])))
		rc2 = int(np.percentile(meanRC, int(rcPercentile[i+6])))
		if i == 5:
			temp = [rc1, rc2, 3*trainRegionNum2]
			highRC = rc1
		else:
			temp = [rc1, rc2, trainRegionNum2]
		if i == 0:
			vari.HIGHRC = rc1

		trainSetMeta.append(temp)

	del meanRC


	#### Get Candidate Regions
	if vari.NUMPROCESS < 11:
		pool = multiprocessing.Pool(vari.NUMPROCESS)
	else:
		pool = multiprocessing.Pool(11)

	trainSetMeta = pool.map_async(FilltrainSetMeta, trainSetMeta).get()
	pool.close()
	pool.join()


	return trainSetMeta


def FilltrainSetMeta(trainBinInfo):
	downLimit = int(float(trainBinInfo[0]))
	upLimit = int(float(trainBinInfo[1]))

	ctrlBW = pyBigWig.open(vari.CTRLBW_NAMES[0])

	result_line = trainBinInfo
	numOfCandiRegion = 0

	resultFile = tempfile.NamedTemporaryFile(mode="w+t", suffix=".txt", dir=vari.OUTPUT_DIR, delete=False)

	'''
	if downLimit == highRC:
		result = []

		for region in vari.REGION:
			regionChromo = region[0]
			regionStart = int(region[1])
			regionEnd = int(region[2])

			numBin = int( (regionEnd - regionStart) / TRAINING_BIN_SIZE )
			if numBin == 0:
				numBin = 1
				meanValue = np.array(ctrlBW.stats(regionChromo, regionStart, regionEnd, nBins=numBin, type="mean"))[0]
				if meanValue is None:
					continue

				if (meanValue >= downLimit) and (meanValue < upLimit):
					result.append([regionChromo, regionStart, regionEnd, meanValue])

			else:
				regionEnd = numBin * TRAINING_BIN_SIZE + regionStart
				meanValues = np.array(ctrlBW.stats(regionChromo, regionStart, regionEnd, nBins=numBin, type="mean"))
				pos = np.array(list(range(0, numBin))) * TRAINING_BIN_SIZE + regionStart

				idx = np.where(meanValues != None)
				meanValues = meanValues[idx]
				pos = pos[idx]

				idx = np.where((meanValues >= downLimit) & (meanValues < upLimit))
				start = pos[idx]
				end = start + TRAINING_BIN_SIZE
				meanValues = meanValues[idx]
				chromoArray = [regionChromo] * len(start)
				result.extend(np.column_stack((chromoArray, start, end, meanValues)).tolist())

		if len(result) == 0:
			return None

		result = np.array(result)
		result = result[result[:,3].astype(float).astype(int).argsort()][:,0:3].tolist()

		numOfCandiRegion = len(result)
		for i in range(len(result)):
			resultFile.write('\t'.join([str(x) for x in result[i]]) + "\n")
	'''

	for region in vari.REGION:
		regionChromo = region[0]
		regionStart = int(region[1])
		regionEnd = int(region[2])

		numBin = int( (regionEnd - regionStart) / TRAINING_BIN_SIZE )
		if numBin == 0:
			numBin = 1
			meanValue = np.array(ctrlBW.stats(regionChromo, regionStart, regionEnd, nBins=numBin, type="mean"))

			if meanValue is None:
				continue

			if (meanValue >= downLimit) and (meanValue < upLimit):
				line = [regionChromo, regionStart, regionEnd]
				resultFile.write('\t'.join([str(x) for x in line]) + "\n")
				numOfCandiRegion = numOfCandiRegion + 1
		else:
			regionEnd = numBin * TRAINING_BIN_SIZE + regionStart
			meanValues = np.array(ctrlBW.stats(regionChromo, regionStart, regionEnd, nBins=numBin, type="mean"))
			pos = np.array(list(range(0, numBin))) * TRAINING_BIN_SIZE + regionStart

			idx = np.where(meanValues != None)
			meanValues = meanValues[idx]
			pos = pos[idx]

			idx = np.where((meanValues >= downLimit) & (meanValues < upLimit))

			if len(idx[0]) == 0:
				continue

			start = pos[idx]
			end = start + TRAINING_BIN_SIZE
			chromoArray = [regionChromo] * len(start)

			numOfCandiRegion = numOfCandiRegion + len(start)
			for i in range(len(start)):
				line = [regionChromo, start[i], start[i] + TRAINING_BIN_SIZE]
				resultFile.write('\t'.join([str(x) for x in line]) + "\n")

	ctrlBW.close()
	resultFile.close()

	if numOfCandiRegion != 0:
		result_line.extend([ resultFile.name, numOfCandiRegion ])
		return result_line

	os.remove(resultFile.name)
	return None


def selectTrainSetFromMeta(trainSetMeta):

	trainSet1 = []
	trainSet2 = []

	### trainSet1
	for binIdx in range(5):
		if trainSetMeta[binIdx] is None:
			continue

		regionNum = int(trainSetMeta[binIdx][2])
		candiRegionFile = trainSetMeta[binIdx][3]
		candiRegionNum = int(trainSetMeta[binIdx][4])

		if candiRegionNum < regionNum:
			subfile_stream = open(candiRegionFile)
			subfile = subfile_stream.readlines()

			for i in range(len(subfile)):
				temp = subfile[i].split()
				trainSet1.append([ temp[0], int(temp[1]), int(temp[2])])
		else:
			selectRegionIdx = np.random.choice(list(range(candiRegionNum)), regionNum, replace=False)

			for i in range(len(selectRegionIdx)):
				temp = linecache.getline(candiRegionFile, selectRegionIdx[i]+1).split()
				trainSet1.append([ temp[0], int(temp[1]), int(temp[2])])
		os.remove(candiRegionFile)

	### trainSet2
	for binIdx in range(5, len(trainSetMeta)):
		if trainSetMeta[binIdx] is None:
			continue

		downLimit = int(trainSetMeta[binIdx][0])
		regionNum = int(trainSetMeta[binIdx][2])
		candiRegionFile = trainSetMeta[binIdx][3]
		candiRegionNum = int(trainSetMeta[binIdx][4])

		'''
		if downLimit == highRC:
			subfile_stream = open(candiRegionFile)
			subfile = subfile_stream.readlines()

			i = len(subfile) - 1
			while regionNum > 0 and i >= 0:
				temp = subfile[i].split()
				trainSet2.append([ temp[0], int(temp[1]), int(temp[2])])
				i = i - 1
				regionNum = regionNum - 1

		else:
		'''
		if candiRegionNum < regionNum:
			subfile_stream = open(candiRegionFile)
			subfile = subfile_stream.readlines()

			for i in range(len(subfile)):
				temp = subfile[i].split()
				trainSet2.append([ temp[0], int(temp[1]), int(temp[2])])
		else:
			selectRegionIdx = np.random.choice(list(range(candiRegionNum)), regionNum, replace=False)

			for i in range(len(selectRegionIdx)):
				temp = linecache.getline(candiRegionFile, selectRegionIdx[i]+1).split()
				trainSet2.append([ temp[0], int(temp[1]), int(temp[2])])

		os.remove(candiRegionFile)

	return trainSet1, trainSet2


def getScaler(trainSet):
	###### OBTAIN READ COUNTS OF THE FIRST REPLICATE OF CTRLBW.
	global ob1Values

	ob1 = pyBigWig.open(vari.CTRLBW_NAMES[0])

	ob1Values = []
	for i in range(len(trainSet)):
		chromo = trainSet[i][0]
		start = int(trainSet[i][1])
		end = int(trainSet[i][2])

		temp = np.array(ob1.values(chromo, start, end))
		idx = np.where(np.isnan(temp) == True)
		temp[idx] = 0
		temp = temp.tolist()
		ob1Values.extend(temp)

	task = []
	for i in range(1, vari.SAMPLE_NUM):
		task.append([i, trainSet])

	###### OBTAIN A SCALER FOR EACH SAMPLE
	pool = multiprocessing.Pool(len(task))
	scalerResult = pool.map_async(getScalerForEachSample, task).get()
	pool.close()
	pool.join()
	del pool
	gc.collect()

	del ob1Values

	return scalerResult


def getScalerForEachSample(args):
	taskNum = int(args[0])
	trainSet = args[1]

	if taskNum < vari.CTRLBW_NUM:
		ob2 = pyBigWig.open(vari.CTRLBW_NAMES[taskNum])
	else:
		ob2 = pyBigWig.open(vari.EXPBW_NAMES[taskNum-vari.CTRLBW_NUM])

	ob2Values = []
	for i in range(len(trainSet)):
		chromo = trainSet[i][0]
		start = int(trainSet[i][1])
		end = int(trainSet[i][2])

		temp = np.array(ob2.values(chromo, start, end))
		idx = np.where(np.isnan(temp) == True)
		temp[idx] = 0
		temp = temp.tolist()
		ob2Values.extend(temp)
	ob2.close()

	model = sm.OLS(ob2Values, ob1Values).fit()
	scaler = model.params[0]

	return scaler


def divideGenome():
	binSize = 50000

	task = []
	for idx in range(len(vari.REGION)):
		regionChromo = vari.REGION[idx][0]
		regionStart = int(vari.REGION[idx][1])
		regionEnd = int(vari.REGION[idx][2])
		regionLen = regionEnd - regionStart

		if regionLen <= binSize:
			task.append(vari.REGION[idx])
		else:
			numBin = float(regionLen) / binSize
			if numBin > int(numBin):
				numBin = int(numBin) + 1

			numBin = int(numBin)
			for i in range(numBin):
				start = regionStart + i * binSize
				end = regionStart + (i + 1) * binSize
				if i == 0:
					start = regionStart
				if i == (numBin -1):
					end = regionEnd

				task.append([regionChromo, start, end])

	return task


def getResultBWHeader():
	chromoInData = np.array(vari.REGION)[:,0]

	chromoInData_unique = []
	bw = pyBigWig.open(vari.CTRLBW_NAMES[0])

	resultBWHeader = []
	for i in range(len(chromoInData)):
		chromo = chromoInData[i]
		if chromo in chromoInData_unique:
			continue
		chromoInData_unique.extend([chromo])
		chromoSize = bw.chroms(chromo)
		resultBWHeader.append( (chromo, chromoSize) )

	return resultBWHeader


def mergeCorrectedBedfilesTobw(args):
	meta = args[0]
	bwHeader = args[1]
	dataInfo = args[2]
	repInfo = args[3]
	observedBWName = args[4]

	signalBWName = '.'.join( observedBWName.rsplit('/', 1)[-1].split(".")[:-1])
	signalBWName = vari.OUTPUT_DIR + "/" + signalBWName + "_corrected.bw"
	signalBW = pyBigWig.open(signalBWName, "w")
	signalBW.addHeader(bwHeader)

	for line in meta:
		tempSignalBedName = line[dataInfo][(repInfo-1)]
		tempChrom = line[2]

		if tempSignalBedName is not None:
			tempFile_stream = open(tempSignalBedName)
			tempFile = tempFile_stream.readlines()

			for i in range(len(tempFile)):
				temp = tempFile[i].split()
				regionStart = int(temp[0])
				regionEnd = int(temp[1])
				regionValue = float(temp[2])

				signalBW.addEntries([tempChrom], [regionStart], ends=[regionEnd], values=[regionValue])
			tempFile_stream.close()
			os.remove(tempSignalBedName)

	signalBW.close()

	return signalBWName


def generateNormalizedObBWs(args):
	bwHeader = args[0]
	scaler = float(args[1])
	observedBWName = args[2]

	normObBWName = '.'.join( observedBWName.rsplit('/', 1)[-1].split(".")[:-1])
	normObBWName = vari.OUTPUT_DIR + "/" + normObBWName + "_normalized.bw"
	normObBW = pyBigWig.open(normObBWName, "w")
	normObBW.addHeader(bwHeader)

	obBW = pyBigWig.open(observedBWName)

	for regionIdx in range(len(vari.REGION)):
		chromo = vari.REGION[regionIdx][0]
		start = int(vari.REGION[regionIdx][1])
		end = int(vari.REGION[regionIdx][2])

		starts = np.array(range(start, end))
		values = np.array(obBW.values(chromo, start, end))

		idx = np.where( (np.isnan(values) == False) & (values > 0))[0]
		starts = starts[idx]
		values = values[idx]
		values = values / scaler

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

def alignCoordinatesToHDF(faFile, oldTrainingSet, fragLen):
	trainingSet = []
	xRowCount = 0

	for trainingRegion in oldTrainingSet:
		chromo = trainingRegion[0]
		analysisStart = int(trainingRegion[1])
		analysisEnd = int(trainingRegion[2])
		chromoEnd = faFile.chroms(chromo)

		fragStart = analysisStart - fragLen + 1
		fragEnd = analysisEnd + fragLen - 1
		shearStart = fragStart - 2
		shearEnd = fragEnd + 2

		if shearStart < 1:
			shearStart = 1
			fragStart = 3
			analysisStart = max(analysisStart, fragStart)

		if shearEnd > chromoEnd:
			shearEnd = chromoEnd
			fragEnd = shearEnd - 2
			analysisEnd = min(analysisEnd, fragEnd)  # not included

		xRowCount += (analysisEnd - analysisStart)
		trainingSet.append(TrainingRegion(chromo, analysisStart, analysisEnd))

	return TrainingSet(trainingRegions=trainingSet, xRowCount=xRowCount)

def getScatterplotSamples(trainingSet):
	if trainingSet.xRowCount <= 50000:
		return np.array(range(trainingSet.xRowCount))
	else:
		return np.random.choice(np.array(range(trainingSet.xRowCount)), 50000, replace=False)

def run(args):

	start_time = time.time()
	###### INITIALIZE PARAMETERS
	print("======  INITIALIZING PARAMETERS .... \n")
	vari.setGlobalVariables(args)
	covariates = vari.getStoredCovariates(args.biasType, args.covariDir)


	###### SELECT TRAIN SETS
	print("======  SELECTING TRAIN SETS .... \n")
	rcPercentile = [0, 20, 40, 60, 80, 90, 92, 94, 96, 98, 99, 100]
	trainSetMeta = getCandidateTrainSet(rcPercentile)

	print("-- RUNNING TIME of getting trainSetMeta : %s hour(s)" % ((time.time()-start_time)/3600) )

	trainSet90Percentile, trainSet90To99Percentile = selectTrainSetFromMeta(trainSetMeta)
	del trainSetMeta

	print("-- RUNNING TIME of selecting train set from trainSetMeta : %s hour(s)" % ((time.time()-start_time)/3600) )


	###### NORMALIZING READ COUNTS
	print("======  NORMALIZING READ COUNTS ....")
	if vari.I_NORM:
		if (len(trainSet90Percentile) == 0) or (len(trainSet90To99Percentile) == 0):
			scalerResult = getScaler( list(vari.REGION) )
		else:
			scalerResult = getScaler( np.concatenate((trainSet90Percentile, trainSet90To99Percentile), axis=0).tolist() )
	else:
		scalerResult = [1] * vari.SAMPLE_NUM
	vari.setScaler(scalerResult)

	if vari.I_NORM:
		print("NORMALIZING CONSTANT: ")
		print("CTRKBW: ")
		print(vari.CTRLSCALER)
		print("EXPBW: ")
		print(vari.EXPSCALER)
		print("\n\n")

	print("-- RUNNING TIME of calculating scalers : %s hour(s)" % ((time.time()-start_time)/3600) )


	###### FITTING TRAIN SETS TO THE CORRECTION MODEL
	print("======  FITTING TRAIN SETS TO THE CORRECTION MODEL ....\n")

	## PERFORM REGRESSION
	print("======  PERFORMING REGRESSION ....\n")
	pool = multiprocessing.Pool(2)

	if len(trainSet90Percentile) == 0:
		trainSet90Percentile = vari.REGION
	if len(trainSet90To99Percentile) == 0:
		trainSet90To99Percentile = vari.REGION

	with py2bit.open(vari.FA) as faFile:
		trainSet90Percentile = alignCoordinatesToHDF(faFile, trainSet90Percentile, covariates.fragLen)
		trainSet90To99Percentile = alignCoordinatesToHDF(faFile, trainSet90To99Percentile, covariates.fragLen)

	scatterplotSamples90Percentile = getScatterplotSamples(trainSet90Percentile)
	scatterplotSamples90to99Percentile = getScatterplotSamples(trainSet90To99Percentile)


	coefResult = pool.starmap_async(
		calculateOneBP.performRegression,
		[
			[
				trainSet90Percentile, scatterplotSamples90Percentile, covariates, vari.CTRLBW_NAMES, vari.CTRLSCALER,
				vari.EXPBW_NAMES, vari.EXPSCALER, vari.OUTPUT_DIR, "90_precentile"
			],
			[
				trainSet90To99Percentile, scatterplotSamples90to99Percentile, covariates, vari.CTRLBW_NAMES, vari.CTRLSCALER,
				vari.EXPBW_NAMES, vari.EXPSCALER, vari.OUTPUT_DIR, "90_to_99_percentile"
			]
		]
	).get()
	pool.close()
	pool.join()

	del trainSet90Percentile, trainSet90To99Percentile
	gc.collect()

	vari.COEFCTRL = coefResult[0][0]
	vari.COEFEXP = coefResult[0][1]
	vari.COEFCTRL_HIGHRC = coefResult[1][0]
	vari.COEFEXP_HIGHRC = coefResult[1][1]


	print("The order of coefficients:")
	print(covariates.order)

	noNan_idx = [0]
	temp = np.where(np.isnan(covariates.selected) == False)[0] + 1
	temp = temp.tolist()
	noNan_idx.extend(temp)

	print("COEF_CTRL: ")
	print(np.array(vari.COEFCTRL)[:,noNan_idx])
	print("COEF_EXP: ")
	print(np.array(vari.COEFEXP)[:,noNan_idx])
	print("COEF_CTRL_HIGHRC: ")
	print(np.array(vari.COEFCTRL_HIGHRC)[:,noNan_idx])
	print("COEF_EXP_HIGHRC: ")
	print(np.array(vari.COEFEXP_HIGHRC)[:,noNan_idx])

	print("-- RUNNING TIME of performing regression : %s hour(s)" % ((time.time()-start_time)/3600) )


	###### FITTING THE TEST  SETS TO THE CORRECTION MODEL
	print("======  FITTING ALL THE ANALYSIS REGIONS TO THE CORRECTION MODEL \n")
	tasks = divideGenome()
	numProcesses = min(len(tasks), vari.NUMPROCESS)
	taskCount = len(tasks)
	crcArgs = zip(
		tasks,
		[covariates] * taskCount,
		[vari.FA] * taskCount,
		[vari.CTRLBW_NAMES] * taskCount,
		[vari.CTRLSCALER] * taskCount,
		[vari.COEFCTRL] * taskCount,
		[vari.COEFCTRL_HIGHRC] * taskCount,
		[vari.EXPBW_NAMES] * taskCount,
		[vari.EXPSCALER] * taskCount,
		[vari.COEFEXP] * taskCount,
		[vari.COEFEXP_HIGHRC] * taskCount,
		[vari.HIGHRC] * taskCount,
		[vari.MIN_FRAGNUM_FILTER_VALUE] * taskCount,
		[vari.BINSIZE] * taskCount,
		[vari.OUTPUT_DIR] * taskCount
	)

	pool = multiprocessing.Pool(numProcesses)
	resultMeta = pool.starmap_async(calculateOneBP.correctReadCount, crcArgs).get()
	pool.close()
	pool.join()
	del pool
	gc.collect()

	print("-- RUNNING TIME of calculating Task covariates : %s hour(s)" % ((time.time()-start_time)/3600) )


	###### MERGING TEMP FILES
	print("======  MERGING TEMP FILES \n")
	resultBWHeader = getResultBWHeader()

	jobList = []
	for i in range(vari.CTRLBW_NUM):
		jobList.append([resultMeta, resultBWHeader, 0, (i+1), vari.CTRLBW_NAMES[i]]) # resultMeta, ctrl, rep
	for i in range(vari.EXPBW_NUM):
		jobList.append([resultMeta, resultBWHeader, 1, (i+1), vari.EXPBW_NAMES[i]]) # resultMeta, ctrl, rep

	pool = multiprocessing.Pool(vari.SAMPLE_NUM)
	correctedFileNames = pool.map_async(mergeCorrectedBedfilesTobw, jobList).get()
	pool.close()
	pool.join()


	print("Output File Names: ")
	print(correctedFileNames)

	print("======  Completed Correcting Read Counts! \n\n")

	if vari.I_GENERATE_NormBW:
		print("======  Generating normalized observed bigwigs \n\n")
		# copy the first replicate
		from shutil import copyfile
		observedBWName = vari.CTRLBW_NAMES[0]
		normObBWName = '.'.join( observedBWName.rsplit('/', 1)[-1].split(".")[:-1])
		normObBWName = vari.OUTPUT_DIR + "/" + normObBWName + "_normalized.bw"
		copyfile(observedBWName, normObBWName)

		jobList = []
		for i in range(1, vari.CTRLBW_NUM):
			jobList.append([resultBWHeader, vari.CTRLSCALER[i], vari.CTRLBW_NAMES[i]])
		for i in range(vari.EXPBW_NUM):
			jobList.append([resultBWHeader, vari.EXPSCALER[i], vari.EXPBW_NAMES[i]])

		pool = multiprocessing.Pool(vari.SAMPLE_NUM-1)
		normObFileNames = pool.map_async(generateNormalizedObBWs, jobList).get()
		pool.close()
		pool.join()

		print("Nomralized observed bigwig file names: ")
		print(normObFileNames)

	print("-- RUNNING TIME: %s hour(s)" % ((time.time()-start_time)/3600) )
