import gc
import linecache
import math
import multiprocessing
import os
import sys
import tempfile
import time
import numpy as np
import pyBigWig
import statsmodels.api as sm

from CRADLE.CorrectBias import vari
from CRADLE.CorrectBias import calculateOnebp


def checkArgs(args):
	if ('map' in args.biasType) and (args.mapFile is None) :
		sys.exit("Error: Mappability File is required to correct mappability bias")
	if ('map' in args.biasType) and (args.kmer is None) :
		sys.exit("Error: Kmer is required to correct mappability bias")
	if ('gquad' in args.biasType) and (args.gquadFile is None) :
		sys.exit("Error: Gquadruplex File is required to correct g-gquadruplex bias")

	if args.o is None:
		args.o = os.getcwd() + "/" + "CRADE_correction_result"


def getCandidateTrainSet(rcPercentile):
	global trainBinSize
	global highRC

	trainBinSize = int(vari.BINSIZE) * 2
	if trainBinSize < 1000:
		trainBinSize = 1000

	trainRegionNum = math.pow(10, 6)
	trainRegionNum = trainRegionNum / float(trainBinSize)

	totalBinNum = 0
	for region in vari.REGION:
		regionStart = int(region[1])
		regionEnd = int(region[2])

		numBin = int((regionEnd - regionStart) / trainBinSize)
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

		numBin = int( (regionEnd - regionStart) / trainBinSize)
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

	resultLine = trainBinInfo
	numOfCandiRegion = 0

	resultFile = tempfile.NamedTemporaryFile(mode="w+t", suffix=".txt", dir=vari.OUTPUT_DIR, delete=False)
	'''
	if downLimit == highRC :
		result = []

		for region in vari.REGION:
			regionChromo = region[0]
			regionStart = int(region[1])
			regionEnd = int(region[2])

			numBin = int( (regionEnd - regionStart) / trainBinSize )
			if numBin == 0:
				numBin = 1
				meanValue = np.array(ctrlBW.stats(regionChromo, regionStart, regionEnd, nBins=numBin, type="mean"))[0]

				if meanValue is None:
					continue

				if (meanValue >= downLimit) and (meanValue < upLimit):
					result.append([regionChromo, regionStart, regionEnd, meanValue])

			else:
				regionEnd = numBin * trainBinSize + regionStart
				meanValues = np.array(ctrlBW.stats(regionChromo, regionStart, regionEnd, nBins=numBin, type="mean"))
				pos = np.array(list(range(0, numBin))) * trainBinSize + regionStart

				idx = np.where(meanValues != None)
				meanValues = meanValues[idx]
				pos = pos[idx]

				idx = np.where((meanValues >= downLimit) & (meanValues < upLimit))
				start = pos[idx]
				end = start + trainBinSize
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

		numBin = int( (regionEnd - regionStart) / trainBinSize )
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
			regionEnd = numBin * trainBinSize + regionStart
			meanValues = np.array(ctrlBW.stats(regionChromo, regionStart, regionEnd, nBins=numBin, type="mean"))
			pos = np.array(list(range(0, numBin))) * trainBinSize + regionStart

			idx = np.where(meanValues != None)
			meanValues = meanValues[idx]
			pos = pos[idx]

			idx = np.where((meanValues >= downLimit) & (meanValues < upLimit))

			if len(idx[0]) == 0:
				continue

			start = pos[idx]
			end = start + trainBinSize
			chromoArray = [regionChromo] * len(start)

			numOfCandiRegion = numOfCandiRegion + len(start)
			for i in range(len(start)):
				line = [regionChromo, start[i], start[i] + trainBinSize]
				resultFile.write('\t'.join([str(x) for x in line]) + "\n")

	ctrlBW.close()
	resultFile.close()

	if numOfCandiRegion != 0:
		resultLine.extend([ resultFile.name, numOfCandiRegion ])
		return resultLine

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
			subfileStream = open(candiRegionFile)
			subfile = subfileStream.readlines()

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
			subfileStream = open(candiRegionFile)
			subfile = subfileStream.readlines()

			i = len(subfile) - 1
			while regionNum > 0 and i >= 0:
				temp = subfile[i].split()
				trainSet2.append([ temp[0], int(temp[1]), int(temp[2])])
				i = i - 1
				regionNum = regionNum - 1
		else:
		'''
		if candiRegionNum < regionNum:
			subfileStream = open(candiRegionFile)
			subfile = subfileStream.readlines()

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
		regionChromo = trainSet[i][0]
		regionStart = int(trainSet[i][1])
		regionEnd = int(trainSet[i][2])

		binIdx = 0
		while binIdx < vari.BINSIZE:
			subregionStart = regionStart + binIdx
			subregionEnd = subregionStart + ( int( (regionEnd - subregionStart) / vari.BINSIZE ) * vari.BINSIZE )

			if subregionEnd <= subregionStart:
				break

			numBin = int((subregionEnd - subregionStart) / vari.BINSIZE)

			temp = np.array(ob1.stats(regionChromo, subregionStart, subregionEnd, nBins=numBin, type="mean"))
			idx = np.where(temp == None)
			temp[idx] = 0
			temp = temp.tolist()
			ob1Values.extend(temp)

			binIdx = binIdx + 1
	ob1.close()

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
		regionChromo = trainSet[i][0]
		regionStart = int(trainSet[i][1])
		regionEnd = int(trainSet[i][2])

		binIdx = 0
		while binIdx < vari.BINSIZE:
			subregionStart = regionStart + binIdx
			subregionEnd = subregionStart + ( int( (regionEnd - subregionStart) / vari.BINSIZE ) * vari.BINSIZE )

			if subregionEnd <= subregionStart:
				break

			numBin = int((subregionEnd - subregionStart) / vari.BINSIZE)

			temp = np.array(ob2.stats(regionChromo, subregionStart, subregionEnd, nBins=numBin, type="mean"))
			idx = np.where(temp == None)
			temp[idx] = 0
			temp = temp.tolist()
			ob2Values.extend(temp)

			binIdx = binIdx + 1

	ob2.close()

	model = sm.OLS(ob2Values, ob1Values).fit()
	scaler = model.params[0]

	return scaler


def divideGenome():
	binSize = 50000

	binSize = int( int( binSize / float(vari.BINSIZE) ) * vari.BINSIZE )

	task = []
	for idx in range(len(vari.REGION)):
		regionChromo = vari.REGION[idx][0]
		regionStart = int(vari.REGION[idx][1])
		regionEnd = int(vari.REGION[idx][2])
		regionLen = regionEnd - regionStart

		if regionLen <= binSize :
			task.append(vari.REGION[idx])
		else:
			numBin = float(regionLen) / binSize
			if numBin > int(numBin):
				numBin = int(numBin) + 1

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

	chromoInDataUnique = []
	bw = pyBigWig.open(vari.CTRLBW_NAMES[0])

	resultBWHeader = []
	for i in range(len(chromoInData)):
		chromo = chromoInData[i]
		if chromo in chromoInDataUnique:
			continue
		chromoInDataUnique.extend([chromo])
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
	signalBWName = vari.OUTPUT_DIR + "/"+ signalBWName + "_corrected.bw"
	signalBW = pyBigWig.open(signalBWName, "w")
	signalBW.addHeader(bwHeader)

	for line in meta:
		tempSignalBedName = line[dataInfo][(repInfo-1)]
		tempChrom = line[2]

		if tempSignalBedName is not None:
			tempFileStream = open(tempSignalBedName)
			tempFile = tempFileStream.readlines()

			for i in range(len(tempFile)):
				temp = tempFile[i].split()
				regionStart = int(temp[0])
				regionEnd = int(temp[1])
				regionValue = float(temp[2])

				signalBW.addEntries([tempChrom], [regionStart], ends=[regionEnd], values=[regionValue])
			tempFileStream.close()
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


def run(args):

	startTime = time.time()
	###### INITIALIZE PARAMETERS
	print("======  INITIALIZING PARAMETERS .... \n")
	checkArgs(args)
	vari.setGlobalVariables(args)


	###### SELECT TRAIN SETS
	print("======  SELECTING TRAIN SETS .... \n")
	rcPercentile = [0, 20, 40, 60, 80, 90, 92, 94, 96, 98, 99, 100]
	trainSetMeta = getCandidateTrainSet(rcPercentile)

	print("-- RUNNING TIME of getting trainSetMeta : %s hour(s)" % ((time.time()-startTime)/3600) )

	trainSet1, trainSet2 = selectTrainSetFromMeta(trainSetMeta)
	del trainSetMeta
	print("-- RUNNING TIME of selecting train set from trainSetMeta : %s hour(s)" % ((time.time()-startTime)/3600) )


	###### NORMALIZING READ COUNTS
	print("======  NORMALIZING READ COUNTS ....")
	if vari.I_NORM:
		if (len(trainSet1) == 0) or (len(trainSet2) == 0):
			scalerResult = getScaler( list(vari.REGION) )
		else:
			scalerResult = getScaler( np.concatenate((trainSet1, trainSet2), axis=0).tolist() )
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

	print("-- RUNNING TIME of calculating scalers : %s hour(s)" % ((time.time()-startTime)/3600) )


	###### FITTING TRAIN SETS TO THE CORRECTION MODEL
	print("======  FITTING TRAIN SETS TO THE CORRECTION MODEL ....\n")
	## SELECTING TRAINING SET
	if len(trainSet1) == 0:
		trainSet1 = vari.REGION

	if len(trainSet1) < vari.NUMPROCESS:
		numProcess = len(trainSet1)
	else:
		numProcess = vari.NUMPROCESS

	pool = multiprocessing.Pool(numProcess)
	trainSetResult1 = pool.map_async(calculateOnebp.calculateTrainCovariates, trainSet1).get()
	pool.close()
	pool.join()
	del pool, trainSet1
	gc.collect()
	print("-- RUNNING TIME of calculating 1st Train Cavariates : %s hour(s)" % ((time.time()-startTime)/3600) )


	### trainSet2
	## SELECTING TRAINING SET
	if len(trainSet2) == 0:
		trainSet2 = vari.REGION
	if len(trainSet2) < vari.NUMPROCESS:
		numProcess = len(trainSet2)
	else:
		numProcess = vari.NUMPROCESS

	pool = multiprocessing.Pool(numProcess)
	trainSetResult2 = pool.map_async(calculateOnebp.calculateTrainCovariates, trainSet2).get()
	pool.close()
	pool.join()
	del pool, trainSet2
	gc.collect()

	print("-- RUNNING TIME of calculating 2nd Train Cavariates : %s hour(s)" % ((time.time()-startTime)/3600) )


	## PERFORM REGRESSION
	pool = multiprocessing.Pool(2)
	coefResult = pool.map_async(calculateOnebp.performRegression, [trainSetResult1, trainSetResult2]).get()
	pool.close()
	pool.join()
	del trainSetResult1, trainSetResult2
	gc.collect()

	vari.COEFCTRL = coefResult[0][0]
	vari.COEFEXP = coefResult[0][1]
	vari.COEFCTRL_HIGHRC = coefResult[1][0]
	vari.COEFEXP_HIGHRC = coefResult[1][1]

	print("The order of coefficients:")
	print(vari.COVARI_ORDER)


	print("COEF_CTRL: ")
	print(vari.COEFCTRL)
	print("COEF_EXP: ")
	print(vari.COEFEXP)
	print("COEF_CTRL_HIGHRC: ")
	print(vari.COEFCTRL_HIGHRC)
	print("COEF_EXP_HIGHRC: ")
	print(vari.COEFEXP_HIGHRC)

	print("-- RUNNING TIME of performing regression : %s hour(s)" % ((time.time()-startTime)/3600) )


	###### FITTING THE TEST  SETS TO THE CORRECTION MODEL
	print("======  FITTING ALL THE ANALYSIS REGIONS TO THE CORRECTION MODEL \n")
	task = divideGenome()

	if len(task) < vari.NUMPROCESS:
		numProcess = len(task)
	else:
		numProcess = vari.NUMPROCESS

	pool = multiprocessing.Pool(numProcess)
	resultMeta = pool.map_async(calculateOnebp.calculateTaskCovariates, task).get()
	pool.close()
	pool.join()
	del pool
	gc.collect()

	print("-- RUNNING TIME of calculating Task covariates : %s hour(s)" % ((time.time()-startTime)/3600) )


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

	print("-- RUNNING TIME: %s hour(s)" % ((time.time()-startTime)/3600) )
