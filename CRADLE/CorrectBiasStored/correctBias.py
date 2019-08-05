import os
import numpy as np
import multiprocessing
import argparse
import sys
import gc
import math
import random
import pyBigWig
import statsmodels.api as sm
import gc
import time
import linecache
import tempfile

from CRADLE.CorrectBiasStored import vari
from CRADLE.CorrectBiasStored import calculateOnebp



def getCandidateTrainSet(rcPercentile):
	global trainBinSize
	global highRC

	trainBinSize = 1000

	trainRegionNum = math.pow(10, 6)
	trainRegionNum = trainRegionNum / float(trainBinSize)

	totalBinNum = 0
	for region in vari.REGION:
		regionStart = int(region[1])
		regionEnd = int(region[2])

		numBin = int( (regionEnd - regionStart) / trainBinSize)
		if(numBin == 0):
			numBin = 1
		totalBinNum = totalBinNum + numBin

	if(totalBinNum < trainRegionNum):
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

		numBin = int( (regionEnd - regionStart) / trainBinSize )
		if(numBin == 0):
			numBin = 1
		temp = np.array(ctrlBW.stats(regionChromo, regionStart, regionEnd, nBins=numBin, type="mean"))
		temp = temp[np.where(temp!=None)]
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
		if(i == 5):
			temp = [rc1, rc2, 3*trainRegionNum2]
			highRC = rc1
		else:
			temp = [rc1, rc2, trainRegionNum2]
		if(i == 0):
			vari.HIGHRC = rc1

		trainSetMeta.append(temp)

	del meanRC


	#### Get Candidate Regions
	if(vari.NUMPROCESS < 11):
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

	if( downLimit == highRC ):
		result = []

		for region in vari.REGION:
			regionChromo = region[0]
			regionStart = int(region[1])
			regionEnd = int(region[2])

			numBin = int( (regionEnd - regionStart) / trainBinSize )
			if(numBin == 0):
				numBin = 1
				meanValue = np.array(ctrlBW.stats(regionChromo, regionStart, regionEnd, nBins=numBin, type="mean"))[0]
				if(meanValue == None):
					continue

				if( (meanValue >= downLimit) and (meanValue < upLimit)):
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

		if(len(result) == 0):
			return None

		result = np.array(result)
		result = result[result[:,3].astype(float).astype(int).argsort()][:,0:3].tolist()

		numOfCandiRegion = len(result)
		for i in range(len(result)):
			resultFile.write('\t'.join([str(x) for x in result[i]]) + "\n")

	else:
		for region in vari.REGION:
			regionChromo = region[0]
			regionStart = int(region[1])
			regionEnd = int(region[2])

			numBin = int( (regionEnd - regionStart) / trainBinSize )
			if(numBin == 0):
				numBin = 1
				meanValue = np.array(ctrlBW.stats(regionChromo, regionStart, regionEnd, nBins=numBin, type="mean"))

				if(meanValue == None):
					continue

				if( (meanValue >= downLimit) and (meanValue < upLimit)):
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

				if(len(idx[0]) == 0):
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

	if(numOfCandiRegion != 0):
		result_line.extend([ resultFile.name, numOfCandiRegion ])
		return result_line
	else:
		os.remove(resultFile.name)
		return None



def selectTrainSetFromMeta(trainSetMeta):

	trainSet1 = []
	trainSet2 = []

	### trainSet1
	for binIdx in range(5):
		if(trainSetMeta[binIdx] == None):
			continue

		regionNum = int(trainSetMeta[binIdx][2])
		candiRegionFile = trainSetMeta[binIdx][3]
		candiRegionNum = int(trainSetMeta[binIdx][4])

		if(candiRegionNum < regionNum):
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
		if(trainSetMeta[binIdx] == None):
			continue

		downLimit = int(trainSetMeta[binIdx][0])
		regionNum = int(trainSetMeta[binIdx][2])
		candiRegionFile = trainSetMeta[binIdx][3]
		candiRegionNum = int(trainSetMeta[binIdx][4])

		if(downLimit == highRC):
			subfile_stream = open(candiRegionFile)
			subfile = subfile_stream.readlines()

			i = len(subfile) - 1
			while( regionNum > 0 and i >= 0):
				temp = subfile[i].split()
				trainSet2.append([ temp[0], int(temp[1]), int(temp[2])])
				i = i - 1
				regionNum = regionNum - 1
		else:
			if(candiRegionNum < regionNum):
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
		idx = np.where(np.isnan(temp)==True)
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

	if(taskNum < vari.CTRLBW_NUM):
		ob2 = pyBigWig.open(vari.CTRLBW_NAMES[taskNum])
	else:
		ob2 = pyBigWig.open(vari.EXPBW_NAMES[taskNum-vari.CTRLBW_NUM])

	ob2Values = []
	for i in range(len(trainSet)):
		chromo = trainSet[i][0]
		start = int(trainSet[i][1])
		end = int(trainSet[i][2])

		temp = np.array(ob2.values(chromo, start, end))
		idx = np.where(np.isnan(temp)==True)
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

		if(  regionLen <= binSize ):
			task.append(vari.REGION[idx])
		else:
			numBin = float(regionLen) / binSize
			if(numBin > int(numBin)):
				numBin = int(numBin) + 1
			
			for i in range(numBin):
				start = regionStart + i * binSize
				end = regionStart + (i + 1) * binSize
				if(i == 0):
					start = regionStart
				if(i == (numBin -1)):
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
		if(chromo in chromoInData_unique):
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

	signalBWName = observedBWName.rsplit('/', 1)[-1]
	signalBWName = vari.OUTPUT_DIR + "/" + signalBWName[:-3] + "_corrected.bw"
	signalBW = pyBigWig.open(signalBWName, "w")
	signalBW.addHeader(bwHeader)

	for line in meta:
		tempSignalBedName = line[dataInfo][(repInfo-1)]
		tempChrom = line[2]

		if(tempSignalBedName != None):
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




def run(args):

	start_time = time.time()
	###### INITIALIZE PARAMETERS
	print("======  INITIALIZING PARAMETERS .... \n")
	vari.setGlobalVariables(args)

	
	###### SELECT TRAIN SETS
	print("======  SELECTING TRAIN SETS .... \n")
	rcPercentile = [0, 20, 40, 60, 80, 90, 92, 94, 96, 98, 99, 100]		
	trainSetMeta = getCandidateTrainSet(rcPercentile)
	
	print("-- RUNNING TIME of getting trainSetMeta : %s hour(s)" % ((time.time()-start_time)/3600) )
	
	trainSet1, trainSet2 = selectTrainSetFromMeta(trainSetMeta)
	del trainSetMeta

	print("-- RUNNING TIME of selecting train set from trainSetMeta : %s hour(s)" % ((time.time()-start_time)/3600) )


	###### NORMALIZING READ COUNTS
	print("======  NORMALIZING READ COUNTS ....")
	scalerResult = getScaler( np.concatenate((trainSet1, trainSet2), axis=0).tolist() )
	vari.setScaler(scalerResult)
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
	pool = multiprocessing.Pool(2)
	coefResult = pool.map_async(calculateOnebp.performRegression, [trainSet1, trainSet2]).get() 
	pool.close()
	pool.join
	del trainSet1, trainSet2
	gc.collect()

	vari.COEFCTRL = coefResult[0][0]
	vari.COEFEXP = coefResult[0][1]
	vari.COEFCTRL_HIGHRC = coefResult[1][0]
	vari.COEFEXP_HIGHRC = coefResult[1][1]
	print("COEF_CTRL: ")
	print(vari.COEFCTRL)
	print("COEF_EXP: ")
	print(vari.COEFEXP)
	print("COEF_CTRL_HIGHRC: ")
	print(vari.COEFCTRL_HIGHRC)
	print("COEF_EXP_HIGHRC: ")
	print(vari.COEFEXP_HIGHRC)
	
	print("-- RUNNING TIME of performing regression : %s hour(s)" % ((time.time()-start_time)/3600) )


	###### FITTING THE TEST  SETS TO THE CORRECTION MODEL	
	print("======  FITTING ALL THE ANALYSIS REGIONS TO THE CORRECTION MODEL \n")
	task = divideGenome()
	if(len(task) < vari.NUMPROCESS):
		numProcess = len(task)
	else:
		numProcess = vari.NUMPROCESS

	pool = multiprocessing.Pool(numProcess)
	resultMeta = pool.map_async(calculateOnebp.correctReadCount, task).get()
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
		
	print("-- RUNNING TIME: %s hour(s)" % ((time.time()-start_time)/3600) )


