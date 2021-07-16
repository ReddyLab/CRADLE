import gc
import multiprocessing
import os
import sys
import time

import numpy as np
import pyBigWig
import statsmodels.api as sm

import CRADLE.correctbiasutils as utils

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

def getScaler(trainingSet):

	###### OBTAIN READ COUNTS OF THE FIRST REPLICATE OF CTRLBW.
	ob1 = pyBigWig.open(vari.CTRLBW_NAMES[0])

	ob1Values = []
	for region in trainingSet:
		binIdx = 0
		while binIdx < vari.BINSIZE:
			subregionStart = region.start + binIdx
			subregionEnd = subregionStart + ( int( (region.end - subregionStart) / vari.BINSIZE ) * vari.BINSIZE )

			if subregionEnd <= subregionStart:
				break

			numBin = int((subregionEnd - subregionStart) / vari.BINSIZE)

			temp = np.array(ob1.stats(region.chromo, subregionStart, subregionEnd, nBins=numBin, type="mean"))
			idx = np.where(temp == None)
			temp[idx] = 0
			temp = temp.tolist()
			ob1Values.extend(temp)

			binIdx = binIdx + 1
	ob1.close()

	tasks = []
	for i in range(1, vari.SAMPLE_NUM):
		tasks.append([i, trainingSet, ob1Values])

	###### OBTAIN A SCALER FOR EACH SAMPLE
	pool = multiprocessing.Pool(len(tasks))
	scalerResult = pool.starmap_async(getScalerForEachSample, tasks).get()
	pool.close()
	pool.join()
	del pool
	gc.collect()

	return scalerResult

def getScalerForEachSample(taskNum, trainingSet, ob1Values):

	if taskNum < vari.CTRLBW_NUM:
		ob2 = pyBigWig.open(vari.CTRLBW_NAMES[taskNum])
	else:
		ob2 = pyBigWig.open(vari.EXPBW_NAMES[taskNum - vari.CTRLBW_NUM])

	ob2Values = []
	for region in trainingSet:
		binIdx = 0
		while binIdx < vari.BINSIZE:
			subregionStart = region.start + binIdx
			subregionEnd = subregionStart + ( int( (region.end - subregionStart) / vari.BINSIZE ) * vari.BINSIZE )

			if subregionEnd <= subregionStart:
				break

			numBin = int((subregionEnd - subregionStart) / vari.BINSIZE)

			temp = np.array(ob2.stats(region.chromo, subregionStart, subregionEnd, nBins=numBin, type="mean"))
			idx = np.where(temp == None)
			temp[idx] = 0
			temp = temp.tolist()
			ob2Values.extend(temp)

			binIdx = binIdx + 1

	ob2.close()

	model = sm.OLS(ob2Values, ob1Values).fit()
	scaler = model.params[0]

	return scaler

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

def getScatterplotSamples(covariFiles):
	xNumRows = 0
	for covariFile in covariFiles:
		xNumRows += int(covariFile[1])

	if xNumRows <= 50000:
		return np.array(range(xNumRows))
	else:
		return np.random.choice(np.array(range(xNumRows)), 50000, replace=False)

RC_PERCENTILE = [0, 20, 40, 60, 80, 90, 92, 94, 96, 98, 99, 100]

def run(args):

	startTime = time.time()
	###### INITIALIZE PARAMETERS
	print("======  INITIALIZING PARAMETERS .... \n")
	checkArgs(args)
	vari.setGlobalVariables(args)


	###### SELECT TRAIN SETS
	print("======  SELECTING TRAIN SETS .... \n")
	trainingSetMeta, rc90Percentile, rc99Percentile = utils.getCandidateTrainingSet(
		RC_PERCENTILE,
		vari.REGIONS,
		vari.CTRLBW_NAMES[0],
		vari.OUTPUT_DIR
	)
	vari.HIGHRC = rc90Percentile

	trainingSetMeta = utils.process(min(11, vari.NUMPROCESS), utils.fillTrainingSetMeta, trainingSetMeta)

	print("-- RUNNING TIME of getting trainSetMeta : %s hour(s)" % ((time.time()-startTime)/3600) )

	trainSet90Percentile, trainSet90To99Percentile = utils.selectTrainingSetFromMeta(trainingSetMeta, rc99Percentile)
	del trainingSetMeta

	print("-- RUNNING TIME of selecting train set from trainSetMeta : %s hour(s)" % ((time.time()-startTime)/3600) )


	###### NORMALIZING READ COUNTS
	print("======  NORMALIZING READ COUNTS ....")
	if vari.I_NORM:
		if (len(trainSet90Percentile) == 0) or (len(trainSet90To99Percentile) == 0):
			scalerResult = getScaler(vari.REGIONS)
		else:
			scalerResult = getScaler(trainSet90Percentile + trainSet90To99Percentile)
	else:
		scalerResult = [1] * vari.SAMPLE_NUM

	# Sets vari.CTRLSCALER and vari.EXPSCALER
	vari.setScaler(scalerResult)

	if vari.I_NORM:
		print("NORMALIZING CONSTANT: ")
		print("CTRLBW: ")
		print(vari.CTRLSCALER)
		print("EXPBW: ")
		print(vari.EXPSCALER)
		print("\n\n")

	print("-- RUNNING TIME of calculating scalers : %s hour(s)" % ((time.time()-startTime)/3600) )


	###### FITTING TRAIN SETS TO THE CORRECTION MODEL
	print("======  FITTING TRAIN SETS TO THE CORRECTION MODEL ....\n")
	## SELECTING TRAINING SET
	if len(trainSet90Percentile) == 0:
		trainSet90Percentile = vari.REGIONS

	if len(trainSet90Percentile) < vari.NUMPROCESS:
		numProcess = len(trainSet90Percentile)
	else:
		numProcess = vari.NUMPROCESS

	pool = multiprocessing.Pool(numProcess)
	trainSetResult1 = pool.map_async(calculateOnebp.calculateTrainCovariates, trainSet90Percentile).get()
	pool.close()
	pool.join()
	del pool, trainSet90Percentile
	gc.collect()
	print("-- RUNNING TIME of calculating 1st Train Cavariates : %s hour(s)" % ((time.time()-startTime)/3600) )


	### trainSet2
	## SELECTING TRAINING SET
	if len(trainSet90To99Percentile) == 0:
		trainSet90To99Percentile = vari.REGIONS
	if len(trainSet90To99Percentile) < vari.NUMPROCESS:
		numProcess = len(trainSet90To99Percentile)
	else:
		numProcess = vari.NUMPROCESS

	pool = multiprocessing.Pool(numProcess)
	trainSetResult2 = pool.map_async(calculateOnebp.calculateTrainCovariates, trainSet90To99Percentile).get()
	pool.close()
	pool.join()
	del pool, trainSet90To99Percentile
	gc.collect()

	print("-- RUNNING TIME of calculating 2nd Train Cavariates : %s hour(s)" % ((time.time()-startTime)/3600) )


	## PERFORM REGRESSION
	print("======  PERFORMING REGRESSION ....\n")

	scatterplotSamples90Percentile = getScatterplotSamples(trainSetResult1)
	scatterplotSamples90to99Percentile = getScatterplotSamples(trainSetResult2)

	startTime = time.time()
	pool = multiprocessing.Pool(2)
	coefResult = pool.starmap_async(
		calculateOnebp.performRegression,
		[
			(trainSetResult1, scatterplotSamples90Percentile),
			(trainSetResult2, scatterplotSamples90to99Percentile)
		]
	).get()
	pool.close()
	pool.join()

	for name in vari.CTRLBW_NAMES:
		fileName = utils.figureFileName(vari.OUTPUT_DIR, name)
		regRCReadCounts, regRCFittedValues = coefResult[0][2][name]
		highRCReadCounts, highRCFittedValues = coefResult[1][2][name]
		utils.plot(
			regRCReadCounts, regRCFittedValues,
			highRCReadCounts, highRCFittedValues,
			fileName
		)

	for name in vari.EXPBW_NAMES:
		fileName = utils.figureFileName(vari.OUTPUT_DIR, name)
		regRCReadCounts, regRCFittedValues = coefResult[0][3][name]
		highRCReadCounts, highRCFittedValues = coefResult[1][3][name]
		utils.plot(
			regRCReadCounts, regRCFittedValues,
			highRCReadCounts, highRCFittedValues,
			fileName
		)

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
	tasks = utils.divideGenome(vari.REGIONS)

	if len(tasks) < vari.NUMPROCESS:
		numProcess = len(tasks)
	else:
		numProcess = vari.NUMPROCESS

	pool = multiprocessing.Pool(numProcess)
	resultMeta = pool.map_async(calculateOnebp.calculateTaskCovariates, tasks).get()
	pool.close()
	pool.join()
	del pool
	gc.collect()

	print("-- RUNNING TIME of calculating Task covariates : %s hour(s)" % ((time.time()-startTime)/3600) )


	###### MERGING TEMP FILES
	print("======  MERGING TEMP FILES \n")
	resultBWHeader = utils.getResultBWHeader(vari.REGIONS, vari.CTRLBW_NAMES[0])

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
		normObFileNames = utils.genNormalizedObBWs(
			vari.OUTPUT_DIR,
			resultBWHeader,
			vari.REGIONS,
			vari.CTRLBW_NAMES,
			vari.CTRLSCALER,
			vari.EXPBW_NAMES,
			vari.EXPSCALER
		)

		print("Nomralized observed bigwig file names: ")
		print(normObFileNames)

	print("-- RUNNING TIME: %s hour(s)" % ((time.time()-startTime)/3600) )
