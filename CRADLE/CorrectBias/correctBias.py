import gc
import multiprocessing
import os
import sys
import time

import numpy as np
import pyBigWig
import statsmodels.api as sm

import CRADLE.correctbiasutils as utils
import CRADLE.CorrectBias.regression as reg

from CRADLE.CorrectBias import vari
from CRADLE.CorrectBias.taskCovariates import calculateTaskCovariates
from CRADLE.CorrectBias.trainCovariates import calculateTrainCovariates
from CRADLE.correctbiasutils import vari as commonVari
from CRADLE.logging import timer

RC_PERCENTILE = [0, 20, 40, 60, 80, 90, 92, 94, 96, 98, 99, 100]


def checkArgs(args):
	if ('map' in args.biasType) and (args.mapFile is None) :
		sys.exit("Error: Mappability File is required to correct mappability bias")
	if ('map' in args.biasType) and (args.kmer is None) :
		sys.exit("Error: Kmer is required to correct mappability bias")
	if ('gquad' in args.biasType) and (args.gquadFile is None) :
		sys.exit("Error: Gquadruplex File is required to correct g-gquadruplex bias")


@timer("INITIALIZING PARAMETERS")
def init(args):
	checkArgs(args)
	commonVari.setGlobalVariables(args)
	vari.setGlobalVariables(args)


@timer("Filling Training Sets", 1)
def fillTrainingSets(trainingSetMeta):
	return utils.process(min(11, commonVari.NUMPROCESS), utils.fillTrainingSetMeta, trainingSetMeta, context="fork")


@timer("SELECTING TRAINING SETS")
def selectTrainingSets():
	trainingSetMeta, rc90Percentile, rc99Percentile = utils.getCandidateTrainingSet(
		RC_PERCENTILE,
		commonVari.REGIONS,
		commonVari.CTRLBW_NAMES[0],
		commonVari.OUTPUT_DIR
	)
	vari.HIGHRC = rc90Percentile

	trainingSetMeta = fillTrainingSets(trainingSetMeta)

	trainSet90Percentile, trainSet90To99Percentile = utils.selectTrainingSetFromMeta(trainingSetMeta, rc99Percentile)
	return trainSet90Percentile, trainSet90To99Percentile


def getScaler(trainingSet):

	###### OBTAIN READ COUNTS OF THE FIRST REPLICATE OF CTRLBW.
	ob1 = pyBigWig.open(commonVari.CTRLBW_NAMES[0])

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
	for i in range(1, commonVari.SAMPLE_NUM):
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

	if taskNum < commonVari.CTRLBW_NUM:
		ob2 = pyBigWig.open(commonVari.CTRLBW_NAMES[taskNum])
	else:
		ob2 = pyBigWig.open(commonVari.EXPBW_NAMES[taskNum - commonVari.CTRLBW_NUM])

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


@timer("Calculating Scalers", 1)
def calculateScalers(trainSet90Percentile, trainSet90To99Percentile):
	if vari.I_NORM:
		if (len(trainSet90Percentile) == 0) or (len(trainSet90To99Percentile) == 0):
			scalerResult = getScaler(commonVari.REGIONS)
		else:
			scalerResult = getScaler(trainSet90Percentile + trainSet90To99Percentile)
	else:
		scalerResult = [1] * commonVari.SAMPLE_NUM

	# Sets vari.CTRLSCALER and vari.EXPSCALER
	commonVari.setScaler(scalerResult)

	if vari.I_NORM:
		print("NORMALIZING CONSTANT: ")
		print("CTRLBW: ")
		print(commonVari.CTRLSCALER)
		print("EXPBW: ")
		print(commonVari.EXPSCALER)


@timer("Calculating Covariates", 1)
def calculateTrainingCovariates(trainingSet):
	if len(trainingSet) == 0:
		trainingSet = commonVari.REGIONS

	if len(trainingSet) < commonVari.NUMPROCESS:
		numProcess = len(trainingSet)
	else:
		numProcess = commonVari.NUMPROCESS

	pool = multiprocessing.Pool(numProcess)
	covariates = pool.map_async(calculateTrainCovariates, trainingSet).get()
	pool.close()
	pool.join()
	del pool

	return covariates


@timer("PERFORMING REGRESSION")
def performRegression(trainSetResult1, trainSetResult2):
	scatterplotSamples90Percentile = getScatterplotSamples(trainSetResult1)
	scatterplotSamples90to99Percentile = getScatterplotSamples(trainSetResult2)

	pool = multiprocessing.Pool(2)
	coefResult = pool.starmap_async(
		reg.performRegression,
		[
			(trainSetResult1, scatterplotSamples90Percentile),
			(trainSetResult2, scatterplotSamples90to99Percentile)
		]
	).get()
	pool.close()
	pool.join()

	for name in commonVari.CTRLBW_NAMES:
		fileName = utils.figureFileName(commonVari.OUTPUT_DIR, name)
		regRCReadCounts, regRCFittedValues = coefResult[0][2][name]
		highRCReadCounts, highRCFittedValues = coefResult[1][2][name]
		utils.plot(
			regRCReadCounts, regRCFittedValues,
			highRCReadCounts, highRCFittedValues,
			fileName
		)

	for name in commonVari.EXPBW_NAMES:
		fileName = utils.figureFileName(commonVari.OUTPUT_DIR, name)
		regRCReadCounts, regRCFittedValues = coefResult[0][3][name]
		highRCReadCounts, highRCFittedValues = coefResult[1][3][name]
		utils.plot(
			regRCReadCounts, regRCFittedValues,
			highRCReadCounts, highRCFittedValues,
			fileName
		)

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


@timer("Fitting All Analysis Regions to the Correction Model", 1)
def fitToCorrectionModel():
	tasks = utils.divideGenome(commonVari.REGIONS)

	if len(tasks) < commonVari.NUMPROCESS:
		numProcess = len(tasks)
	else:
		numProcess = commonVari.NUMPROCESS

	pool = multiprocessing.Pool(numProcess)

	# `caluculateTaskCovariates` calls `correctReadCounts`. `correctReadCounts` is the function that
	# fits regions to the correction model.
	resultMeta = pool.map_async(calculateTaskCovariates, tasks).get()
	pool.close()
	pool.join()
	del pool
	gc.collect()

	return resultMeta


def mergeCorrectedBedfilesTobw(args):
	meta = args[0]
	bwHeader = args[1]
	dataInfo = args[2]
	repInfo = args[3]
	observedBWName = args[4]

	signalBWName = '.'.join( observedBWName.rsplit('/', 1)[-1].split(".")[:-1])
	signalBWName = commonVari.OUTPUT_DIR + "/"+ signalBWName + "_corrected.bw"
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


@timer("Merging Temp Files", 1)
def mergeTempFiles(resultBWHeader, resultMeta):
	jobList = []
	for i in range(commonVari.CTRLBW_NUM):
		jobList.append([resultMeta, resultBWHeader, 0, (i+1), commonVari.CTRLBW_NAMES[i]]) # resultMeta, ctrl, rep
	for i in range(commonVari.EXPBW_NUM):
		jobList.append([resultMeta, resultBWHeader, 1, (i+1), commonVari.EXPBW_NAMES[i]]) # resultMeta, ctrl, rep

	pool = multiprocessing.Pool(commonVari.SAMPLE_NUM)
	correctedFileNames = pool.map_async(mergeCorrectedBedfilesTobw, jobList).get()
	pool.close()
	pool.join()

	print("Output File Names: ")
	print(correctedFileNames)


@timer("CORRECTING READ COUNTS")
def correctReadCounts(resultBWHeader):
	resultMeta = fitToCorrectionModel()
	mergeTempFiles(resultBWHeader, resultMeta)

def getScatterplotSamples(covariFiles):
	xNumRows = 0
	for covariFile in covariFiles:
		xNumRows += int(covariFile[1])

	if xNumRows <= 50000:
		return np.array(range(xNumRows))
	else:
		return np.random.choice(np.array(range(xNumRows)), 50000, replace=False)


@timer("GENERATING NORMALIZED OBSERVED BIGWIGS")
def normalizeBigWigs(resultBWHeader):
	normObFileNames = utils.genNormalizedObBWs(
		commonVari.OUTPUT_DIR,
		resultBWHeader,
		commonVari.REGIONS,
		commonVari.CTRLBW_NAMES,
		commonVari.CTRLSCALER,
		commonVari.EXPBW_NAMES,
		commonVari.EXPSCALER
	)

	print("Nomralized observed bigwig file names: ")
	print(normObFileNames)


def run(args):
	startTime = time.perf_counter()

	init(args)

	trainSet90Percentile, trainSet90To99Percentile = selectTrainingSets()

	calculateScalers(trainSet90Percentile, trainSet90To99Percentile)

	trainSetResult1 = calculateTrainingCovariates(trainSet90Percentile)
	trainSetResult2 = calculateTrainingCovariates(trainSet90To99Percentile)
	del trainSet90Percentile, trainSet90To99Percentile
	gc.collect()

	performRegression(trainSetResult1, trainSetResult2)
	del trainSetResult1, trainSetResult2

	resultBWHeader = utils.getResultBWHeader(commonVari.REGIONS, commonVari.CTRLBW_NAMES[0])

	correctReadCounts(resultBWHeader)

	if vari.I_GENERATE_NORM_BW:
		normalizeBigWigs(resultBWHeader)


	print(f"-- RUNNING TIME: {((time.perf_counter() - startTime)/3600)} hour(s)")
