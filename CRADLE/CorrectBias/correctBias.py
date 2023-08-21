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
	commonGlobals = commonVari.setGlobalVariables(args)
	cbGlobals = vari.setGlobalVariables(args)

	return commonGlobals | cbGlobals


@timer("Filling Training Sets", 1)
def fillTrainingSets(trainingSetMeta, globalVars):
	return utils.process(min(11, globalVars["numprocess"]), utils.fillTrainingSetMeta, trainingSetMeta, context="fork")


@timer("SELECTING TRAINING SETS")
def selectTrainingSets(globalVars):
	trainingSetMeta, rc90Percentile, rc99Percentile = utils.getCandidateTrainingSet(
		RC_PERCENTILE,
		globalVars["regions"],
		globalVars["ctrlbwNames"][0],
		globalVars["outputDir"]
	)
	globalVars["highrc"] = rc90Percentile

	trainingSetMeta = fillTrainingSets(trainingSetMeta, globalVars)

	trainSet90Percentile, trainSet90To99Percentile = utils.selectTrainingSetFromMeta(trainingSetMeta, rc99Percentile)
	return trainSet90Percentile, trainSet90To99Percentile


def getScaler(trainingSet, globalVars):

	###### OBTAIN READ COUNTS OF THE FIRST REPLICATE OF CTRLBW.
	ob1 = pyBigWig.open(globalVars["ctrlbwNames"][0])

	ob1Values = []
	for region in trainingSet:
		binIdx = 0
		while binIdx < globalVars["binSize"]:
			subregionStart = region.start + binIdx
			subregionEnd = subregionStart + ( int( (region.end - subregionStart) / globalVars["binSize"] ) * globalVars["binSize"] )

			if subregionEnd <= subregionStart:
				break

			numBin = int((subregionEnd - subregionStart) / globalVars["binSize"])

			temp = np.array(ob1.stats(region.chromo, subregionStart, subregionEnd, nBins=numBin, type="mean"))
			idx = np.where(temp == None)
			temp[idx] = 0
			temp = temp.tolist()
			ob1Values.extend(temp)

			binIdx = binIdx + 1
	ob1.close()

	tasks = []
	for i in range(1, globalVars["sampleNum"]):
		tasks.append([i, trainingSet, ob1Values])

	###### OBTAIN A SCALER FOR EACH SAMPLE
	pool = multiprocessing.Pool(len(tasks))
	scalerResult = pool.starmap_async(getScalerForEachSample, tasks).get()
	pool.close()
	pool.join()
	del pool
	gc.collect()

	return scalerResult


def getScalerForEachSample(taskNum, trainingSet, ob1Values, globalVars):

	if taskNum < globalVars["ctrlbwNum"]:
		ob2 = pyBigWig.open(globalVars["ctrlbwNames"][taskNum])
	else:
		ob2 = pyBigWig.open(globalVars["expbwNames"][taskNum - globalVars["ctrlbwNum"]])

	ob2Values = []
	for region in trainingSet:
		binIdx = 0
		while binIdx < globalVars["binSize"]:
			subregionStart = region.start + binIdx
			subregionEnd = subregionStart + ( int( (region.end - subregionStart) / globalVars["binSize"] ) * globalVars["binSize"] )

			if subregionEnd <= subregionStart:
				break

			numBin = int((subregionEnd - subregionStart) / globalVars["binSize"])

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
def calculateScalers(trainSet90Percentile, trainSet90To99Percentile, globalVars):
	if globalVars["i_norm"]:
		if (len(trainSet90Percentile) == 0) or (len(trainSet90To99Percentile) == 0):
			scalerResult = getScaler(globalVars["regions"], globalVars)
		else:
			scalerResult = getScaler(trainSet90Percentile + trainSet90To99Percentile, globalVars)
	else:
		scalerResult = [1] * globalVars["sampleNum"]

	globalVars["ctrlScaler"] = [0] * globalVars["ctrlbwNum"]
	globalVars["expScaler"] = [0] * globalVars["expbwNum"]
	globalVars["ctrlScaler"][0] = 1

	for i in range(1, globalVars["ctrlbwNum"]):
		globalVars["ctrlScaler"][i] = scalerResult[i-1]

	for i in range(globalVars["expbwNum"]):
		globalVars["expScaler"][i] = scalerResult[i+globalVars["ctrlbwNum"]-1]

	if globalVars["i_norm"]:
		print("NORMALIZING CONSTANT: ")
		print("CTRLBW: ")
		print(globalVars["ctrlScaler"])
		print("EXPBW: ")
		print(globalVars["expScaler"])


@timer("Calculating Covariates", 1)
def calculateTrainingCovariates(trainingSet, globalVars):
	if len(trainingSet) == 0:
		trainingSet = globalVars["regions"]

	tasks = [(region, globalVars) for region in trainingSet]

	numProcess = min(len(trainingSet), globalVars["numprocess"])
	pool = multiprocessing.Pool(numProcess)
	covariates = pool.starmap_async(calculateTrainCovariates, tasks).get()
	pool.close()
	pool.join()
	del pool

	return covariates


@timer("PERFORMING REGRESSION")
def performRegression(trainSetResult1, trainSetResult2, globalVars):
	scatterplotSamples90Percentile = getScatterplotSamples(trainSetResult1)
	scatterplotSamples90to99Percentile = getScatterplotSamples(trainSetResult2)

	pool = multiprocessing.Pool(2)
	coefResult = pool.starmap_async(
		reg.performRegression,
		[
			(trainSetResult1, scatterplotSamples90Percentile, globalVars),
			(trainSetResult2, scatterplotSamples90to99Percentile, globalVars)
		]
	).get()
	pool.close()
	pool.join()

	for name in globalVars["ctrlbwNames"]:
		fileName = utils.figureFileName(globalVars["output_dir"], name)
		regRCReadCounts, regRCFittedValues = coefResult[0][2][name]
		highRCReadCounts, highRCFittedValues = coefResult[1][2][name]
		utils.plot(
			regRCReadCounts, regRCFittedValues,
			highRCReadCounts, highRCFittedValues,
			fileName
		)

	for name in globalVars["expbwNames"]:
		fileName = utils.figureFileName(globalVars["output_dir"], name)
		regRCReadCounts, regRCFittedValues = coefResult[0][3][name]
		highRCReadCounts, highRCFittedValues = coefResult[1][3][name]
		utils.plot(
			regRCReadCounts, regRCFittedValues,
			highRCReadCounts, highRCFittedValues,
			fileName
		)

	gc.collect()

	globalVars["coefctrl"] = coefResult[0][0]
	globalVars["coefexp"] = coefResult[0][1]
	globalVars["coefctrlHighrc"] = coefResult[1][0]
	globalVars["coefexpHighrc"] = coefResult[1][1]

	print("The order of coefficients:")
	print(globalVars["covari_order"])

	print("COEF_CTRL: ")
	print(globalVars["coefctrl"])
	print("COEF_EXP: ")
	print(globalVars["coefexp"])
	print("COEF_CTRL_HIGHRC: ")
	print(globalVars["coefctrlHighrc"])
	print("COEF_EXP_HIGHRC: ")
	print(globalVars["coefexpHighrc"])


@timer("Fitting All Analysis Regions to the Correction Model", 1)
def fitToCorrectionModel(globalVars):
	regions = utils.divideGenome(globalVars["regions"])

	tasks = [(region, globalVars) for region in regions]

	numProcess = min(len(tasks), globalVars["numprocess"])
	pool = multiprocessing.Pool(numProcess)
	# `caluculateTaskCovariates` calls `correctReadCounts`. `correctReadCounts` is the function that
	# fits regions to the correction model.
	resultMeta = pool.starmap_async(calculateTaskCovariates, tasks).get()
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
	globalVars = args[5]

	signalBWName = '.'.join( observedBWName.rsplit('/', 1)[-1].split(".")[:-1])
	signalBWName = globalVars["output_dir"] + "/"+ signalBWName + "_corrected.bw"
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
def mergeTempFiles(resultBWHeader, resultMeta, globalVars):
	jobList = []
	for i in range(globalVars["ctrlbwNum"]):
		jobList.append([resultMeta, resultBWHeader, 0, (i+1), globalVars["ctrlbwNames"][i], globalVars]) # resultMeta, ctrl, rep
	for i in range(globalVars["expbwNum"]):
		jobList.append([resultMeta, resultBWHeader, 1, (i+1), globalVars["expbwNames"][i], globalVars]) # resultMeta, ctrl, rep

	pool = multiprocessing.Pool(globalVars["sampleNum"])
	correctedFileNames = pool.map_async(mergeCorrectedBedfilesTobw, jobList).get()
	pool.close()
	pool.join()

	print("Output File Names: ")
	print(correctedFileNames)


@timer("CORRECTING READ COUNTS")
def correctReadCounts(resultBWHeader, globalVars):
	resultMeta = fitToCorrectionModel(globalVars)
	mergeTempFiles(resultBWHeader, resultMeta, globalVars)

def getScatterplotSamples(covariFiles):
	xNumRows = 0
	for covariFile in covariFiles:
		xNumRows += int(covariFile[1])

	if xNumRows <= 50000:
		return np.array(range(xNumRows))
	else:
		return np.random.choice(np.array(range(xNumRows)), 50000, replace=False)


@timer("GENERATING NORMALIZED OBSERVED BIGWIGS")
def normalizeBigWigs(resultBWHeader, globalVars):
	normObFileNames = utils.genNormalizedObBWs(
		globalVars["output_dir"],
		resultBWHeader,
		globalVars["regions"],
		globalVars["ctrlbwNames"],
		globalVars["ctrlScaler"],
		globalVars["expbwNames"],
		globalVars["expScaler"]
	)

	print("Nomralized observed bigwig file names: ")
	print(normObFileNames)


def run(args):
	startTime = time.perf_counter()

	globalVars = init(args)

	trainSet90Percentile, trainSet90To99Percentile = selectTrainingSets(globalVars)

	calculateScalers(trainSet90Percentile, trainSet90To99Percentile, globalVars)

	trainSetResult1 = calculateTrainingCovariates(trainSet90Percentile, globalVars)
	trainSetResult2 = calculateTrainingCovariates(trainSet90To99Percentile, globalVars)
	del trainSet90Percentile, trainSet90To99Percentile
	gc.collect()

	performRegression(trainSetResult1, trainSetResult2, globalVars)
	del trainSetResult1, trainSetResult2

	resultBWHeader = utils.getResultBWHeader(globalVars["regions"], globalVars["ctrlbwNames"][0])

	correctReadCounts(resultBWHeader, globalVars)

	if globalVars["i_generate_norm_bw"]:
		normalizeBigWigs(resultBWHeader, globalVars)


	print(f"-- RUNNING TIME: {((time.perf_counter() - startTime)/3600)} hour(s)")
