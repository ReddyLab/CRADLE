import gc
import multiprocessing
import time

import numpy as np
import py2bit

import CRADLE.correctbiasutils as utils

from CRADLE.correctbiasutils.cython import arraySplit
from CRADLE.correctbiasutils import vari as commonVari
from CRADLE.CorrectBiasStored import vari
from CRADLE.CorrectBiasStored import calculateOneBP


RC_PERCENTILE = [0, 20, 40, 60, 80, 90, 92, 94, 96, 98, 99, 100]

def run(args):
	startTime = time.time()
	###### INITIALIZE PARAMETERS
	print("======  INITIALIZING PARAMETERS .... \n")
	commonVari.setGlobalVariables(args)
	vari.setGlobalVariables(args)
	covariates = vari.getStoredCovariates(args.biasType, args.covariDir)

	###### SELECT TRAIN SETS
	print("======  SELECTING TRAIN SETS .... \n")
	trainingSetMeta, rc90Percentile, rc99Percentile = utils.getCandidateTrainingSet(
		RC_PERCENTILE,
		commonVari.REGIONS,
		commonVari.CTRLBW_NAMES[0],
		commonVari.OUTPUT_DIR
	)
	highRC = rc90Percentile

	trainingSetMeta = utils.process(min(11, commonVari.NUMPROCESS), utils.fillTrainingSetMeta, trainingSetMeta)

	trainSet90Percentile, trainSet90To99Percentile = utils.selectTrainingSetFromMeta(trainingSetMeta, rc99Percentile)
	del trainingSetMeta

	print("-- RUNNING TIME of selecting training sets from trainSetMeta : %s hour(s)" % ((time.time() - startTime) / 3600) )


	###### NORMALIZING READ COUNTS
	print("======  NORMALIZING READ COUNTS ....")
	if commonVari.I_NORM:
		if (len(trainSet90Percentile) == 0) or (len(trainSet90To99Percentile) == 0):
			trainingSet = commonVari.REGIONS
		else:
			trainingSet = trainSet90Percentile + trainSet90To99Percentile

		###### OBTAIN READ COUNTS OF THE FIRST REPLICATE OF CTRLBW.
		observedReadCounts1Values = utils.getReadCounts(trainingSet, commonVari.CTRLBW_NAMES[0])

		scalerTasks = utils.getScalerTasks(trainingSet, observedReadCounts1Values, commonVari.CTRLBW_NAMES, commonVari.EXPBW_NAMES)
		scalerResult = utils.process(len(scalerTasks), utils.getScalerForEachSample, scalerTasks)

	else:
		sampleSetCount = len(commonVari.CTRLBW_NAMES) + len(commonVari.EXPBW_NAMES)
		scalerResult = [1] * sampleSetCount

	# Sets vari.CTRLSCALER and vari.EXPSCALER
	commonVari.setScaler(scalerResult)

	if commonVari.I_NORM:
		print("NORMALIZING CONSTANT: ")
		print("CTRLBW: ")
		print(commonVari.CTRLSCALER)
		print("EXPBW: ")
		print(commonVari.EXPSCALER)
		print("\n\n")

	print("-- RUNNING TIME of calculating scalers : %s hour(s)" % ((time.time() - startTime) / 3600) )

	## PERFORM REGRESSION
	print("======  PERFORMING REGRESSION ....\n")
	pool = multiprocessing.Pool(2)

	if len(trainSet90Percentile) == 0:
		trainSet90Percentile = commonVari.REGIONS
	if len(trainSet90To99Percentile) == 0:
		trainSet90To99Percentile = commonVari.REGIONS

	with py2bit.open(vari.GENOME) as genome:
		trainSet90Percentile = utils.alignCoordinatesToCovariateFileBoundaries(genome, trainSet90Percentile, covariates.fragLen)
		trainSet90To99Percentile = utils.alignCoordinatesToCovariateFileBoundaries(genome, trainSet90To99Percentile, covariates.fragLen)

	scatterplotSamples90Percentile = utils.getScatterplotSampleIndices(trainSet90Percentile.cumulativeRegionSize)
	scatterplotSamples90to99Percentile = utils.getScatterplotSampleIndices(trainSet90To99Percentile.cumulativeRegionSize)

	coefResult = pool.starmap_async(
		calculateOneBP.performRegression,
		[
			[
				trainSet90Percentile, covariates, commonVari.CTRLBW_NAMES, commonVari.CTRLSCALER, commonVari.EXPBW_NAMES, commonVari.EXPSCALER, scatterplotSamples90Percentile
			],
			[
				trainSet90To99Percentile, covariates, commonVari.CTRLBW_NAMES, commonVari.CTRLSCALER, commonVari.EXPBW_NAMES, commonVari.EXPSCALER, scatterplotSamples90to99Percentile
			]
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

	del trainSet90Percentile, trainSet90To99Percentile
	gc.collect()

	coefCtrl = coefResult[0][0]
	coefExp = coefResult[0][1]
	coefCtrlHighrc = coefResult[1][0]
	coefExpHighrc = coefResult[1][1]


	print("The order of coefficients:")
	print(covariates.order)

	noNanIdx = [0]
	temp = np.where(np.isnan(covariates.selected) == False)[0] + 1
	temp = temp.tolist()
	noNanIdx.extend(temp)

	print("COEF_CTRL: ")
	print(np.array(coefCtrl)[:,noNanIdx])
	print("COEF_EXP: ")
	print(np.array(coefExp)[:,noNanIdx])
	print("COEF_CTRL_HIGHRC: ")
	print(np.array(coefCtrlHighrc)[:,noNanIdx])
	print("COEF_EXP_HIGHRC: ")
	print(np.array(coefExpHighrc)[:,noNanIdx])

	print("-- RUNNING TIME of performing regression : %s hour(s)" % ((time.time() - startTime) / 3600) )


	###### FITTING THE TEST  SETS TO THE CORRECTION MODEL
	print("======  FITTING ALL THE ANALYSIS REGIONS TO THE CORRECTION MODEL \n")
	tasks = utils.divideGenome(commonVari.REGIONS)
	# `vari.NUMPROCESS * len(vari.CTRLBW_NAMES)` seems like a good number of jobs
	#   to split the work into. This keeps each individual job from using too much
	#   memory without creating so many jobs that compiling the BigWig files from
	#   the generated temp files will take a long time.
	jobCount = min(len(tasks), commonVari.NUMPROCESS * len(commonVari.CTRLBW_NAMES))
	processCount = min(len(tasks), commonVari.NUMPROCESS)
	taskGroups = arraySplit(tasks, jobCount, fillValue=None)
	crcArgs = zip(
		taskGroups,
		[covariates] * jobCount,
		[vari.GENOME] * jobCount,
		[commonVari.CTRLBW_NAMES] * jobCount,
		[commonVari.CTRLSCALER] * jobCount,
		[coefCtrl] * jobCount,
		[coefCtrlHighrc] * jobCount,
		[commonVari.EXPBW_NAMES] * jobCount,
		[commonVari.EXPSCALER] * jobCount,
		[coefExp] * jobCount,
		[coefExpHighrc] * jobCount,
		[highRC] * jobCount,
		[commonVari.MIN_FRAG_FILTER_VALUE] * jobCount,
		[vari.BINSIZE] * jobCount,
		[commonVari.OUTPUT_DIR] * jobCount
	)
	resultMeta = utils.process(processCount, calculateOneBP.correctReadCount, crcArgs)

	gc.collect()

	print("-- RUNNING TIME of calculating Task covariates : %s hour(s)" % ((time.time() - startTime) / 3600) )

	###### MERGING TEMP FILES
	print("======  MERGING TEMP FILES \n")
	resultBWHeader = utils.getResultBWHeader(commonVari.REGIONS, commonVari.CTRLBW_NAMES[0])
	correctedFileNames = utils.mergeBWFiles(commonVari.OUTPUT_DIR, resultBWHeader, resultMeta, commonVari.CTRLBW_NAMES, commonVari.EXPBW_NAMES)

	print("Output File Names: ")
	print(correctedFileNames)

	print("======  Completed Correcting Read Counts! \n\n")

	if commonVari.I_GENERATE_NORM_BW:
		print("======  Generating normalized observed bigwigs \n\n")
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

	print("-- RUNNING TIME: %s hour(s)" % ((time.time() - startTime) / 3600) )
