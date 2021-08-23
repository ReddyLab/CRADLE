import gc
import multiprocessing
import time

import numpy as np
import py2bit

import CRADLE.correctbiasutils as utils

from CRADLE.correctbiasutils.cython import arraySplit
from CRADLE.correctbiasutils import vari as commonVari
from CRADLE.CorrectBiasStored import vari
from CRADLE.CorrectBiasStored.regression import performRegression
from CRADLE.CorrectBiasStored.readCounts import correctReadCounts


RC_PERCENTILE = [0, 20, 40, 60, 80, 90, 92, 94, 96, 98, 99, 100]

def run(args):
	startTime = time.time()
	runningTime = startTime

	print("")
	print("======  INITIALIZING PARAMETERS ....")
	commonVari.setGlobalVariables(args)
	vari.setGlobalVariables(args)
	covariates = vari.getStoredCovariates(args.biasType, args.covariDir)
	print("======  COMPLETED INITIALIZING PARAMETERS ....\n")

	print("======  SELECTING TRAINING SETS ....")
	selectingTime = time.time()
	print("-- Getting candidate training sets ....")
	runningTime = time.time()
	trainingSetMeta, rc90Percentile, rc99Percentile = utils.getCandidateTrainingSet(
		RC_PERCENTILE,
		commonVari.REGIONS,
		commonVari.CTRLBW_NAMES[0],
		commonVari.OUTPUT_DIR
	)
	highRC = rc90Percentile

	print(f"-- Completed getting candidate training sets .... : {(time.time() - runningTime) / 60} min")
	print("-- Filling Training Sets ....")
	runningTime = time.time()
	trainingSetMeta = utils.process(min(11, commonVari.NUMPROCESS), utils.fillTrainingSetMeta, trainingSetMeta)

	print(f"-- Completed filling training sets .... : {(time.time() - runningTime) / 60} min")
	print("-- Selecting training sets from trainSetMeta ....")
	runningTime = time.time()
	trainSet90Percentile, trainSet90To99Percentile = utils.selectTrainingSetFromMeta(trainingSetMeta, rc99Percentile)
	del trainingSetMeta

	print(f"-- Completed selecting training sets from trainSetMeta : {((time.time() - runningTime) / 60)} min(s)")
	print(f"======  COMPLETED SELECTING TRAINING SETS .... : {(time.time() - selectingTime) / 60} min\n")

	print("======  NORMALIZING READ COUNTS ....")
	normalizingTime = time.time()
	print("-- Calculating scalers ....")
	runningTime = time.time()
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
		print("NORMALIZING CONSTANTS: ")
		print(f"* CTRLBW: {commonVari.CTRLSCALER}")
		print(f"* EXPBW: {commonVari.EXPSCALER}")
		print("")

	print(f"-- Completed calculating scalers: {((time.time() - runningTime) / 60)} min(s)")

	print("-- Performing regression ....")
	runningTime = time.time()
	pool = multiprocessing.Pool(2)

	if len(trainSet90Percentile) == 0:
		trainSet90Percentile = commonVari.REGIONS
	if len(trainSet90To99Percentile) == 0:
		trainSet90To99Percentile = commonVari.REGIONS

	with py2bit.open(vari.GENOME) as genome:
		chromoEnds = {chromo: int(genome.chroms(chromo)) for chromo in commonVari.REGIONS.chromos}
		trainSet90Percentile = utils.alignCoordinatesToCovariateFileBoundaries(chromoEnds, trainSet90Percentile, covariates.fragLen)
		trainSet90To99Percentile = utils.alignCoordinatesToCovariateFileBoundaries(chromoEnds, trainSet90To99Percentile, covariates.fragLen)

	scatterplotSamples90Percentile = utils.getScatterplotSampleIndices(trainSet90Percentile.cumulativeRegionSize)
	scatterplotSamples90to99Percentile = utils.getScatterplotSampleIndices(trainSet90To99Percentile.cumulativeRegionSize)

	coefResult = pool.starmap_async(
		performRegression,
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


	print(f"The order of coefficients: {covariates.order}")

	noNanIdx = [0]
	temp = np.where(np.isnan(covariates.selected) == False)[0] + 1
	temp = temp.tolist()
	noNanIdx.extend(temp)

	print("* COEF_CTRL: ")
	print(np.array(coefCtrl)[:,noNanIdx])
	print("* COEF_EXP: ")
	print(np.array(coefExp)[:,noNanIdx])
	print("* COEF_CTRL_HIGHRC: ")
	print(np.array(coefCtrlHighrc)[:,noNanIdx])
	print("* COEF_EXP_HIGHRC: ")
	print(np.array(coefExpHighrc)[:,noNanIdx])
	print("")

	print(f"-- Completed performing regression: {(time.time() - runningTime) / 60} min(s)")
	print(f"======  COMPLETED NORMALIZING READ COUNTS: {(time.time() - normalizingTime) / 60}\n")

	print("======  FITTING ALL THE ANALYSIS REGIONS TO THE CORRECTION MODEL")
	fittingTime = time.time()
	tasks = utils.divideGenome(commonVari.REGIONS)
	# `vari.NUMPROCESS * len(vari.CTRLBW_NAMES)` seems like a good number of jobs
	#   to split the work into. This keeps each individual job from using too much
	#   memory without creating so many jobs that compiling the BigWig files from
	#   the generated temp files will take a long time.
	jobCount = min(len(tasks), commonVari.NUMPROCESS * len(commonVari.CTRLBW_NAMES))
	processCount = min(len(tasks), commonVari.NUMPROCESS)
	taskGroups = arraySplit(tasks, jobCount, fillValue=None)

	print("-- Correcting read counts ....")
	runningTime = time.time()
	crcArgs = zip(
		taskGroups,
		[covariates] * jobCount,
		[chromoEnds] * jobCount,
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
	resultMeta = utils.process(processCount, correctReadCounts, crcArgs)

	print(f"-- Completed correcting read counts: {((time.time() - runningTime) / 60)} min(s)")

	gc.collect()

	print("-- Merging temp files ....")
	runningTime = time.time()
	resultBWHeader = utils.getResultBWHeader(commonVari.REGIONS, commonVari.CTRLBW_NAMES[0])
	correctedFileNames = utils.mergeBWFiles(commonVari.OUTPUT_DIR, resultBWHeader, resultMeta, commonVari.CTRLBW_NAMES, commonVari.EXPBW_NAMES)

	print("* Output file names: ")
	print(f"{correctedFileNames}\n")
	print(f"-- Completed merging temp files: {((time.time() - runningTime) / 60)} min(s)")

	print(f"======  COMPLETED FITTING ALL THE ANALYSIS REGIONS TO THE CORRECTION MODEL: {((time.time() - fittingTime) / 60)} min(s)\n")

	if commonVari.I_GENERATE_NORM_BW:
		print("======  GENERATING NORMALIZED OBSERVED BIGWIGS")
		runningTime = time.time()
		normObFileNames = utils.genNormalizedObBWs(
			commonVari.OUTPUT_DIR,
			resultBWHeader,
			commonVari.REGIONS,
			commonVari.CTRLBW_NAMES,
			commonVari.CTRLSCALER,
			commonVari.EXPBW_NAMES,
			commonVari.EXPSCALER
		)

		print("* Nomralized observed bigwig file names: ")
		print(f"{normObFileNames}\n")
		print(f"====== COMPLETED GENERATING NORMALIZED OBSERVED BIGWIGS: {((time.time() - runningTime) / 60)} min(s)\n")

	print(f"-- TOTAL RUNNING TIME: {((time.time() - startTime) / 3600)} hour(s)")
