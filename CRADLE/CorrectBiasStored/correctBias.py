import gc
import multiprocessing
import os
import time

import numpy as np
import py2bit
import pyBigWig
import statsmodels.api as sm

import CRADLE.correctbiasutils as utils

from CRADLE.correctbiasutils.cython import arraySplit
from CRADLE.CorrectBiasStored import vari
from CRADLE.CorrectBiasStored import calculateOneBP


RC_PERCENTILE = [0, 20, 40, 60, 80, 90, 92, 94, 96, 98, 99, 100]

def run(args):
	startTime = time.time()
	###### INITIALIZE PARAMETERS
	print("======  INITIALIZING PARAMETERS .... \n")
	vari.setGlobalVariables(args)
	covariates = vari.getStoredCovariates(args.biasType, args.covariDir)

	###### SELECT TRAIN SETS
	print("======  SELECTING TRAIN SETS .... \n")
	trainingSetMeta, rc90Percentile, rc99Percentile = utils.getCandidateTrainingSet(
		RC_PERCENTILE,
		vari.REGION,
		vari.CTRLBW_NAMES[0],
		vari.OUTPUT_DIR
	)
	vari.HIGHRC = rc90Percentile

	trainingSetMeta = utils.process(min(11, vari.NUMPROCESS), utils.fillTrainingSetMeta, trainingSetMeta)

	trainSet90Percentile, trainSet90To99Percentile = utils.selectTrainingSetFromMeta(trainingSetMeta, rc99Percentile)
	del trainingSetMeta

	print("-- RUNNING TIME of selecting training sets from trainSetMeta : %s hour(s)" % ((time.time() - startTime) / 3600) )


	###### NORMALIZING READ COUNTS
	print("======  NORMALIZING READ COUNTS ....")
	if vari.I_NORM:
		if (len(trainSet90Percentile) == 0) or (len(trainSet90To99Percentile) == 0):
			trainingSets = list(vari.REGION)
		else:
			trainingSets = np.concatenate((trainSet90Percentile, trainSet90To99Percentile), axis=0).tolist()

		scalerTasks = utils.getScalerTasks(trainingSets, vari.CTRLBW_NAMES, vari.EXPBW_NAMES, vari.SAMPLE_NUM)
		scalerResult = utils.process(len(scalerTasks), utils.getScalerForEachSample, scalerTasks)

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

	print("-- RUNNING TIME of calculating scalers : %s hour(s)" % ((time.time() - startTime) / 3600) )

	## PERFORM REGRESSION
	print("======  PERFORMING REGRESSION ....\n")
	pool = multiprocessing.Pool(2)

	if len(trainSet90Percentile) == 0:
		trainSet90Percentile = vari.REGION
	if len(trainSet90To99Percentile) == 0:
		trainSet90To99Percentile = vari.REGION

	with py2bit.open(vari.FA) as faFile:
		trainSet90Percentile = utils.alignCoordinatesToHDF(faFile, trainSet90Percentile, covariates.fragLen)
		trainSet90To99Percentile = utils.alignCoordinatesToHDF(faFile, trainSet90To99Percentile, covariates.fragLen)

	scatterplotSamples90Percentile = utils.getScatterplotSamples(trainSet90Percentile)
	scatterplotSamples90to99Percentile = utils.getScatterplotSamples(trainSet90To99Percentile)

	coefResult = pool.starmap_async(
		calculateOneBP.performRegression,
		[
			[
				trainSet90Percentile, covariates, vari.CTRLBW_NAMES, vari.CTRLSCALER, vari.EXPBW_NAMES, vari.EXPSCALER, scatterplotSamples90Percentile
			],
			[
				trainSet90To99Percentile, covariates, vari.CTRLBW_NAMES, vari.CTRLSCALER, vari.EXPBW_NAMES, vari.EXPSCALER, scatterplotSamples90to99Percentile
			]
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

	del trainSet90Percentile, trainSet90To99Percentile
	gc.collect()

	vari.COEFCTRL = coefResult[0][0]
	vari.COEFEXP = coefResult[0][1]
	vari.COEFCTRL_HIGHRC = coefResult[1][0]
	vari.COEFEXP_HIGHRC = coefResult[1][1]


	print("The order of coefficients:")
	print(covariates.order)

	noNanIdx = [0]
	temp = np.where(np.isnan(covariates.selected) == False)[0] + 1
	temp = temp.tolist()
	noNanIdx.extend(temp)

	print("COEF_CTRL: ")
	print(np.array(vari.COEFCTRL)[:,noNanIdx])
	print("COEF_EXP: ")
	print(np.array(vari.COEFEXP)[:,noNanIdx])
	print("COEF_CTRL_HIGHRC: ")
	print(np.array(vari.COEFCTRL_HIGHRC)[:,noNanIdx])
	print("COEF_EXP_HIGHRC: ")
	print(np.array(vari.COEFEXP_HIGHRC)[:,noNanIdx])

	print("-- RUNNING TIME of performing regression : %s hour(s)" % ((time.time() - startTime) / 3600) )


	###### FITTING THE TEST  SETS TO THE CORRECTION MODEL
	print("======  FITTING ALL THE ANALYSIS REGIONS TO THE CORRECTION MODEL \n")
	tasks = utils.divideGenome(vari.REGION)
	# `vari.NUMPROCESS * len(vari.CTRLBW_NAMES)` seems like a good number of jobs
	#   to split the work into. This keeps each individual job from using too much
	#   memory without creating so many jobs that compiling the BigWig files from
	#   the generated temp files will take a long time.
	jobCount = min(len(tasks), vari.NUMPROCESS * len(vari.CTRLBW_NAMES))
	processCount = min(len(tasks), vari.NUMPROCESS)
	taskGroups = arraySplit(tasks, jobCount, fillValue=None)
	crcArgs = zip(
		taskGroups,
		[covariates] * jobCount,
		[vari.FA] * jobCount,
		[vari.CTRLBW_NAMES] * jobCount,
		[vari.CTRLSCALER] * jobCount,
		[vari.COEFCTRL] * jobCount,
		[vari.COEFCTRL_HIGHRC] * jobCount,
		[vari.EXPBW_NAMES] * jobCount,
		[vari.EXPSCALER] * jobCount,
		[vari.COEFEXP] * jobCount,
		[vari.COEFEXP_HIGHRC] * jobCount,
		[vari.HIGHRC] * jobCount,
		[vari.MIN_FRAG_FILTER_VALUE] * jobCount,
		[vari.BINSIZE] * jobCount,
		[vari.OUTPUT_DIR] * jobCount
	)
	resultMeta = utils.process(processCount, calculateOneBP.correctReadCount, crcArgs)

	gc.collect()

	print("-- RUNNING TIME of calculating Task covariates : %s hour(s)" % ((time.time() - startTime) / 3600) )

	###### MERGING TEMP FILES
	print("======  MERGING TEMP FILES \n")
	resultBWHeader = utils.getResultBWHeader(vari.REGION, vari.CTRLBW_NAMES)
	correctedFileNames = utils.mergeBWFiles(vari.OUTPUT_DIR, resultBWHeader, resultMeta, vari.CTRLBW_NAMES, vari.EXPBW_NAMES)

	print("Output File Names: ")
	print(correctedFileNames)

	print("======  Completed Correcting Read Counts! \n\n")

	if vari.I_GENERATE_NORM_BW:
		print("======  Generating normalized observed bigwigs \n\n")
		normObFileNames = utils.genNormalizedObBWs(
			vari.OUTPUT_DIR,
			resultBWHeader,
			vari.REGION,
			vari.CTRLBW_NAMES,
			vari.CTRLSCALER,
			vari.EXPBW_NAMES,
			vari.EXPSCALER
		)

		print("Nomralized observed bigwig file names: ")
		print(normObFileNames)

	print("-- RUNNING TIME: %s hour(s)" % ((time.time() - startTime) / 3600) )
