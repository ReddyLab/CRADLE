import gc
import multiprocessing
import time

import numpy as np
import py2bit
import pyBigWig
import statsmodels.api as sm

import CRADLE.correctbiasutils as utils

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
		vari.CTRLBW_NAMES,
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
		[vari.MIN_FRAG_FILTER_VALUE] * taskCount,
		[vari.BINSIZE] * taskCount,
		[vari.OUTPUT_DIR] * taskCount
	)
	resultMeta = utils.process(numProcesses, calculateOneBP.correctReadCount, crcArgs)

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
