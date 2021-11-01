import gc
import math
import multiprocessing
import time

import numpy as np
import py2bit # type: ignore

import CRADLE.correctbiasutils as utils
import CRADLE.CorrectBiasStored.correctReadCounts as crc
import CRADLE.CorrectBiasStored.regression as reg # type: ignore

from typing import List, Tuple

from CRADLE.correctbiasutils import vari as commonVari
from CRADLE.CorrectBiasStored import vari
from CRADLE.logging import timer


RC_PERCENTILE = [0, 20, 40, 60, 80, 90, 92, 94, 96, 98, 99, 100]


def divideWork(regionSet: List[Tuple[str, int, int]], totalBaseCount: int, numProcesses: int) -> List[List[Tuple[str, int, int]]]:
	""" Break a list of regions into _numProcesses_ separate lists of roughly equal length (as measured by number of base pairs
	covered by the list)
	"""
	idealWorkSize = math.ceil(totalBaseCount / numProcesses)

	# "Priming" currentJobList with the first region helps us avoid some edge cases (e.g., the first region is much bigger than
	# idealWorkSize) where the first list in allJobs would be empty.
	initChromo, initStart, initEnd = regionSet[0]
	currentJobSize = initEnd - initStart
	currentJobList = [(initChromo, initStart, initEnd)]
	allJobs: List[List[Tuple[str, int, int]]] = []
	for chromo, start, end in regionSet[1:]:
		regionLength = end - start
		currentJobSize += regionLength

		# We want len(allJobs) to equal numProcesses so if len(allJobs) == numProcesses - 1
		# then we are currently filling in the very last work set and all the rest of the
		# regions should be added to currentJobList. Once the rest of the regions are added
		# to currentJobList the for loop will end and currentJobList will be appended to allJobs.
		# This will make len(allJobs) == numProcesses, like we want.
		if currentJobSize < idealWorkSize or len(allJobs) == numProcesses - 1:
			currentJobList.append((chromo, start, end))
		elif currentJobSize - idealWorkSize < (regionLength / 5): # It's just a little too big, but that's okay.
			currentJobList.append((chromo, start, end))
			allJobs.append(currentJobList)
			currentJobSize = 0
			currentJobList = []
		else:
			allJobs.append(currentJobList)
			currentJobSize = end - start
			currentJobList = [(chromo, start, end)]

	if len(currentJobList) > 0:
		allJobs.append(currentJobList)

	return allJobs


def divideWorkByChrom(workRegionSets: List[List[Tuple[str, int, int]]]) -> List[List[Tuple[str, int, List[Tuple[str, int, int]]]]]:
	""" Groups and annotates work sets (generated by divideWork) by chromosome.
	Input: [
		[("chr1", 1, 3,), ("chr1", 4, 6)],  # work set 0
		[("chr1", 7, 8), ("chr2", 1, 7)]    # work set 1
	]
	Output: [
		[  # work set 0
			("chr1", 0, [("chr1", 1, 3), ("chr1", 4, 6)])
		],
		[  # work set 1
			("chr1", 1, [("chr1", 7, 8)]),
			("chr2", 0, [("chr2", 1, 7)])
		]
	]
	"""
	currentChrom = ''

	# `currentChromCount` is used to distinguish between the same chromosome in different work sets. I.e.,
	# if part of the chr1 regions are in the first work set and part of the chr1 regions are in the second
	# work set then chr1 will have a `currentChromCount` of 0 for the first work set and a `currentChromCount`
	# of 1 for the second work set. The number is eventually used for naming the temp files so, for instance,
	# two processes working on chr1 regions for "Input1.bw" won't both write to "Input1_chr1.tmp". Instead,
	# the process working on the first part of the chromosome will write to "Input1_chr1_0.tmp" and the
	# process working on the second part of the chromosome will write to "Input1_chr1_1.tmp". The temp files
	# will also be ordered correctly for evntual merging into "Input1_corrected.bw"
	currentChromCount = -1

	allWorkChromoSets = []
	for workRegionSet in workRegionSets:
		newChromo = workRegionSet[0][0]
		if newChromo == currentChrom:
			currentChromCount += 1
		else:
			currentChromCount = 0
			currentChrom = newChromo
		currentRegions = []
		workChromoSet = []
		for region in workRegionSet:
			chromo, _, _ = region
			if chromo == currentChrom:
				currentRegions.append(region)
			else:
				workChromoSet.append((currentChrom, currentChromCount, currentRegions))
				currentChrom = chromo
				currentRegions = [region]
				currentChromCount = 0
		workChromoSet.append((currentChrom, currentChromCount, currentRegions))
		allWorkChromoSets.append(workChromoSet)

	return allWorkChromoSets


@timer("INITIALIZING PARAMETERS")
def init(args):
	commonVari.setGlobalVariables(args)
	vari.setGlobalVariables(args)
	covariates = vari.getStoredCovariates(args.biasType, args.covariDir)

	with py2bit.open(vari.GENOME) as genome:
		chromoEnds = {chromo: int(genome.chroms(chromo)) for chromo in commonVari.REGIONS.chromos}

	resultBWHeader = utils.getResultBWHeader(commonVari.REGIONS, commonVari.CTRLBW_NAMES[0])

	return covariates, chromoEnds, resultBWHeader


@timer("Filling Training Sets", 1)
def fillTrainingSets(trainingSetMeta):
	return utils.process(min(11, commonVari.NUMPROCESS), utils.fillTrainingSetMeta, trainingSetMeta)


@timer("SELECTING TRAINING SETS")
def selectTrainingSets():
	trainingSetMeta, rc90Percentile, rc99Percentile = utils.getCandidateTrainingSet(
		RC_PERCENTILE,
		commonVari.REGIONS,
		commonVari.CTRLBW_NAMES[0],
		commonVari.OUTPUT_DIR
	)
	highRC = rc90Percentile

	trainingSetMeta = fillTrainingSets(trainingSetMeta)

	trainSet90Percentile, trainSet90To99Percentile = utils.selectTrainingSetFromMeta(trainingSetMeta, rc99Percentile)
	del trainingSetMeta

	return trainSet90Percentile, trainSet90To99Percentile, highRC


@timer("Calculating Scalers", 1)
def calculateScalers(trainSet90Percentile, trainSet90To99Percentile):
	if vari.I_NORM:
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

	if vari.I_NORM:
		print("NORMALIZING CONSTANTS: ")
		print(f"* CTRLBW: {commonVari.CTRLSCALER}")
		print(f"* EXPBW: {commonVari.EXPSCALER}")
		print("")


@timer("Performing Regression", 1)
def performRegression(covariates, chromoEnds, trainSet90Percentile, trainSet90To99Percentile ):
	pool = multiprocessing.Pool(2)

	if len(trainSet90Percentile) == 0:
		trainSet90Percentile = commonVari.REGIONS
	if len(trainSet90To99Percentile) == 0:
		trainSet90To99Percentile = commonVari.REGIONS

	trainSet90Percentile = utils.alignCoordinatesToCovariateFileBoundaries(chromoEnds, trainSet90Percentile, covariates.fragLen)
	trainSet90To99Percentile = utils.alignCoordinatesToCovariateFileBoundaries(chromoEnds, trainSet90To99Percentile, covariates.fragLen)

	scatterplotSamples90Percentile = utils.getScatterplotSampleIndices(trainSet90Percentile.cumulativeRegionSize)
	scatterplotSamples90to99Percentile = utils.getScatterplotSampleIndices(trainSet90To99Percentile.cumulativeRegionSize)

	coefResult = pool.starmap_async(
		reg.performRegression,
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

	return coefCtrl, coefExp, coefCtrlHighrc, coefExpHighrc


@timer("NORMALIZING READ COUNTS")
def normalizeReadCounts(covariates, chromoEnds, trainSet90Percentile, trainSet90To99Percentile):
	calculateScalers(trainSet90Percentile, trainSet90To99Percentile)

	return performRegression(covariates, chromoEnds, trainSet90Percentile, trainSet90To99Percentile)


@timer("Correcting Read Counts", 1)
def correctReads(crcArgs):
	utils.process(min(len(crcArgs), commonVari.NUMPROCESS), crc.correctReadCount, crcArgs)


@timer("Merging Temp Files", 1)
def mergeTempFiles(resultBWHeader, jobGroups):
	fileChromoInfo = []
	for jobGroup in jobGroups:
		fileChromoInfo.extend([(chromo, chromoId) for chromo, chromoId, _ in jobGroup])

	correctedFileNames = utils.mergeBWFiles(
		commonVari.OUTPUT_DIR,
		resultBWHeader,
		fileChromoInfo,
		commonVari.CTRLBW_NAMES,
		commonVari.EXPBW_NAMES
	)

	print("* Output file names: ")
	print(f"{correctedFileNames}\n")


@timer("FITTING ALL THE ANALYSIS REGIONS TO THE CORRECTION MODEL")
def correctReadCounts(covariates, chromoEnds, coefCtrl, coefExp, coefCtrlHighrc, coefExpHighrc, highRC):
	binnedRegions = utils.divideGenome(commonVari.REGIONS)

	jobGroups = divideWork(binnedRegions, commonVari.REGIONS.cumulativeRegionSize, commonVari.NUMPROCESS)
	jobGroups = divideWorkByChrom(jobGroups)
	print(f"* {len(binnedRegions)} regions")

	trainingBWName = commonVari.CTRLBW_NAMES[0]
	bwNames = commonVari.CTRLBW_NAMES + commonVari.EXPBW_NAMES
	scalers = commonVari.CTRLSCALER  + commonVari.EXPSCALER
	coefs = np.concatenate((coefCtrl, coefExp), axis=0)
	coefHighrcs =  np.concatenate((coefCtrlHighrc, coefExpHighrc), axis=0)

	crcArgs = [(
		jobGroup,
		chromoEnds,
		covariates,
		trainingBWName,
		bwNames,
		scalers,
		coefs,
		coefHighrcs,
		highRC,
		vari.MIN_FRAG_FILTER_VALUE,
		vari.BINSIZE,
		commonVari.OUTPUT_DIR
	) for jobGroup in jobGroups]

	correctReads(crcArgs)

	return jobGroups


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

	print("* Nomralized observed bigwig file names: ")
	print(f"{normObFileNames}\n")


def run(args):
	startTime = time.perf_counter()

	covariates, chromoEnds, resultBWHeader = init(args)

	trainSet90Percentile, trainSet90To99Percentile, highRC = selectTrainingSets()

	coefCtrl, coefExp, coefCtrlHighrc, coefExpHighrc = normalizeReadCounts(
		covariates,
		chromoEnds,
		trainSet90Percentile,
		trainSet90To99Percentile
	)

	jobGroups = correctReadCounts(covariates, chromoEnds, coefCtrl, coefExp, coefCtrlHighrc, coefExpHighrc, highRC)

	mergeTempFiles(resultBWHeader, jobGroups)

	if vari.I_GENERATE_NORM_BW:
		normalizeBigWigs(resultBWHeader)

	print(f"-- TOTAL RUNNING TIME: {((time.perf_counter() - startTime) / 3600)} hour(s)")
