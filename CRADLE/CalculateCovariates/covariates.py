import itertools
import math
import os.path
import pickle
import sys
import time

import h5py
import py2bit

import CRADLE.correctbiasutils as utils

from typing import List, Tuple

from CRADLE.CalculateCovariates import vari
from CRADLE.CalculateCovariates.taskCovariates import calculateTaskCovariates
from CRADLE.correctbiasutils import vari as commonVari
from CRADLE.logging import timer



# The covariate values stored in the HDF files start at index 0 (0-index, obviously)
# The lowest start point for an analysis region is 3 (1-indexed), so we need to subtract
# 3 from the analysis start and end points to match them up with correct covariate values
# in the HDF files.
COVARIATE_FILE_INDEX_OFFSET = 3


def outputHDF5File(outputDir, baseName, chromo):
	return os.path.join(outputDir, f"{baseName}_{chromo}.hdf5")


def mergeTempFilesToHDF5(chromo, regionGroup, outputFile):
	with py2bit.open(vari.GENOME) as genome:
		chromoEnd = int(genome.chroms(chromo))

	f = h5py.File(outputFile, "w")
	covariDataSet = f.create_dataset("covari", (chromoEnd, vari.COVARI_NUM), dtype='f', compression="gzip")

	for chromo, analysisStart, analysisEnd in regionGroup:
		print(f"{chromo}:{analysisStart}-{analysisEnd}")
		tempFileName = os.path.join(commonVari.OUTPUT_DIR, f"{chromo}_{analysisStart}_{analysisEnd}.pkl")
		with open(tempFileName, "rb") as file:
			covariates = pickle.load(file)
		start = analysisStart - COVARIATE_FILE_INDEX_OFFSET
		covariDataSet[start:start + covariates.shape[0], :] = covariates

		os.remove(tempFileName)

	f.close()

	return outputFile


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


@timer("Merging Temp Files", 1)
def mergeTempFiles(outputRegions):
	flattenedOuputRegions = []
	for region in outputRegions:
		flattenedOuputRegions.extend(region)

	# sort by chromosome
	flattenedOuputRegions.sort(key=lambda x: x[0])

	# group by chromosome
	mergeGroups = []
	for chromo, regionGroup in itertools.groupby(flattenedOuputRegions, lambda x: x[0]):
		outputFile = outputHDF5File(commonVari.OUTPUT_DIR, os.path.basename(commonVari.OUTPUT_DIR), chromo)
		mergeGroups.append((chromo, list(regionGroup), outputFile))

	correctedFileNames = utils.process(len(mergeGroups), mergeTempFilesToHDF5, mergeGroups, context="fork")

	print("* Output file names: ")
	print(f"{correctedFileNames}\n")


@timer("Calculating Covariates", 1)
def calculateCovariates():
	binnedRegions = utils.divideGenome(commonVari.REGIONS)
	jobGroups = divideWork(binnedRegions, commonVari.REGIONS.cumulativeRegionSize, commonVari.NUMPROCESS)

	coefArgs = [(
		jobGroup,
		commonVari.OUTPUT_DIR,
	) for jobGroup in jobGroups]

	outputRegions = utils.process(min(len(coefArgs), commonVari.NUMPROCESS), calculateTaskCovariates, coefArgs, context="fork")
	return outputRegions


def run(args):
	startTime = time.perf_counter()

	init(args)

	outputRegions = calculateCovariates()

	mergeTempFiles(outputRegions)


	print(f"-- RUNNING TIME: {((time.perf_counter() - startTime)/3600)} hour(s)")
