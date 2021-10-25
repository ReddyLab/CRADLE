import gc
import multiprocessing
import sys
import time

import CRADLE.correctbiasutils as utils

from CRADLE.CalculateCovariates import vari
from CRADLE.CalculateCovariates.taskCovariates import calculateTaskCovariates
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


def run(args):
	startTime = time.perf_counter()

	init(args)

	fitToCorrectionModel()

	print(f"-- RUNNING TIME: {((time.perf_counter() - startTime)/3600)} hour(s)")
