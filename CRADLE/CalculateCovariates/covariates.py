import sys
import time

import CRADLE.correctbiasutils as utils

from CRADLE.CalculateCovariates import vari
from CRADLE.CalculateCovariates.taskCovariates import calculateTaskCovariates
from CRADLE.correctbiasutils import vari as commonVari
from CRADLE.logging import timer



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

	numProcess = min(len(tasks), commonVari.NUMPROCESS)

	covariFileNames = utils.process(numProcess, calculateTaskCovariates, tasks, context="fork")

	return covariFileNames


def run(args):
	startTime = time.perf_counter()

	init(args)

	fitToCorrectionModel()

	print(f"-- RUNNING TIME: {((time.perf_counter() - startTime)/3600)} hour(s)")
