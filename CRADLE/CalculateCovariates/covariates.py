import itertools
import os.path
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


@timer("Calculating Covariates", 1)
def calculateCovariates():
	tasks = utils.divideGenome(commonVari.REGIONS)
	taskGroups = []

	outputBaseName = os.path.basename(commonVari.OUTPUT_DIR)

	for chromo, tasks in itertools.groupby(tasks, lambda x: x[0]):
		taskGroups.append((chromo, os.path.join(commonVari.OUTPUT_DIR, f"{outputBaseName}_{chromo}.hdf5"), list(tasks)))

	numProcess = min(len(taskGroups), commonVari.NUMPROCESS)

	utils.process(numProcess, calculateTaskCovariates, taskGroups, context="fork")


def run(args):
	startTime = time.perf_counter()

	init(args)

	calculateCovariates()

	print(f"-- RUNNING TIME: {((time.perf_counter() - startTime)/3600)} hour(s)")
