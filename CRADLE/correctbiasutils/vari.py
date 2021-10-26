import multiprocessing
import os
import pyBigWig # type: ignore

from CRADLE.correctbiasutils import ChromoRegionSet



def setGlobalVariables(args):
	global NUMPROCESS
	global OUTPUT_DIR
	global REGIONS

	### input bigwig files

	regionSet = ChromoRegionSet.loadBed(args.r)
	blacklistRegionSet = ChromoRegionSet.loadBed(args.bl) if "bl" in dir(args) and args.bl else None

	if hasattr(args, "ctrlbw"):
		global CTRLBW_NAMES
		global CTRLBW_NUM

		CTRLBW_NAMES = args.ctrlbw
		CTRLBW_NUM = len(CTRLBW_NAMES)

		with pyBigWig.open(CTRLBW_NAMES[0]) as ctrlBW:
			REGIONS = setAnlaysisRegion(regionSet, blacklistRegionSet, ctrlBW)
	else:
		REGIONS = setAnlaysisRegion(regionSet, blacklistRegionSet)

	if hasattr(args, "expbw"):
		global EXPBW_NAMES
		global EXPBW_NUM

		EXPBW_NAMES = args.expbw
		EXPBW_NUM = len(EXPBW_NAMES)

	if hasattr(args, "ctrlbw") and hasattr(args, "expbw"):
		global SAMPLE_NUM

		SAMPLE_NUM = CTRLBW_NUM + EXPBW_NUM

	NUMPROCESS = setNumProcess(args.p)
	OUTPUT_DIR = setOutputDirectory(args.o)


def setOutputDirectory(outputDir):
	if not os.path.isdir(outputDir):
		os.makedirs(outputDir)

	return outputDir.rstrip("/")


def setAnlaysisRegion(regionSet, blacklistRegionSet, ctrlBW=None):
	regionSet.mergeRegions()

	if blacklistRegionSet is not None:
		blacklistRegionSet.mergeRegions()
		untrimmedRegionSet = regionSet - blacklistRegionSet
	else:
		untrimmedRegionSet = regionSet

	if ctrlBW is not None:
		finalRegionSet = ChromoRegionSet()
		for region in untrimmedRegionSet:
			chromoLen = ctrlBW.chroms(region.chromo)
			if chromoLen is None or chromoLen <= region.start:
				continue

			if region.end > chromoLen:
				region.end = chromoLen

			finalRegionSet.addRegion(region)
	else:
		finalRegionSet = untrimmedRegionSet

	return finalRegionSet

def setScaler(scalerResult):
	global CTRLSCALER
	global EXPSCALER

	CTRLSCALER = [0] * CTRLBW_NUM
	EXPSCALER = [0] * EXPBW_NUM
	CTRLSCALER[0] = 1

	for i in range(1, CTRLBW_NUM):
		CTRLSCALER[i] = scalerResult[i-1]

	for i in range(EXPBW_NUM):
		EXPSCALER[i] = scalerResult[i+CTRLBW_NUM-1]

def setNumProcess(numProcess):
	systemCPUs = multiprocessing.cpu_count()

	if numProcess is None:
		numProcess = max(1, systemCPUs // 2)

	if numProcess > systemCPUs:
		print("ERROR: You specified too many cpus! (-p). Running with the maximum cpus in the system")
		numProcess = systemCPUs

	return numProcess
