import multiprocessing
import os
import pyBigWig # type: ignore

from CRADLE.correctbiasutils import ChromoRegionSet

def setGlobalVariables(args):
	global CTRLBW_NAMES
	global EXPBW_NAMES

	global CTRLBW_NUM
	global EXPBW_NUM
	global SAMPLE_NUM

	global NUMPROCESS
	global OUTPUT_DIR
	global REGIONS

	### input bigwig files
	CTRLBW_NAMES = args.ctrlbw
	EXPBW_NAMES = args.expbw
	CTRLBW_NUM = len(CTRLBW_NAMES)
	EXPBW_NUM = len(EXPBW_NAMES)
	SAMPLE_NUM = CTRLBW_NUM + EXPBW_NUM
	OUTPUT_DIR = setOutputDirectory(args.o)
	with pyBigWig.open(CTRLBW_NAMES[0]) as ctrlBW:
		regionSet = ChromoRegionSet.loadBed(args.r)
		blacklistRegionSet = ChromoRegionSet.loadBed(args.bl) if "bl" in dir(args) and args.bl else None
		REGIONS = setAnlaysisRegion(regionSet, blacklistRegionSet, ctrlBW)
	NUMPROCESS = setNumProcess(args.p)


def setOutputDirectory(outputDir):
	if not os.path.isdir(outputDir):
		os.makedirs(outputDir)

	return outputDir.rstrip("/")

def setAnlaysisRegion(regionSet, blacklistRegionSet, ctrlBW):
	regionSet.mergeRegions()

	if blacklistRegionSet is not None:
		blacklistRegionSet.mergeRegions()
		regionSetWoBL = regionSet - blacklistRegionSet
	else:
		regionSetWoBL = regionSet

	finalRegionSet = ChromoRegionSet()
	for region in regionSetWoBL:
		chromoLen = ctrlBW.chroms(region.chromo)
		if chromoLen is None or chromoLen <= region.start:
			continue

		if region.end > chromoLen:
			region.end = chromoLen

		finalRegionSet.addRegion(region)

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
