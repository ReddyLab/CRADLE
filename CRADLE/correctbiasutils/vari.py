import multiprocessing
import os
import sys
import numpy as np
import pyBigWig

from CRADLE.correctbiasutils import ChromoRegionSet

def setGlobalVariables(args):
	global CTRLBW_NAMES
	global EXPBW_NAMES

	global CTRLBW_NUM
	global EXPBW_NUM
	global SAMPLE_NUM

	global I_GENERATE_NORM_BW
	global I_NORM
	global MIN_FRAG_FILTER_VALUE
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
		blacklistRegionSet = ChromoRegionSet.loadBed(args.bl) if args.bl else None
		REGIONS = setAnlaysisRegion(regionSet, blacklistRegionSet, ctrlBW)
	MIN_FRAG_FILTER_VALUE = setFilterCriteria(args.mi)
	NUMPROCESS = setNumProcess(args.p)
	I_GENERATE_NORM_BW, I_NORM = setNormalization(args.norm, args.generateNormBW)
	seed = setRngSeed(args.rngSeed)
	writeRngSeed(seed, args.o)

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

def setFilterCriteria(minFrag):
	if minFrag is None:
		return SAMPLE_NUM
	else:
		return minFrag

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

def setNormalization(norm, generateNormBW):
	norm = not norm.lower() == 'false'
	generateNormBW = not generateNormBW.lower() == 'false'

	if (not norm) and generateNormBW:
		print("ERROR: I_NOMR should be 'True' if I_GENERATE_NORM_BW is 'True'")
		sys.exit()

	return generateNormBW, norm

def setRngSeed(seed):
	if seed is None:
		seed = np.random.randint(0, 2**32 - 1)

	np.random.seed(seed)
	print(f"RNG Seed: {seed}")
	return seed

def writeRngSeed(seed, outputDir):
	seedFileName = os.path.join(outputDir, "rngseed.txt")
	with open(seedFileName, "w") as seedFile:
		seedFile.write(f"{seed}\n")
