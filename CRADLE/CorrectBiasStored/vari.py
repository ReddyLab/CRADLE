import multiprocessing
import os
import sys
import numpy as np
import pyBigWig

from CRADLE.correctbiasutils import ChromoRegionSet

def setGlobalVariables(args):
	global REGIONS

	### input bigwig files
	setInputFiles(args.ctrlbw, args.expbw)
	setOutputDirectory(args.o)
	setCovariDir(args.biasType, args.covariDir, args.genome)
	with pyBigWig.open(CTRLBW_NAMES[0]) as ctrlBW:
		regionSet = ChromoRegionSet.loadBed(args.r)
		blacklistRegionSet = ChromoRegionSet.loadBed(args.bl) if args.bl else None
		REGIONS = setAnlaysisRegion(regionSet, blacklistRegionSet, ctrlBW)
	setFilterCriteria(args.mi)
	setNumProcess(args.p)
	setNormalization(args.norm, args.generateNormBW)
	seed = setRngSeed(args.rngSeed)
	writeRngSeed(seed, args.o)

class StoredCovariates:
	def __init__(self, biasTypes, directory):
		self.directory = directory.rstrip('/')
		self.name = self.directory.split('/')[-1]
		self.fragLen = int(self.name.split('_')[1][7:]) # 7 = len("fragLen")
		self.order = ['Intercept']
		self.selected = np.array([np.nan] * 6)

		biasTypes = {x.lower() for x in biasTypes} # set comprehension
		validBiasTypes = {'shear', 'pcr', 'map', 'gquad'}

		if not biasTypes.issubset(validBiasTypes):
			print("Warning! Invalid values in -biasType. Only 'shear', 'pcr', 'map', 'gquad' are allowed")
			biasTypes = biasTypes.intersection(validBiasTypes)

		if 'shear' in biasTypes:
			self.selected[0] = 1
			self.selected[1] = 1
			self.order.extend(["MGW_shear", "ProT_shear"])
		if 'pcr' in biasTypes:
			self.selected[2] = 1
			self.selected[3] = 1
			self.order.extend(["Anneal_pcr", "Denature_pcr"])
		if 'map' in biasTypes:
			self.selected[4] = 1
			self.order.extend(["Map_map"])
		if 'gquad' in biasTypes:
			self.selected[5] = 1
			self.order.extend(["Gquad_gquad"])
		self.num = len(self.order) - 1

	def covariateFileName(self, chromosome):
		return self.directory + "/" + self.name + "_" + chromosome + ".hdf5"

def setInputFiles(ctrlbwFiles, expbwFiles):
	global CTRLBW_NAMES
	global EXPBW_NAMES

	global CTRLBW_NUM
	global EXPBW_NUM
	global SAMPLE_NUM

	global COEFCTRL
	global COEFEXP
	global COEFCTRL_HIGHRC
	global COEFEXP_HIGHRC

	global HIGHRC


	CTRLBW_NUM = len(ctrlbwFiles)
	EXPBW_NUM = len(expbwFiles)
	SAMPLE_NUM = CTRLBW_NUM + EXPBW_NUM

	CTRLBW_NAMES = [0] * CTRLBW_NUM
	for i in range(CTRLBW_NUM):
		CTRLBW_NAMES[i] = ctrlbwFiles[i]

	EXPBW_NAMES = [0] * EXPBW_NUM
	for i in range(EXPBW_NUM):
		EXPBW_NAMES[i] = expbwFiles[i]

def setOutputDirectory(outputDir):
	global OUTPUT_DIR

	if outputDir is None:
		outputDir = os.getcwd() + "/CRADLE_correctionResult"

	if outputDir[-1] == "/":
		outputDir = outputDir[:-1]

	OUTPUT_DIR = outputDir

	dirExist = os.path.isdir(OUTPUT_DIR)
	if not dirExist:
		os.makedirs(OUTPUT_DIR)

def getStoredCovariates(biasTypes, covariDir):
	return StoredCovariates(biasTypes, covariDir)

def setCovariDir(biasType, covariDir, genome):
	global COVARI_DIR
	global COVARI_NAME
	global SELECT_COVARI
	global FRAGLEN
	global GENOME
	global COVARI_NUM
	global BINSIZE
	global COVARI_ORDER

	COVARI_DIR = covariDir

	if not os.path.isdir(COVARI_DIR):
		print("Error! There is no covariate directory")
		sys.exit()

	if COVARI_DIR[-1] == "/":
		COVARI_DIR = COVARI_DIR[:-1]

	tempStr = COVARI_DIR.split('/')
	COVARI_NAME = tempStr[len(tempStr)-1]
	tempStr = COVARI_NAME.split("_")[1]
	FRAGLEN = int(tempStr.split("fragLen")[1])

	GENOME = genome
	BINSIZE = 1

	SELECT_COVARI = np.array([np.nan] * 6)
	COVARI_NUM = 0

	COVARI_ORDER = ['Intercept']

	biasType = [x.lower() for x in biasType]

	for i in range(len(biasType)):
		if (biasType[i] != 'shear') and (biasType[i] != 'pcr') and (biasType[i] != 'map') and (biasType[i] != 'gquad'):
			print("Error! Wrong value in -biasType. Only 'shear', 'pcr', 'map', 'gquad' are allowed")
			sys.exit()

	if 'shear' in biasType:
		SELECT_COVARI[0] = 1
		SELECT_COVARI[1] = 1
		COVARI_NUM = COVARI_NUM + 2
		COVARI_ORDER.extend(["MGW_shear", "ProT_shear"])
	if 'pcr' in biasType:
		SELECT_COVARI[2] = 1
		SELECT_COVARI[3] = 1
		COVARI_NUM = COVARI_NUM + 2
		COVARI_ORDER.extend(["Anneal_pcr", "Denature_pcr"])
	if 'map' in biasType:
		SELECT_COVARI[4] = 1
		COVARI_NUM = COVARI_NUM + 1
		COVARI_ORDER.extend(["Map_map"])
	if 'gquad' in biasType:
		SELECT_COVARI[5] = 1
		COVARI_NUM = COVARI_NUM + 1
		COVARI_ORDER.extend(["Gquad_gquad"])

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
	global MIN_FRAG_FILTER_VALUE

	if minFrag is None:
		MIN_FRAG_FILTER_VALUE = SAMPLE_NUM
	else:
		MIN_FRAG_FILTER_VALUE = int(minFrag)

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
	global NUMPROCESS

	systemCPUs = int(multiprocessing.cpu_count())

	if numProcess is None:
		NUMPROCESS = int(systemCPUs / 2.0 )
		if NUMPROCESS < 1:
			NUMPROCESS = 1
	else:
		NUMPROCESS = int(numProcess)

	if NUMPROCESS > systemCPUs:
		print("ERROR: You specified too many cpus! (-p). Running with the maximum cpus in the system")
		NUMPROCESS = systemCPUs

def setNormalization(norm, generateNormBW):
	global I_NORM
	global I_GENERATE_NORM_BW

	if norm.lower() == 'false':
		I_NORM = False
	else:
		I_NORM = True

	if generateNormBW.lower() == 'false':
		I_GENERATE_NORM_BW = False
	else:
		I_GENERATE_NORM_BW = True

	if (not I_NORM) and I_GENERATE_NORM_BW:
		print("ERROR: I_NOMR should be 'True' if I_GENERATE_NORM_BW is 'True'")
		sys.exit()

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
