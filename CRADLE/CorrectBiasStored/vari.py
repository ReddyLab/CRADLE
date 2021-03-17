import multiprocessing
import os
import sys
import numpy as np
import pyBigWig

def setGlobalVariables(args):
	### input bigwig files
	setInputFiles(args.ctrlbw, args.expbw)
	setOutputDirectory(args.o)
	setCovariDir(args.biasType, args.covariDir, args.faFile)
	setAnlaysisRegion(args.r, args.bl)
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

	def hdfFileName(self, chromosome):
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

def setCovariDir(biasType, covariDir, faFile):
	global COVARI_DIR
	global COVARI_NAME
	global SELECT_COVARI
	global FRAGLEN
	global FA
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

	FA = faFile
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


def setAnlaysisRegion(region, bl):
	global REGION

	REGION = []
	inputFilename = region
	inputStream = open(inputFilename)
	inputFile = inputStream.readlines()

	for i in range(len(inputFile)):
		temp = inputFile[i].split()
		temp[1] = int(temp[1])
		temp[2] = int(temp[2])
		REGION.append(temp)
	inputStream.close()

	if len(REGION) > 1:
		REGION = np.array(REGION)
		REGION = REGION[np.lexsort(( REGION[:,1].astype(int), REGION[:,0])  ) ]
		REGION = REGION.tolist()

		regionMerged = []

		pos = 0
		pastChromo = REGION[pos][0]
		pastStart = int(REGION[pos][1])
		pastEnd = int(REGION[pos][2])
		regionMerged.append([pastChromo, pastStart, pastEnd])
		resultIdx = 0

		pos = 1
		while pos < len(REGION):
			currChromo = REGION[pos][0]
			currStart = int(REGION[pos][1])
			currEnd = int(REGION[pos][2])

			if (currChromo == pastChromo) and (currStart >= pastStart) and (currStart <= pastEnd):
				maxEnd = np.max([currEnd, pastEnd])
				regionMerged[resultIdx][2] = maxEnd
				pos = pos + 1
				pastChromo = currChromo
				pastStart = currStart
				pastEnd = maxEnd
			else:
				regionMerged.append([currChromo, currStart, currEnd])
				resultIdx = resultIdx + 1
				pos = pos + 1
				pastChromo = currChromo
				pastStart = currStart
				pastEnd = currEnd

		REGION = regionMerged

	## BL
	if bl is not None:  ### REMOVE BLACKLIST REGIONS FROM 'REGION'
		blRegionTemp = []
		inputStream = open(bl)
		inputFile = inputStream.readlines()
		for i in range(len(inputFile)):
			temp = inputFile[i].split()
			temp[1] = int(temp[1])
			temp[2] = int(temp[2])
			blRegionTemp.append(temp)

		## merge overlapping blacklist regions
		if len(blRegionTemp) == 1:
			blRegion = blRegionTemp
			blRegion = np.array(blRegion)
		else:
			blRegionTemp = np.array(blRegionTemp)
			blRegionTemp = blRegionTemp[np.lexsort( ( blRegionTemp[:,1].astype(int), blRegionTemp[:,0] ) )]
			blRegionTemp = blRegionTemp.tolist()

			blRegion = []
			pos = 0
			pastChromo = blRegionTemp[pos][0]
			pastStart = int(blRegionTemp[pos][1])
			pastEnd = int(blRegionTemp[pos][2])
			blRegion.append([pastChromo, pastStart, pastEnd])
			resultIdx = 0

			pos = 1
			while pos < len(blRegionTemp):
				currChromo = blRegionTemp[pos][0]
				currStart = int(blRegionTemp[pos][1])
				currEnd = int(blRegionTemp[pos][2])

				if (currChromo == pastChromo) and (currStart >= pastStart) and (currStart <= pastEnd):
					blRegion[resultIdx][2] = currEnd
					pos = pos + 1
					pastChromo = currChromo
					pastStart = currStart
					pastEnd = currEnd
				else:
					blRegion.append([currChromo, currStart, currEnd])
					resultIdx = resultIdx + 1
					pos = pos + 1
					pastChromo = currChromo
					pastStart = currStart
					pastEnd = currEnd
			blRegion = np.array(blRegion)

		regionWoBL = []
		for region in REGION:
			regionChromo = region[0]
			regionStart = int(region[1])
			regionEnd = int(region[2])

			overlappedBL = []
			## overlap Case 1 : A blacklist region completely covers the region.
			idx = np.where(
				(blRegion[:,0] == regionChromo) &
				(blRegion[:,1].astype(int) <= regionStart) &
				(blRegion[:,2].astype(int) >= regionEnd)
				)[0]
			if len(idx) > 0:
				continue

			## overlap Case 2
			idx = np.where(
				(blRegion[:,0] == regionChromo) &
				(blRegion[:,2].astype(int) > regionStart) &
				(blRegion[:,2].astype(int) <= regionEnd)
				)[0]
			if len(idx) > 0:
				overlappedBL.extend( blRegion[idx].tolist() )

			## overlap Case 3
			idx = np.where(
				(blRegion[:,0] == regionChromo) &
				(blRegion[:,1].astype(int) >= regionStart) &
				(blRegion[:,1].astype(int) < regionEnd)
				)[0]
			if len(idx) > 0:
				overlappedBL.extend( blRegion[idx].tolist() )

			if len(overlappedBL) == 0:
				regionWoBL.append(region)
				continue

			overlappedBL = np.array(overlappedBL)
			overlappedBL = overlappedBL[overlappedBL[:,1].astype(int).argsort()]
			overlappedBL = np.unique(overlappedBL, axis=0)
			overlappedBL = overlappedBL[overlappedBL[:,1].astype(int).argsort()]

			currStart = regionStart
			for pos in range(len(overlappedBL)):
				blStart = int(overlappedBL[pos][1])
				blEnd = int(overlappedBL[pos][2])

				if blStart <= regionStart:
					currStart = blEnd
				else:
					if currStart == blStart:
						currStart = blEnd
						continue

					regionWoBL.append([ regionChromo, currStart, blStart ])
					currStart = blEnd

				if (pos == (len(overlappedBL)-1)) and (blEnd < regionEnd):
					if blEnd == regionEnd:
						break
					regionWoBL.append([ regionChromo, blEnd, regionEnd ])

		REGION = regionWoBL

	# check if all chromosomes in the REGION in bigwig files
	bw = pyBigWig.open(CTRLBW_NAMES[0])
	regionFinal = []
	for regionIdx in range(len(REGION)):
		chromo = REGION[regionIdx][0]
		start = int(REGION[regionIdx][1])
		end = int(REGION[regionIdx][2])

		chromoLen = bw.chroms(chromo)
		if chromoLen is None:
			continue
		if end > chromoLen:
			REGION[regionIdx][2] = chromoLen
			if chromoLen <= start:
				continue
		regionFinal.append([chromo, start, end])
	bw.close()

	REGION = regionFinal


def setFilterCriteria(minFrag):
	global MIN_FRAGNUM_FILTER_VALUE

	if minFrag is None:
		MIN_FRAGNUM_FILTER_VALUE = SAMPLE_NUM
	else:
		MIN_FRAGNUM_FILTER_VALUE = int(minFrag)


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
	global I_GENERATE_NormBW

	if norm.lower() == 'false':
		I_NORM = False
	else:
		I_NORM = True

	if generateNormBW.lower() == 'false':
		I_GENERATE_NormBW = False
	else:
		I_GENERATE_NormBW = True

	if (not I_NORM) and I_GENERATE_NormBW:
		print("ERROR: I_NOMR should be 'True' if I_GENERATE_NormBW is 'True'")
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
