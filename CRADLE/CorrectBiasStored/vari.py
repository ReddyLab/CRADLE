import os
import sys
import numpy as np

def setGlobalVariables(args):
	### input bigwig files
	global I_GENERATE_NORM_BW
	global I_NORM
	global MIN_FRAG_FILTER_VALUE

	sampleNum = len(args.ctrlbw) + len(args.expbw)

	I_GENERATE_NORM_BW, I_NORM = setNormalization(args.norm, args.generateNormBW)
	MIN_FRAG_FILTER_VALUE = setFilterCriteria(args.mi, sampleNum)
	setCovariDir(args.biasType, args.covariDir, args.genome)
	seed = setRngSeed(args.rngSeed)
	writeRngSeed(seed, args.o)

class StoredCovariates:
	__slots__ = ["directory", "name", "fragLen", "order", "selected", "num"]

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


def setFilterCriteria(minFrag, sampleNum):
	if minFrag is None:
		return sampleNum
	else:
		return minFrag


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
