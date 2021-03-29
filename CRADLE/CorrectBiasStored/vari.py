import multiprocessing
import os
import sys
import numpy as np
import pyBigWig

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

def setGlobalVariables(args):
	global BINSIZE
	# global COEFCTRL
	# global COEFCTRL_HIGHRC
	# global COEFEXP
	# global COEFEXP_HIGHRC
	global CTRLBW_NAMES
	global CTRLBW_NUM
	global EXPBW_NAMES
	global EXPBW_NUM
	global FA
	global HIGHRC
	global I_GENERATE_NORM_BW
	global I_NORM
	global MIN_FRAG_FILTER_VALUE
	global NUMPROCESS
	global OUTPUT_DIR
	global REGION
	global SAMPLE_NUM

	BINSIZE = 1
	HIGHRC = None

	### input bigwig files
	CTRLBW_NAMES, CTRLBW_NUM, EXPBW_NAMES, EXPBW_NUM, SAMPLE_NUM = inputFiles(args.ctrlbw, args.expbw)

	FA = args.faFile
	I_NORM, I_GENERATE_NORM_BW = normalization(args.norm, args.generateNormBW)
	MIN_FRAG_FILTER_VALUE = minFragFilterCriteria(args.mi, SAMPLE_NUM)
	NUMPROCESS = subprocessCount(int(args.p))
	OUTPUT_DIR = outputDirectory(args.o)
	REGION = anlaysisRegion(args.r, args.bl, CTRLBW_NAMES[0])


def inputFiles(ctrlBWFiles, experiBWFiles):
	return (ctrlBWFiles, len(ctrlBWFiles), experiBWFiles, len(experiBWFiles), len(ctrlBWFiles) + len(experiBWFiles))


def outputDirectory(outputDir):
	if outputDir is None:
		outputDir = os.getcwd() + "/CRADLE_correctionResult"

	outputDir = outputDir.rstrip('/')

	if not os.path.exists(outputDir):
		os.makedirs(outputDir)
	elif not os.path.isdir(outputDir):
		print("Error! Output directory (-o) exists but is _not_ a directory")
		sys.exit()

	return outputDir


def getStoredCovariates(biasTypes, covariDir):
	return StoredCovariates(biasTypes, covariDir)


def anlaysisRegion(regionFile, bl, ctrlBWFile):
	regions = []

	with open(regionFile) as inputStream:
		inputFileLines = inputStream.readlines()

		for line in inputFileLines:
			temp = line.split()
			temp[1] = int(temp[1])
			temp[2] = int(temp[2])
			regions.append(temp)

	if len(regions) > 1:
		regions = np.array(regions)
		regions = regions[np.lexsort(( regions[:,1].astype(int), regions[:,0])  ) ]
		regions = regions.tolist()

		regionMerged = []

		pastChromo = regions[0][0]
		pastStart = int(regions[0][1])
		pastEnd = int(regions[0][2])
		regionMerged.append([pastChromo, pastStart, pastEnd])
		resultIdx = 0

		pos = 1
		while pos < len(regions):
			currChromo = regions[pos][0]
			currStart = int(regions[pos][1])
			currEnd = int(regions[pos][2])

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

		regions = regionMerged

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
		for region in regions:
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

					regionWoBL.append([regionChromo, currStart, blStart ])
					currStart = blEnd

				if (pos == (len(overlappedBL)-1)) and (blEnd < regionEnd):
					if blEnd == regionEnd:
						break
					regionWoBL.append([regionChromo, blEnd, regionEnd ])

		regions = regionWoBL

	# check if all chromosomes in the REGION in bigwig files
	bwFile = pyBigWig.open(ctrlBWFile)
	regionsFinal = []
	for region in regions:
		chromo = region[0]
		start = int(region[1])
		end = int(region[2])

		chromoLen = bwFile.chroms(chromo)
		if chromoLen is None:
			continue
		if end > chromoLen:
			region[2] = chromoLen
			if chromoLen <= start:
				continue
		regionsFinal.append([chromo, start, end])
	bwFile.close()

	return regionsFinal


def minFragFilterCriteria(minFrag, default):
	if minFrag is None:
		return default

	return int(minFrag)


def subprocessCount(numProcess):
	systemCPUs = int(multiprocessing.cpu_count())

	if numProcess is None:
		processCount = min(1, int(systemCPUs / 2.0 ))
	else:
		processCount = min(numProcess, systemCPUs)

	if numProcess > systemCPUs:
		print("ERROR: You specified too many cpus! (-p). Running with the maximum cpus in the system")

	return processCount


def normalization(norm, generateNormBW):
	norm = not norm.lower() == 'false'
	generateNormBW = not generateNormBW.lower() == 'false'

	if generateNormBW and (not norm):
		print("ERROR: I_NORM should be 'True' if I_GENERATE_NORM_BW is 'True'")
		sys.exit()

	return (norm, generateNormBW)


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
