import multiprocessing
import os
import sys
import numpy as np
import pyBigWig

REGION_DTYPE = np.dtype([("chromo", "U7"), ("start", "i4"), ("end", "i4")])

def setGlobalVariables(args):
	setInputFiles(args.ctrlbw, args.expbw)
	setNormalizedInputFiles(args.normCtrlbw, args.normExpbw)
	setOutputDirectory(args.o)
	setAnlaysisRegion(args.r, args.bl)
	setFilterCriteria(args.fdr)
	setBinSize(args.rbin, args.wbin)
	setPeakLen(args.pl)
	setNumProcess(args.p)
	setDistance(args.d)
	setStatTesting(args.stat)

	return {
		"ctrlbwNames": CTRLBW_NAMES,
		"expbwNames": EXPBW_NAMES,
		"ctrlbwNum": CTRLBW_NUM,
		"expbwNum": EXPBW_NUM,
		"sampleNum": SAMPLE_NUM,
		"allZero": ALL_ZERO,
		"nullStd": None,
		"i_log2fc": I_LOG2FC,
		"normCtrlbwNames": NORM_CTRLBW_NAMES,
		"normExpbwNames": NORM_EXPBW_NAMES,
		"outputDir": OUTPUT_DIR,
		"region": REGION,
		"fdr": FDR,
		"theta": None,
		"filterCutoffs": FILTER_CUTOFFS,
		"filterCutoffsThetas": FILTER_CUTOFFS_THETAS,
		"regionCutoff": None,
		"adjFDR": None,
		"binSize1": BINSIZE1,
		"binSize2": BINSIZE2,
		"shiftSize2": SHIFTSIZE2,
		"numprocess": NUMPROCESS,
		"distance": DISTANCE,
		"peakLen": PEAKLEN,
		"equalVar": EQUALVAR,
	}

def setInputFiles(ctrlbwFiles, expbwFiles):
	global CTRLBW_NAMES
	global EXPBW_NAMES

	global CTRLBW_NUM
	global EXPBW_NUM
	global SAMPLE_NUM

	global ALL_ZERO

	global NULL_STD

	CTRLBW_NUM = len(ctrlbwFiles)
	EXPBW_NUM = len(expbwFiles)
	SAMPLE_NUM = CTRLBW_NUM + EXPBW_NUM

	CTRLBW_NAMES = [0] * CTRLBW_NUM
	for i in range(CTRLBW_NUM):
		CTRLBW_NAMES[i] = ctrlbwFiles[i]

	EXPBW_NAMES = [0] * EXPBW_NUM
	for i in range(EXPBW_NUM):
		EXPBW_NAMES[i] = expbwFiles[i]

	ALL_ZERO = [0] * SAMPLE_NUM


def setNormalizedInputFiles(normCtrlbw, normExpbw):
	global I_LOG2FC
	global NORM_CTRLBW_NAMES
	global NORM_EXPBW_NAMES

	if normCtrlbw is None or normExpbw is None:
		I_LOG2FC = False
		NORM_CTRLBW_NAMES = None
		NORM_EXPBW_NAMES = None
		return

	I_LOG2FC = True

	if  len(normCtrlbw) != CTRLBW_NUM or len(normExpbw) != EXPBW_NUM:
		print("""Error: The number of normalized observed bigwigs does not match with the
	number of input bigwigs. The number of bigwigs in -ctrlbw and -expbw should match
	with the nubmer of biwigs in -normCtrlbw and -normExpbw, respectively.""")
		sys.exit()

	NORM_CTRLBW_NAMES = normCtrlbw
	NORM_EXPBW_NAMES = normExpbw


def setOutputDirectory(outputDir):
	global OUTPUT_DIR

	if outputDir is None:
		outputDir = os.path.join(os.getcwd(), "CRADLE_peak_result")

	if outputDir[-1] == "/":
		outputDir = outputDir[:-1]

	OUTPUT_DIR = outputDir

	dirExist = os.path.isdir(OUTPUT_DIR)
	if not dirExist:
		os.makedirs(OUTPUT_DIR)


def setAnlaysisRegion(regionsFile, blacklistFile):
	global REGION

	REGION = []
	with open(regionsFile) as regions:
		for region in regions:
			temp = region.split()
			REGION.append((temp[0], int(temp[1]), int(temp[2])))

	if len(REGION) > 1:
		REGION = np.array(REGION, dtype=REGION_DTYPE)
		REGION = REGION[np.lexsort((REGION[:]["start"], REGION[:]["chromo"]))]

		regionMerged = []

		pastChromo, pastStart, pastEnd = REGION[0]
		regionMerged.append([ pastChromo, pastStart, pastEnd])
		resultIdx = 0

		for currChromo, currStart, currEnd in REGION[1:]:
			if (currChromo == pastChromo) and (pastStart <= currStart <= pastEnd):
				maxEnd = np.max([currEnd, pastEnd])
				regionMerged[resultIdx][2] = maxEnd
				pastEnd = maxEnd
			else:
				regionMerged.append([currChromo, currStart, currEnd])
				resultIdx += 1
				pastEnd = currEnd

			pastChromo = currChromo
			pastStart = currStart

		REGION = regionMerged

	if blacklistFile is not None:  ### REMOVE BLACKLIST REGIONS FROM 'REGION'
		blRegionsTemp = []
		with open(blacklistFile) as blacklistRegions:
			for blacklistRegion in blacklistRegions:
				temp = blacklistRegion.split()
				blRegionsTemp.append((temp[0], int(temp[1]), int(temp[2])))

		## merge overlapping blacklist regions
		if len(blRegionsTemp) == 1:
			blRegions = blRegionsTemp
		else:
			blRegionsTemp = np.array(blRegionsTemp, dtype=REGION_DTYPE)
			blRegionsTemp = blRegionsTemp[np.lexsort((blRegionsTemp[:]["start"], blRegionsTemp[:]["chromo"]))]

			blRegions = []
			pastChromo, pastStart, pastEnd = blRegionsTemp[0]

			blRegions.append((pastChromo, pastStart, pastEnd))
			resultIdx = 0

			for currChromo, currStart, currEnd in blRegionsTemp[1:]:
				if (currChromo == pastChromo) and (pastStart <= currStart <= pastEnd):
					blRegions[resultIdx] = (blRegions[resultIdx][0], blRegions[resultIdx][1], currEnd)
				else:
					blRegions.append((currChromo, currStart, currEnd))
					resultIdx += 1

				pastChromo = currChromo
				pastStart = currStart
				pastEnd = currEnd
		blRegions = np.array(blRegions, dtype=REGION_DTYPE)

		regionWoBL = []
		for regionChromo, regionStart, regionEnd in REGION:
			overlappedBL = []
			## overlap Case 1 : A blacklist region completely covers the region.
			idx = np.where(
				(blRegions[:]["chromo"] == regionChromo) &
				(blRegions[:]["start"] <= regionStart) &
				(blRegions[:]["end"] >= regionEnd)
				)[0]
			if len(idx) > 0:
				continue

			## overlap Case 2
			idx = np.where(
				(blRegions[:]["chromo"] == regionChromo) &
				(blRegions[:]["end"] > regionStart) &
				(blRegions[:]["end"] <= regionEnd)
				)[0]
			if len(idx) > 0:
				overlappedBL.extend(blRegions[idx].tolist() )

			## overlap Case 3
			idx = np.where(
				(blRegions[:]["chromo"] == regionChromo) &
				(blRegions[:]["start"] >= regionStart) &
				(blRegions[:]["start"] < regionEnd)
				)[0]

			if len(idx) > 0:
				overlappedBL.extend(blRegions[idx].tolist() )

			if len(overlappedBL) == 0:
				regionWoBL.append((regionChromo, regionStart, regionEnd))
				continue

			overlappedBL = np.array(overlappedBL, dtype=REGION_DTYPE)
			overlappedBL = overlappedBL[overlappedBL[:]["start"].argsort()]
			overlappedBL = np.unique(overlappedBL, axis=0)
			overlappedBL = overlappedBL[overlappedBL[:]["start"].argsort()]

			currStart = regionStart
			for pos, blRegion in enumerate(overlappedBL):
				blStart = blRegion["start"]
				blEnd = blRegion["end"]

				if blStart <= regionStart:
					currStart = blEnd
				else:
					if currStart == blStart:
						currStart = blEnd
						continue

					regionWoBL.append((regionChromo, currStart, blStart))
					currStart = blEnd

				if (pos == (len(overlappedBL) - 1)) and (blEnd < regionEnd):
					if blEnd == regionEnd:
						break
					regionWoBL.append((regionChromo, blEnd, regionEnd))

		REGION = regionWoBL

	# check if all chromosomes in the REGION in bigwig files
	bigWig = pyBigWig.open(CTRLBW_NAMES[0])
	regionFinal = []
	for chromo, start, end in REGION:
		chromoLen = bigWig.chroms(chromo)

		if chromoLen is None or chromoLen <= start:
			continue

		regionFinal.append((chromo, start, min(end, chromoLen)))
	bigWig.close()

	REGION = np.array(regionFinal, dtype=REGION_DTYPE)


def setFilterCriteria(fdr):
	global FDR
	global THETA
	global FILTER_CUTOFFS
	global FILTER_CUTOFFS_THETAS
	global REGION_CUTOFF
	global ADJ_FDR

	FDR = float(fdr)
	FILTER_CUTOFFS_THETAS = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
	FILTER_CUTOFFS = [0] * 10 ## theta: 0, 10, 20, 30,...., 90


def setBinSize(binSize1, binSize2):
	global BINSIZE1    ## used to define 'regions'
	global BINSIZE2    ## used to do statistical testing in window-levels
	global SHIFTSIZE2

	if (binSize1 is not None) and (binSize2 is not None):
		if int(binSize1) < int(binSize2):
			print("Error: wbin cannot be larger than rbin")
			sys.exit()
		else:
			BINSIZE1 = int(binSize1)
			BINSIZE2 = int(binSize2)
			SHIFTSIZE2 = BINSIZE2

	elif (binSize1 is None) and (binSize2 is None):
		BINSIZE1 = 300
		BINSIZE2 = int(float(BINSIZE1) / 6)
		SHIFTSIZE2 = BINSIZE2

	elif (binSize1 is None)  and (binSize2 is not None):
		BINSIZE1 = 300
		if int(binSize2) > BINSIZE1:
			print(f"Error: wbin cannot be larger than rbin ({BINSIZE1}bp)")
			sys.exit()
		else:
			BINSIZE2 = int(binSize2)
			SHIFTSIZE2 = BINSIZE2

	else: # binSize1 is not None binSize2 is None
		BINSIZE1 = int(binSize1)
		BINSIZE2 = int(float(BINSIZE1) / 6)
		SHIFTSIZE2 = BINSIZE2


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
		print("ERROR: You specified too many cpus!")
		sys.exit()


def setDistance(distance):
	global DISTANCE

	if distance is None:
		DISTANCE = 10
	else:
		DISTANCE = int(distance)


def setPeakLen(peakLen):
	global PEAKLEN

	if peakLen is None:
		PEAKLEN = int(BINSIZE2)
	else:
		PEAKLEN = int(peakLen)


def setStatTesting(stat):
	global EQUALVAR

	if stat is None:
		EQUALVAR = True
	else:
		stat = stat.lower()
		if stat == 't-test':
			EQUALVAR = True
		elif stat == 'welch':
			EQUALVAR = False
		else:
			print("ERROR: You should use either 't-test' or 'welch' in -stat")
			sys.exit()
