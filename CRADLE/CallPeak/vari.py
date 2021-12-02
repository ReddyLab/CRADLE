import multiprocessing
import os
import sys
import numpy as np
import pyBigWig

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

	if normCtrlbw is None or normExpbw is None:
		I_LOG2FC = False
		return
	
	I_LOG2FC = True
	global NORM_CTRLBW_NAMES
	global NORM_EXPBW_NAMES
	
	if  len(normCtrlbw) != CTRLBW_NUM or len(normExpbw) != EXPBW_NUM:
		print("Error: The number of normalized observed bigwigs does not match with the number of input bigwigs. The number of bigwigs in -ctrlbw and -expbw should match with the nubmer of biwigs in -normCtrlbw and -normExpbw, respectively.")	
		sys.exit()

	NORM_CTRLBW_NAMES = normCtrlbw
	NORM_EXPBW_NAMES = normExpbw




def setOutputDirectory(outputDir):
	global OUTPUT_DIR

	if outputDir is None:
		outputDir = os.getcwd() + "/" + "CRADLE_peak_result"

	if outputDir[-1] == "/":
		outputDir = outputDir[:-1]

	OUTPUT_DIR = outputDir

	dirExist = os.path.isdir(OUTPUT_DIR)
	if not dirExist:
		os.makedirs(OUTPUT_DIR)


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
		regionMerged.append([ pastChromo, pastStart, pastEnd])
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
			print("Error: wbin cannot be larger than rbin (%sbp)" % BINSIZE1)
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
