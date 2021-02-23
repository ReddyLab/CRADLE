import multiprocessing
import os
import sys
import numpy as np
import pyBigWig

def setGlobalVariables(args):
	setInputFiles(args.ctrlbw, args.expbw)
	setOutputDirectory(args.o)
	setAnlaysisRegion(args.r, args.bl)
	setFilterCriteria(args.fdr)
	binSize2 = setBinSize(args.l, args.rbin, args.wbin)
	setPeakLen(args.pl, binSize2)
	setNumProcess(args.p)
	setDistance(args.l, args.d)


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
	input_filename = region
	input_stream = open(input_filename)
	input_file = input_stream.readlines()

	for i in range(len(input_file)):
		temp = input_file[i].split()
		temp[1] = int(temp[1])
		temp[2] = int(temp[2])
		REGION.append(temp)
	input_stream.close()

	if len(REGION) > 1:
		REGION = np.array(REGION)
		REGION = REGION[np.lexsort(( REGION[:,1].astype(int), REGION[:,0])  ) ]
		REGION = REGION.tolist()

		region_merged = []

		pos = 0
		pastChromo = REGION[pos][0]
		pastStart = int(REGION[pos][1])
		pastEnd = int(REGION[pos][2])
		region_merged.append([ pastChromo, pastStart, pastEnd])
		resultIdx = 0

		pos = 1
		while pos < len(REGION):
			currChromo = REGION[pos][0]
			currStart = int(REGION[pos][1])
			currEnd = int(REGION[pos][2])

			if (currChromo == pastChromo) and (currStart >= pastStart) and (currStart <= pastEnd):
				maxEnd = np.max([currEnd, pastEnd])
				region_merged[resultIdx][2] = maxEnd
				pos = pos + 1
				pastChromo = currChromo
				pastStart = currStart
				pastEnd = maxEnd
			else:
				region_merged.append([currChromo, currStart, currEnd])
				resultIdx = resultIdx + 1
				pos = pos + 1
				pastChromo = currChromo
				pastStart = currStart
				pastEnd = currEnd

		REGION = region_merged

	if bl is not None:  ### REMOVE BLACKLIST REGIONS FROM 'REGION'
		bl_region_temp = []
		input_stream = open(bl)
		input_file = input_stream.readlines()
		for i in range(len(input_file)):
			temp = input_file[i].split()
			temp[1] = int(temp[1])
			temp[2] = int(temp[2])
			bl_region_temp.append(temp)

		## merge overlapping blacklist regions
		if len(bl_region_temp) == 1:
			bl_region = bl_region_temp
			bl_region = np.array(bl_region)
		else:
			bl_region_temp = np.array(bl_region_temp)
			bl_region_temp = bl_region_temp[np.lexsort( ( bl_region_temp[:,1].astype(int), bl_region_temp[:,0] ) )]
			bl_region_temp = bl_region_temp.tolist()

			bl_region = []
			pos = 0
			pastChromo = bl_region_temp[pos][0]
			pastStart = int(bl_region_temp[pos][1])
			pastEnd = int(bl_region_temp[pos][2])
			bl_region.append([pastChromo, pastStart, pastEnd])
			resultIdx = 0

			pos = 1
			while pos < len(bl_region_temp):
				currChromo = bl_region_temp[pos][0]
				currStart = int(bl_region_temp[pos][1])
				currEnd = int(bl_region_temp[pos][2])

				if (currChromo == pastChromo) and (currStart >= pastStart) and (currStart <= pastEnd):
					bl_region[resultIdx][2] = currEnd
					pos = pos + 1
					pastChromo = currChromo
					pastStart = currStart
					pastEnd = currEnd
				else:
					bl_region.append([currChromo, currStart, currEnd])
					resultIdx = resultIdx + 1
					pos = pos + 1
					pastChromo = currChromo
					pastStart = currStart
					pastEnd = currEnd
			bl_region = np.array(bl_region)

		region_woBL = []
		for region in REGION:
			regionChromo = region[0]
			regionStart = int(region[1])
			regionEnd = int(region[2])

			overlapped_bl = []
			## overlap Case 1 : A blacklist region completely covers the region.
			idx = np.where(
				(bl_region[:,0] == regionChromo) &
				(bl_region[:,1].astype(int) <= regionStart) &
				(bl_region[:,2].astype(int) >= regionEnd)
				)[0]
			if len(idx) > 0:
				continue

			## overlap Case 2
			idx = np.where(
				(bl_region[:,0] == regionChromo) &
				(bl_region[:,2].astype(int) > regionStart) &
				(bl_region[:,2].astype(int) <= regionEnd)
				)[0]
			if len(idx) > 0:
				overlapped_bl.extend( bl_region[idx].tolist() )

			## overlap Case 3
			idx = np.where(
				(bl_region[:,0] == regionChromo) &
				(bl_region[:,1].astype(int) >= regionStart) &
				(bl_region[:,2].astype(int) < regionEnd)
				)[0]

			if len(idx) > 0:
				overlapped_bl.extend( bl_region[idx].tolist() )

			if len(overlapped_bl) == 0:
				region_woBL.append(region)
				continue

			overlapped_bl = np.array(overlapped_bl)
			overlapped_bl = overlapped_bl[overlapped_bl[:,1].astype(int).argsort()]
			overlapped_bl = np.unique(overlapped_bl, axis=0)
			overlapped_bl = overlapped_bl[overlapped_bl[:,1].astype(int).argsort()]

			currStart = regionStart
			for pos in range(len(overlapped_bl)):
				blStart = int(overlapped_bl[pos][1])
				blEnd = int(overlapped_bl[pos][2])

				if blStart <= regionStart:
					currStart = blEnd
				else:
					if currStart == blStart:
						currStart = blEnd
						continue

					region_woBL.append([ regionChromo, currStart, blStart ])
					currStart = blEnd

				if (pos == (len(overlapped_bl)-1)) and (blEnd < regionEnd):
					if blEnd == regionEnd:
						break
					region_woBL.append([ regionChromo, blEnd, regionEnd ])

		REGION = region_woBL

	# check if all chromosomes in the REGION in bigwig files
	bw = pyBigWig.open(CTRLBW_NAMES[0])
	region_final = []
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
		region_final.append([chromo, start, end])
	bw.close()

	REGION = region_final


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


def setBinSize(fragLen, binSize1, binSize2):
	global BINSIZE1    ## used to define 'regions'
	global BINSIZE2    ## used to do statistical testing in window-levels
	global SHIFTSIZE2

	fragLen = float(fragLen)

	if (binSize1 is not None) and (binSize2 is not None):
		if int(binSize1) < int(binSize2):
			print("Error: wbin cannot be larger than rbin")
			sys.exit()
		else:
			BINSIZE1 = int(binSize1)
			BINSIZE2 = int(binSize2)
			SHIFTSIZE2 = BINSIZE2
			return BINSIZE2
	elif (binSize1 is None) and (binSize2 is None):
		BINSIZE1 = int(fragLen * 1.5)
		BINSIZE2 = int(float(BINSIZE1) / 6)
		SHIFTSIZE2 = BINSIZE2
		return BINSIZE2
	elif (binSize1 is None)  and (binSize2 is not None):
		BINSIZE1 = int(fragLen * 1.5)
		if int(binSize2) > BINSIZE1:
			print("Error: wbin cannot be larger than rbin (%sbp)" % BINSIZE1)
			sys.exit()
		else:
			BINSIZE2 = int(binSize2)
			SHIFTSIZE2 = BINSIZE2
			return BINSIZE2
	else: # binSize1 is not None binSize2 is None
		BINSIZE1 = int(binSize1)
		BINSIZE2 = int(float(BINSIZE1) / 6)
		SHIFTSIZE2 = BINSIZE2

		return BINSIZE2

	return BINSIZE2


def setNumProcess(numProcess):
	global NUMPROCESS

	system_cpus = int(multiprocessing.cpu_count())

	if numProcess is None:
		NUMPROCESS = int(system_cpus / 2.0 )
		if NUMPROCESS < 1:
			NUMPROCESS = 1
	else:
		NUMPROCESS = int(numProcess)

	if NUMPROCESS > system_cpus:
		print("ERROR: You specified too many cpus!")
		sys.exit()


def setDistance(fragLen, distance):
	global DISTANCE

	fragLen = float(fragLen)

	if distance is None:
		DISTANCE = int(fragLen / 2.0)
	else:
		DISTANCE = int(distance)


def setPeakLen(peakLen, binSize2):
	global PEAKLEN

	if peakLen is None:
		PEAKLEN = int(binSize2)
	else:
		PEAKLEN = int(peakLen)
