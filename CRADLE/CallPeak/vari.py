import sys

def setGlobalVariables(args):
	global DISTANCE
	global NULL_STD
	global PEAKLEN

	setFilterCriteria(args.fdr)
	setBinSize(args.rbin, args.wbin)
	PEAKLEN = setPeakLen(args.pl)
	DISTANCE = args.d
	setStatTesting(args.stat)

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


def setPeakLen(peakLen):
	if peakLen is None:
		return BINSIZE2
	else:
		return peakLen

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

