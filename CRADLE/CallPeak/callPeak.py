import gc
import multiprocessing
import os
import numpy as np
import pyBigWig
import statsmodels.sandbox.stats.multicomp

from CRADLE.CallPeak import vari
from CRADLE.CallPeak import calculateRC
from CRADLE.correctbiasutils import vari as commonVari

def mergePeaks(peakResult):
	## open bigwig files to calculate effect size
	ctrlBW = [0] * commonVari.CTRLBW_NUM
	expBW = [0] * commonVari.EXPBW_NUM

	for i in range(commonVari.CTRLBW_NUM):
		ctrlBW[i] = pyBigWig.open(commonVari.CTRLBW_NAMES[i])
	for i in range(commonVari.EXPBW_NUM):
		expBW[i] = pyBigWig.open(commonVari.EXPBW_NAMES[i])

	mergedPeak = []

	pastChromo = peakResult[0][0]
	pastEnd = int(peakResult[0][2])
	pastEnrich = int(peakResult[0][3])
	pvalues = [float(peakResult[0][4])]
	qvalues = [float(peakResult[0][5])]

	mergedPeak.append(peakResult[0])
	resultIdx = 0
	if len(peakResult) == 1:
		minPValue = np.min(pvalues)
		if minPValue == 0:
			mergedPeak[resultIdx][4] = np.nan
		else:
			mergedPeak[resultIdx][4] = np.round((-1) * np.log10(minPValue), 2)

		minQValue = np.min(qvalues)
		if minQValue == 0:
			mergedPeak[resultIdx][5] = np.nan
		else:
			mergedPeak[resultIdx][5] = np.round((-1) * np.log10(minQValue), 2)

		regionChromo = mergedPeak[resultIdx][0]
		regionStart = int(mergedPeak[resultIdx][1])
		regionEnd = int(mergedPeak[resultIdx][2])

		ctrlRC = []
		for rep in range(commonVari.CTRLBW_NUM):
			rc = np.nanmean(np.array(ctrlBW[rep].values(regionChromo, regionStart, regionEnd)))
			ctrlRC.extend([rc])
		ctrlRCPosMean = np.nanmean(ctrlRC)

		expRC = []
		for rep in range(commonVari.EXPBW_NUM):
			rc = np.nanmean(np.array(expBW[rep].values(regionChromo, regionStart, regionEnd)))
			expRC.extend([rc])
		expRCPosMean = np.nanmean(expRC)

		diffPos = int(expRCPosMean - ctrlRCPosMean)
		cohens_D = calculateCohenD(ctrlRC, expRC)
		mergedPeak[resultIdx][6] = diffPos
		mergedPeak[resultIdx].extend([ctrlRCPosMean, expRCPosMean, cohens_D])

		for i in range(commonVari.CTRLBW_NUM):
			ctrlBW[i].close()
		for i in range(commonVari.EXPBW_NUM):
			expBW[i].close()

		return mergedPeak


	i = 1
	while i < len(peakResult):
		currChromo = peakResult[i][0]
		currStart = int(peakResult[i][1])
		currEnd = int(peakResult[i][2])
		currEnrich = int(peakResult[i][3])
		currpvalue = float(peakResult[i][4])
		currqvalue = float(peakResult[i][5])

		if (currChromo == pastChromo) and (currEnrich == pastEnrich) and ( (currStart-pastEnd) <= vari.DISTANCE):
			mergedPeak[resultIdx][2] = currEnd
			pvalues.extend([ currpvalue ])
			qvalues.extend([ currqvalue ])
		else:
			## update the continuous regions
			minPValue = np.min(pvalues)
			if(minPValue == 0):
				mergedPeak[resultIdx][4] = np.nan
			else:
				mergedPeak[resultIdx][4] = np.round((-1) * np.log10(minPValue), 2)

			minQValue = np.min(qvalues)
			if(minQValue == 0):
				mergedPeak[resultIdx][5] = np.nan
			else:
				mergedPeak[resultIdx][5] = np.round((-1) * np.log10(minQValue), 2)

			regionChromo = mergedPeak[resultIdx][0]
			regionStart = int(mergedPeak[resultIdx][1])
			regionEnd = int(mergedPeak[resultIdx][2])

			ctrlRC = []
			for rep in range(commonVari.CTRLBW_NUM):
				rc = np.nanmean(np.array(ctrlBW[rep].values(regionChromo, regionStart, regionEnd)))
				ctrlRC.extend([rc])
			ctrlRCPosMean = np.nanmean(ctrlRC)

			expRC = []
			for rep in range(commonVari.EXPBW_NUM):
				rc = np.nanmean(np.array(expBW[rep].values(regionChromo, regionStart, regionEnd)))
				expRC.extend([rc])
			expRCPosMean = np.nanmean(expRC)

			diffPos = int(expRCPosMean - ctrlRCPosMean)
			cohens_D = calculateCohenD(ctrlRC, expRC)
			mergedPeak[resultIdx][6] = diffPos
			mergedPeak[resultIdx].extend([ctrlRCPosMean, expRCPosMean, cohens_D])


			## start a new region
			mergedPeak.append(peakResult[i])
			pvalues = [currpvalue]
			qvalues = [currqvalue]
			resultIdx = resultIdx + 1

		if i == (len(peakResult) -1):
			minPValue = np.min(pvalues)
			if(minPValue == 0):
				mergedPeak[resultIdx][4] = np.nan
			else:
				mergedPeak[resultIdx][4] = np.round((-1) * np.log10(minPValue), 2)

			minQValue = np.min(qvalues)
			if(minQValue == 0):
				mergedPeak[resultIdx][5] = np.nan
			else:
				mergedPeak[resultIdx][5] = np.round((-1) * np.log10(minQValue), 2)

			regionChromo = mergedPeak[resultIdx][0]
			regionStart = int(mergedPeak[resultIdx][1])
			regionEnd = int(mergedPeak[resultIdx][2])

			ctrlRC = []
			for rep in range(commonVari.CTRLBW_NUM):
				rc = np.nanmean(np.array(ctrlBW[rep].values(regionChromo, regionStart, regionEnd)))
				ctrlRC.extend([rc])
			ctrlRCPosMean = np.nanmean(ctrlRC)

			expRC = []
			for rep in range(commonVari.EXPBW_NUM):
				rc = np.nanmean(np.array(expBW[rep].values(regionChromo, regionStart, regionEnd)))
				expRC.extend([rc])
			expRCPosMean = np.nanmean(expRC)

			diffPos = int(expRCPosMean - ctrlRCPosMean)
			mergedPeak[resultIdx][6] = diffPos
			cohens_D = calculateCohenD(ctrlRC, expRC)
			mergedPeak[resultIdx].extend([ctrlRCPosMean, expRCPosMean, cohens_D])

		pastChromo = currChromo
		pastEnd = currEnd
		pastEnrich = currEnrich

		i = i + 1

	for i in range(commonVari.CTRLBW_NUM):
		ctrlBW[i].close()
	for i in range(commonVari.EXPBW_NUM):
		expBW[i].close()

	return mergedPeak


def calculateCohenD(ctrlRC, expRC):
	dof = commonVari.CTRLBW_NUM + commonVari.EXPBW_NUM - 2

	ctrlRC_mean = np.mean(ctrlRC)
	expRC_mean = np.mean(expRC)

	s = np.sqrt( (  (commonVari.CTRLBW_NUM-1)*np.power(np.std(ctrlRC, ddof=1), 2) + (commonVari.EXPBW_NUM-1)*np.power(np.std(expRC, ddof=1), 2)  ) / dof )

	cohenD = (expRC_mean - ctrlRC_mean) / s

	return cohenD


def filterSmallPeaks(peakResult):


	maxNegLogPValue = 1
	maxNegLogQValue = 1

	finalResult = []
	for i in range(len(peakResult)):
		start = int(peakResult[i][1])
		end = int(peakResult[i][2])

		if (end - start) >= vari.PEAKLEN:
			finalResult.append(peakResult[i])

			neglogPvalue = float(peakResult[i][4])
			neglogQvalue = float(peakResult[i][5])

			if(np.isnan(neglogPvalue) == False):
				if(neglogPvalue > maxNegLogPValue):
					maxNegLogPValue = neglogPvalue

			if(np.isnan(neglogQvalue) == False):
				if(neglogQvalue > maxNegLogQValue):
					maxNegLogQValue = neglogQvalue

	return finalResult, maxNegLogPValue, maxNegLogQValue


def run(args):
	###### INITIALIZE PARAMETERS
	print("======  INITIALIZING PARAMETERS ...\n")
	commonVari.setGlobalVariables(args)
	vari.setGlobalVariables(args)

	##### CALCULATE vari.FILTER_CUTOFF
	print("======  CALCULATING OVERALL VARIANCE FILTER CUTOFF ...")
	regionTotal = 0
	taskVari = []
	for region in commonVari.REGIONS:
		regionSize = int(region[2]) - int(region[1])
		regionTotal = regionTotal + regionSize
		taskVari.append(region)

		if regionTotal > 3* np.power(10, 8):
			break

	if len(taskVari) < commonVari.NUMPROCESS:
		pool = multiprocessing.Pool(len(taskVari))
	else:
		pool = multiprocessing.Pool(commonVari.NUMPROCESS)

	resultFilter = pool.map_async(calculateRC.getVariance, taskVari).get()
	pool.close()
	pool.join()

	var = []
	for i in range(len(resultFilter)):
		if resultFilter[i] is not None:
			var.extend(resultFilter[i])

	vari.FILTER_CUTOFFS[0] = -1
	for i in range(1, len(vari.FILTER_CUTOFFS_THETAS)):
		vari.FILTER_CUTOFFS[i] = np.percentile(var, vari.FILTER_CUTOFFS_THETAS[i])
	vari.FILTER_CUTOFFS = np.array(vari.FILTER_CUTOFFS)

	print("Variance Cutoff: %s" % np.round(vari.FILTER_CUTOFFS))
	del pool, var, resultFilter
	gc.collect()


	##### DEFINING REGIONS
	print("======  DEFINING REGIONS ...")
	# 1)  CALCULATE REGION_CUFOFF
	regionTotal = 0
	taskDiff = []
	for region in commonVari.REGIONS:
		regionSize = int(region[2]) - int(region[1])
		regionTotal = regionTotal + regionSize
		taskDiff.append(region)

		if regionTotal > 3* np.power(10, 8):
			break

	if len(taskDiff) < commonVari.NUMPROCESS:
		pool = multiprocessing.Pool(len(taskDiff))
	else:
		pool = multiprocessing.Pool(commonVari.NUMPROCESS)
	resultDiff = pool.map_async(calculateRC.getRegionCutoff, taskDiff).get()
	pool.close()
	pool.join()

	diff = []
	for i in range(len(resultDiff)):
		if resultDiff[i] is not None:
			diff.extend(resultDiff[i])

	vari.NULL_STD = np.sqrt(np.nanvar(diff))
	print("Null_std: %s" % vari.NULL_STD)
	vari.REGION_CUTOFF = np.percentile(np.array(diff), 99)
	print("Region cutoff: %s " % vari.REGION_CUTOFF)
	del pool, resultDiff, diff, taskDiff
	gc.collect()


	# 2)  DEINING REGIONS WITH 'vari.REGION_CUTOFF'
	if len(commonVari.REGIONS) < commonVari.NUMPROCESS:
		pool = multiprocessing.Pool(len(commonVari.REGIONS))
	else:
		pool = multiprocessing.Pool(commonVari.NUMPROCESS)
	resultRegion = pool.map_async(calculateRC.defineRegion, commonVari.REGIONS).get()
	pool.close()
	pool.join()
	gc.collect()


	##### STATISTICAL TESTING FOR EACH REGION
	print("======  PERFORMING STAITSTICAL TESTING FOR EACH REGION ...")
	taskWindow = []
	for i in range(len(resultRegion)):
		if resultRegion[i] is not None:
			taskWindow.append(resultRegion[i])
	del resultRegion

	if len(taskWindow) < commonVari.NUMPROCESS:
		pool = multiprocessing.Pool(len(taskWindow))
	else:
		pool = multiprocessing.Pool(commonVari.NUMPROCESS)
	resultTTest = pool.map_async(calculateRC.doWindowApproach, taskWindow).get()
	pool.close()
	pool.join()

	metaFilename = os.path.join(commonVari.OUTPUT_DIR, "metaData_pvalues")
	metaStream = open(metaFilename, "w")
	for i in range(len(resultTTest)):
		if resultTTest[i] is not None:
			metaStream.write(resultTTest[i] + "\n")
	metaStream.close()
	del taskWindow, pool, resultTTest

	##### CHOOSING THETA
	taskTheta = [metaFilename]
	pool = multiprocessing.Pool(1)
	resultTheta = pool.map_async(calculateRC.selectTheta, taskTheta).get()
	pool.close()
	pool.join()

	vari.THETA = resultTheta[0][0]
	selectRegionNum = resultTheta[0][1]
	totalRegionNum = resultTheta[0][2]


	##### FDR control
	print("======  CALLING PEAKS ...")
	vari.ADJ_FDR = ( vari.FDR * selectRegionNum ) / float(totalRegionNum)
	print("Selected Variance Theta: %s" % vari.THETA)
	print("Total number of regions: %s" % totalRegionNum)
	print("The number of selected regions: %s" % selectRegionNum)
	print("Newly adjusted cutoff: %s" % vari.ADJ_FDR)


	##### Applying the selected theta
	inputFilename = metaFilename
	inputStream = open(inputFilename)
	inputFile = inputStream.readlines()

	PValueSimes = []

	### Apply the selected thata to the data
	for subFileIdx in range(len(inputFile)):
		subfileName = inputFile[subFileIdx].split()[0]
		subfileStream = open(subfileName)
		subfileFile = subfileStream.readlines()

		for regionIdx in range(len(subfileFile)):
			line = subfileFile[regionIdx].split()
			regionTheta = int(line[3])
			regionPvalue = float(line[4])

			if np.isnan(regionPvalue):
				continue

			if regionTheta >= vari.THETA:
				PValueSimes.extend([ regionPvalue ])

	PValueGroupBh = statsmodels.sandbox.stats.multicomp.multipletests(PValueSimes, alpha=vari.FDR, method='fdr_bh')[0]


	##### Selecting windows
	taskCallPeak = []

	inputFilename = metaFilename
	inputStream = open(inputFilename)
	inputFile = inputStream.readlines()

	groupPvalueIdx = 0
	for subFileIdx in range(len(inputFile)):
		subfileName = inputFile[subFileIdx].split()[0]
		subfileStream = open(subfileName)
		subfileFile = subfileStream.readlines()

		selectRegionIdx = []
		selectedIdx = 0

		for regionIdx in range(len(subfileFile)):
			line = subfileFile[regionIdx].split()
			regionTheta = int(line[3])
			regionPvalue = float(line[4])

			if regionTheta < vari.THETA:
				continue
			if np.isnan(regionPvalue):
				continue
			if PValueGroupBh[groupPvalueIdx + selectedIdx]:
				selectRegionIdx.extend([ regionIdx ])
			selectedIdx = selectedIdx + 1

		groupPvalueIdx = groupPvalueIdx + selectedIdx

		if len(selectRegionIdx) != 0:
			taskCallPeak.append([subfileName, selectRegionIdx])
		else:
			os.remove(subfileName)

	inputStream.close()
	os.remove(metaFilename)

	if len(taskCallPeak) == 0:
		print("======= COMPLETED! ===========")
		print("There is no peak detected in %s." % commonVari.OUTPUT_DIR)
		return

	if len(taskCallPeak) < commonVari.NUMPROCESS:
		pool = multiprocessing.Pool(len(taskCallPeak))
	else:
		pool = multiprocessing.Pool(commonVari.NUMPROCESS)
	resultCallPeak = pool.map_async(calculateRC.doFDRprocedure, taskCallPeak).get()
	pool.close()
	pool.join()

	del pool, taskCallPeak
	gc.collect()

	peakResult = []
	for i in range(len(resultCallPeak)):
		inputFilename = resultCallPeak[i]
		inputStream = open(inputFilename)
		inputFile = inputStream.readlines()

		for j in range(len(inputFile)):
			temp = inputFile[j].split()
			peakResult.append(temp)
		inputStream.close()
		os.remove(inputFilename)

	if len(peakResult) == 0:
		print("======= COMPLETED! ===========")
		print("There is no peak detected in %s." % commonVari.OUTPUT_DIR)
		return


	######## WRITE A RESULT FILE
	colNames = ["chr", "start", "end", "name", "score", "strand", "effectSize", "inputCount", "outputCount", "-log(pvalue)", "-log(qvalue)", "cohen's_d" ]
	mergedPeaks = mergePeaks(peakResult)
	finalResult, maxNegLogPValue, maxNegLogQValue = filterSmallPeaks(mergedPeaks)

	numActi = 0
	numRepress = 0

	outputFilename = commonVari.OUTPUT_DIR + "/CRADLE_peaks"
	outputStream = open(outputFilename, "w")
	outputStream.write('\t'.join([str(x) for x in colNames]) + "\n")

	for i in range(len(finalResult)):
		if int(finalResult[i][3]) == 1:
			numActi = numActi + 1
		else:
			numRepress = numRepress + 1

		## order in a common file ormat
		chromo = finalResult[i][0]
		start = finalResult[i][1]
		end = finalResult[i][2]
		name = chromo + ":" + str(start) + "-" + str(end)
		score = "."
		strand = "."
		effectSize = finalResult[i][6]
		inputCount = int(finalResult[i][7])
		outputCount = int(finalResult[i][8])
		neglogPvalue = float(finalResult[i][4])
		cohens_D = float(finalResult[i][9])
		if(np.isnan(neglogPvalue) == True):
			if(maxNegLogPValue == 1):
				neglogPvalue = "-log(0)"
			else:
				neglogPvalue = maxNegLogPValue
		neglogQvalue = float(finalResult[i][5])
		if(np.isnan(neglogQvalue) == True):
			if(maxNegLogQValue == 1):
				neglogQvalue = "-log(0)"
			else:
				neglogQvalue = maxNegLogQValue

		peakToAdd = [chromo, start, end, name, score, strand, effectSize, inputCount, outputCount, neglogPvalue, neglogQvalue, cohens_D]

		outputStream.write('\t'.join([str(x) for x in peakToAdd]) + "\n")
	outputStream.close()

	print("======= COMPLETED! ===========")
	print("The peak result was saved in %s" % commonVari.OUTPUT_DIR)
	print("Total number of peaks: %s" % len(finalResult))
	print("Activated peaks: %s" % numActi)
	print("Repressed peaks: %s" % numRepress)
