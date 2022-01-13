import gc
import multiprocessing
import os
import numpy as np
import pyBigWig
import statsmodels.sandbox.stats.multicomp

from CRADLE.CallPeak import vari
from CRADLE.CallPeak import calculateRC


def setResultValues(mergedResult, pvalues, qvalues, ctrlBW, normCtrlBW, expBW, normExpBW):
	regionChromo = mergedResult[0]
	regionStart = int(mergedResult[1])
	regionEnd = int(mergedResult[2])

	mergedResult[4] = takeMinusLog(pvalues)
	mergedResult[5] = takeMinusLog(qvalues)

	ctrlRC, expRC = getRCFromBWs(ctrlBW, expBW, regionChromo, regionStart, regionEnd)

	ctrlRCPosMean = np.nanmean(ctrlRC)
	expRCPosMean = np.nanmean(expRC)
	diffPos = int(expRCPosMean - ctrlRCPosMean)
	mergedResult[6] = diffPos

	cohens_D = calculateCohenD(ctrlRC, expRC)
	if cohens_D == np.nan:
		print(f"""
		Warning: Pooled Std Dev of Cohen's D is 0.
		  This could mean that all your read counts are the same or that you
		  aren't using enough BigWig files (You need at least 3).

		  Location {regionChromo}:{regionStart}-{regionEnd}
		""")

	mergedResult.extend([ctrlRCPosMean, expRCPosMean, cohens_D])

	if vari.I_LOG2FC:
		normCtrlRC, normExpRC = getRCFromBWs(normCtrlBW, normExpBW, regionChromo, regionStart, regionEnd)
		normCtrlRCPosMean = np.nanmean(normCtrlRC)
		normExpRCPosMean = np.nanmean(normExpRC)

		peusdoLog2FC = calculatePeusdoLog2FC(ctrlRCPosMean, expRCPosMean, normCtrlRCPosMean, normExpRCPosMean)
	else:
		peusdoLog2FC = np.nan

	mergedResult.append(peusdoLog2FC)


def mergePeaks(peakResult):
	## open bigwig files to calculate effect size
	ctrlBW = [0] * vari.CTRLBW_NUM
	expBW = [0] * vari.EXPBW_NUM

	for i in range(vari.CTRLBW_NUM):
		ctrlBW[i] = pyBigWig.open(vari.CTRLBW_NAMES[i])
	for i in range(vari.EXPBW_NUM):
		expBW[i] = pyBigWig.open(vari.EXPBW_NAMES[i])

	if vari.I_LOG2FC:
		normCtrlBW = [pyBigWig.open(bwName) for bwName in vari.NORM_CTRLBW_NAMES]
		normExpBW = [pyBigWig.open(bwName) for bwName in vari.NORM_EXPBW_NAMES]

	mergedPeak = []

	pastChromo = peakResult[0][0]
	pastEnd = int(peakResult[0][2])
	pastEnrich = int(peakResult[0][3])
	pvalues = [float(peakResult[0][4])]
	qvalues = [float(peakResult[0][5])]

	mergedPeak.append(peakResult[0])
	resultIdx = 0
	if len(peakResult) == 1:
		setResultValues(mergedPeak[resultIdx], pvalues, qvalues, ctrlBW, normCtrlBW, expBW, normExpBW)

		for i in range(vari.CTRLBW_NUM):
			ctrlBW[i].close()

			if vari.I_LOG2FC:
				normCtrlBW[i].close()

		for i in range(vari.EXPBW_NUM):
			expBW[i].close()

			if vari.I_LOG2FC:
				normExpBW[i].close()

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
			pvalues.append(currpvalue)
			qvalues.append(currqvalue)
		else:
			setResultValues(mergedPeak[resultIdx], pvalues, qvalues, ctrlBW, normCtrlBW, expBW, normExpBW)

			## start a new region
			mergedPeak.append(peakResult[i])
			pvalues = [currpvalue]
			qvalues = [currqvalue]
			resultIdx = resultIdx + 1

		if i == (len(peakResult) - 1):
			setResultValues(mergedPeak[resultIdx], pvalues, qvalues, ctrlBW, normCtrlBW, expBW, normExpBW)

		pastChromo = currChromo
		pastEnd = currEnd
		pastEnrich = currEnrich

		i = i + 1

	for i in range(vari.CTRLBW_NUM):
		ctrlBW[i].close()

		if vari.I_LOG2FC:
			normCtrlBW[i].close()

	for i in range(vari.EXPBW_NUM):
		expBW[i].close()

		if vari.I_LOG2FC:
			normExpBW[i].close()

	return mergedPeak

def takeMinusLog(values):
	minValue = np.min(values)

	return np.nan if minValue == 0 else np.round((-1) * np.log10(minValue), 2)

def getRCFromBWs(ctrlBW, expBW, regionChromo, regionStart, regionEnd):
	ctrlRC = [np.nanmean(np.array(bw.values(regionChromo, regionStart, regionEnd))) for bw in ctrlBW]
	expRC = [np.nanmean(np.array(bw.values(regionChromo, regionStart, regionEnd))) for bw in expBW]

	return ctrlRC, expRC

def calculateCohenD(ctrlRC, expRC):
	dof = vari.CTRLBW_NUM + vari.EXPBW_NUM - 2

	ctrlRC_mean = np.mean(ctrlRC)
	expRC_mean = np.mean(expRC)

	s = np.sqrt( (  (vari.CTRLBW_NUM-1)*np.power(np.std(ctrlRC, ddof=1), 2) + (vari.EXPBW_NUM-1)*np.power(np.std(expRC, ddof=1), 2)  ) / dof )

	if s == 0:
		return np.nan

	cohenD = (expRC_mean - ctrlRC_mean) / s

	return cohenD

def calculatePeusdoLog2FC(ctrlRCPosMean, expRCPosMean, normCtrlRCPosMean, normExpRCPosMean):
	constant = np.max([normExpRCPosMean - expRCPosMean, normCtrlRCPosMean - ctrlRCPosMean])
	fc = (expRCPosMean + constant) / (ctrlRCPosMean + constant)
	peusdoLog2FC = np.log2(fc)

	return peusdoLog2FC

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

			if not np.isnan(neglogPvalue):
				if neglogPvalue > maxNegLogPValue:
					maxNegLogPValue = neglogPvalue

			if not np.isnan(neglogQvalue):
				if neglogQvalue > maxNegLogQValue:
					maxNegLogQValue = neglogQvalue

	return finalResult, maxNegLogPValue, maxNegLogQValue


def run(args):
	###### INITIALIZE PARAMETERS
	print("======  INITIALIZING PARAMETERS ...\n")
	vari.setGlobalVariables(args)

	##### CALCULATE vari.FILTER_CUTOFF
	print("======  CALCULATING OVERALL VARIANCE FILTER CUTOFF ...")
	regionTotal = 0
	taskVari = []
	for region in vari.REGION:
		regionSize = int(region[2]) - int(region[1])
		regionTotal = regionTotal + regionSize
		taskVari.append(region)

		if regionTotal > 3* np.power(10, 8):
			break

	if len(taskVari) < vari.NUMPROCESS:
		pool = multiprocessing.Pool(len(taskVari))
	else:
		pool = multiprocessing.Pool(vari.NUMPROCESS)

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
	for region in vari.REGION:
		regionSize = int(region[2]) - int(region[1])
		regionTotal = regionTotal + regionSize
		taskDiff.append(region)

		if regionTotal > 3* np.power(10, 8):
			break

	if len(taskDiff) < vari.NUMPROCESS:
		pool = multiprocessing.Pool(len(taskDiff))
	else:
		pool = multiprocessing.Pool(vari.NUMPROCESS)
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
	if len(vari.REGION) < vari.NUMPROCESS:
		pool = multiprocessing.Pool(len(vari.REGION))
	else:
		pool = multiprocessing.Pool(vari.NUMPROCESS)
	resultRegion = pool.map_async(calculateRC.defineRegion, vari.REGION).get()
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

	if len(taskWindow) < vari.NUMPROCESS:
		pool = multiprocessing.Pool(len(taskWindow))
	else:
		pool = multiprocessing.Pool(vari.NUMPROCESS)
	resultTTest = pool.map_async(calculateRC.doWindowApproach, taskWindow).get()
	pool.close()
	pool.join()

	metaFilename = vari.OUTPUT_DIR + "/metaData_pvalues"
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
				PValueSimes.append(regionPvalue)

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
				selectRegionIdx.append(regionIdx)
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
		print("There is no peak detected in %s." % vari.OUTPUT_DIR)
		return

	if len(taskCallPeak) < vari.NUMPROCESS:
		pool = multiprocessing.Pool(len(taskCallPeak))
	else:
		pool = multiprocessing.Pool(vari.NUMPROCESS)
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
		print("There is no peak detected in %s." % vari.OUTPUT_DIR)
		return


	######## WRITE A RESULT FILE
	colNames = ["chr", "start", "end", "name", "score", "strand", "effectSize", "inputCount", "outputCount", "-log(pvalue)", "-log(qvalue)", "cohen's_d", "peusdoLog2FC"]
	mergedPeaks = mergePeaks(peakResult)
	finalResult, maxNegLogPValue, maxNegLogQValue = filterSmallPeaks(mergedPeaks)

	numActi = 0
	numRepress = 0

	outputFilename = vari.OUTPUT_DIR + "/CRADLE_peaks"
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
		peusdoLog2FC = float(finalResult[i][10])
		if np.isnan(neglogPvalue):
			if maxNegLogPValue == 1:
				neglogPvalue = "-log(0)"
			else:
				neglogPvalue = maxNegLogPValue
		neglogQvalue = float(finalResult[i][5])
		if np.isnan(neglogQvalue):
			if maxNegLogQValue == 1:
				neglogQvalue = "-log(0)"
			else:
				neglogQvalue = maxNegLogQValue

		peakToAdd = [chromo, start, end, name, score, strand, effectSize, inputCount, outputCount, neglogPvalue, neglogQvalue, cohens_D, peusdoLog2FC]

		outputStream.write('\t'.join([str(x) for x in peakToAdd]) + "\n")
	outputStream.close()

	print("======= COMPLETED! ===========")
	print("The peak result was saved in %s" % vari.OUTPUT_DIR)
	print("Total number of peaks: %s" % len(finalResult))
	print("Activated peaks: %s" % numActi)
	print("Repressed peaks: %s" % numRepress)
