import multiprocessing
import os
import os.path
import numpy as np
import pyBigWig
import statsmodels.sandbox.stats.multicomp

from CRADLE.CallPeak import vari
from CRADLE.CallPeak import calculateRC


def setResultValues(mergedResult, pvalues, qvalues, ctrlBW, normCtrlBW, expBW, normExpBW, globalVars):
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

	cohensD = calculateCohenD(ctrlRC, expRC, globalVars["ctrlbwNum"], globalVars["expbwNum"])
	if cohensD == np.nan:
		print(f"""
		Warning: Pooled Std Dev of Cohen's D is 0.
		  This could mean that all your read counts are the same or that you
		  aren't using enough BigWig files (You need at least 3).

		  Location {regionChromo}:{regionStart}-{regionEnd}
		""")

	mergedResult.extend([ctrlRCPosMean, expRCPosMean, cohensD])

	if globalVars["i_log2fc"]:
		normCtrlRC, normExpRC = getRCFromBWs(normCtrlBW, normExpBW, regionChromo, regionStart, regionEnd)
		normCtrlRCPosMean = np.nanmean(normCtrlRC)
		normExpRCPosMean = np.nanmean(normExpRC)

		peusdoLog2FC = calculatePeusdoLog2FC(ctrlRCPosMean, expRCPosMean, normCtrlRCPosMean, normExpRCPosMean)
	else:
		peusdoLog2FC = np.nan

	mergedResult.append(peusdoLog2FC)


def mergePeaks(peakResult, globalVars):
	## open bigwig files to calculate effect size
	ctrlBW = [0] * globalVars["ctrlbwNum"]
	expBW = [0] * globalVars["expbwNum"]

	for i in range(globalVars["ctrlbwNum"]):
		ctrlBW[i] = pyBigWig.open(globalVars["ctrlbwNames"][i])
	for i in range(globalVars["expbwNum"]):
		expBW[i] = pyBigWig.open(globalVars["expbwNames"][i])

	if globalVars["i_log2fc"]:
		normCtrlBW = [pyBigWig.open(bwName) for bwName in globalVars["normCtrlbwNames"]]
		normExpBW = [pyBigWig.open(bwName) for bwName in globalVars["normExpbwNames"]]
	else:
		normCtrlBW = None
		normExpBW = None

	mergedPeak = []

	pastChromo = peakResult[0][0]
	pastEnd = int(peakResult[0][2])
	pastEnrich = int(peakResult[0][3])
	pvalues = [float(peakResult[0][4])]
	qvalues = [float(peakResult[0][5])]

	mergedPeak.append(peakResult[0])
	resultIdx = 0
	if len(peakResult) == 1:
		setResultValues(mergedPeak[resultIdx], pvalues, qvalues, ctrlBW, normCtrlBW, expBW, normExpBW, globalVars)

		for i in range(globalVars["ctrlbwNum"]):
			ctrlBW[i].close()

			if globalVars["i_log2fc"]:
				normCtrlBW[i].close()

		for i in range(globalVars["expbwNum"]):
			expBW[i].close()

			if globalVars["i_log2fc"]:
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

		if (currChromo == pastChromo) and (currEnrich == pastEnrich) and ( (currStart-pastEnd) <= globalVars["distance"]):
			mergedPeak[resultIdx][2] = currEnd
			pvalues.append(currpvalue)
			qvalues.append(currqvalue)
		else:
			setResultValues(mergedPeak[resultIdx], pvalues, qvalues, ctrlBW, normCtrlBW, expBW, normExpBW, globalVars)

			## start a new region
			mergedPeak.append(peakResult[i])
			pvalues = [currpvalue]
			qvalues = [currqvalue]
			resultIdx = resultIdx + 1

		if i == (len(peakResult) - 1):
			setResultValues(mergedPeak[resultIdx], pvalues, qvalues, ctrlBW, normCtrlBW, expBW, normExpBW, globalVars)

		pastChromo = currChromo
		pastEnd = currEnd
		pastEnrich = currEnrich

		i += 1

	for i in range(globalVars["ctrlbwNum"]):
		ctrlBW[i].close()

		if globalVars["i_log2fc"]:
			normCtrlBW[i].close()

	for i in range(globalVars["expbwNum"]):
		expBW[i].close()

		if globalVars["i_log2fc"]:
			normExpBW[i].close()

	return mergedPeak


def takeMinusLog(values):
	minValue = np.min(values)

	return np.nan if minValue == 0 else np.round((-1) * np.log10(minValue), 2)


def getRCFromBWs(ctrlBW, expBW, regionChromo, regionStart, regionEnd):
	ctrlRC = [np.nanmean(np.array(bw.values(regionChromo, regionStart, regionEnd))) for bw in ctrlBW]
	expRC = [np.nanmean(np.array(bw.values(regionChromo, regionStart, regionEnd))) for bw in expBW]

	return ctrlRC, expRC


def calculateCohenD(ctrlRC, expRC, ctrlbwNum, expbwNum):
	dof = ctrlbwNum + expbwNum - 2

	ctrlRCMean = np.mean(ctrlRC)
	expRCMean = np.mean(expRC)

	stdDev = np.sqrt(
			((ctrlbwNum - 1) * np.power(np.std(ctrlRC, ddof=1), 2) +
			    (expbwNum - 1) * np.power(np.std(expRC, ddof=1), 2)) /
			dof
		)

	if stdDev == 0:
		return np.nan

	cohenD = (expRCMean - ctrlRCMean) / stdDev

	return cohenD


def calculatePeusdoLog2FC(ctrlRCPosMean, expRCPosMean, normCtrlRCPosMean, normExpRCPosMean):
	constant = np.max([normExpRCPosMean - expRCPosMean, normCtrlRCPosMean - ctrlRCPosMean])
	foldChange = (expRCPosMean + constant) / (ctrlRCPosMean + constant)
	peusdoLog2FC = np.log2(foldChange)

	return peusdoLog2FC


def filterSmallPeaks(peakResult, globalVars):
	maxNegLogPValue = 1
	maxNegLogQValue = 1

	finalResult = []
	for result in peakResult:
		start = int(result[1])
		end = int(result[2])

		if (end - start) >= globalVars["peakLen"]:
			finalResult.append(result)

			neglogPvalue = float(result[4])
			neglogQvalue = float(result[5])

			if not np.isnan(neglogPvalue):
				if neglogPvalue > maxNegLogPValue:
					maxNegLogPValue = neglogPvalue

			if not np.isnan(neglogQvalue):
				if neglogQvalue > maxNegLogQValue:
					maxNegLogQValue = neglogQvalue

	return finalResult, maxNegLogPValue, maxNegLogQValue


def run(args):
	globalVars = vari.setGlobalVariables(args)
	with multiprocessing.Pool(globalVars["numprocess"]) as pool:
		_run(globalVars, pool)

def _run(globalVars, pool):
	###### INITIALIZE PARAMETERS
	print("======  INITIALIZING PARAMETERS ...\n")

	##### CALCULATE FILTER_CUTOFF
	print("======  CALCULATING OVERALL VARIANCE FILTER CUTOFF ...")
	regionTotal = 0
	taskVari = []
	for region in globalVars["region"]:
		regionSize = region["end"] - region["start"]
		regionTotal += regionSize
		taskVari.append((region, globalVars))

		if regionTotal > 300_000_000:
			break

	variancesAndCutoffs = pool.starmap(calculateRC.getVarianceAndRegionCutoff, taskVari)

	variances = []
	diff = []
	for variance, cutoff in variancesAndCutoffs:
		if variance is not None:
			variances.extend(variance)
		if cutoff is not None:
			diff.extend(cutoff)

	globalVars["filterCutoffs"][0] = -1
	for i in range(1, len(globalVars["filterCutoffsThetas"])):
		globalVars["filterCutoffs"][i] = np.percentile(variances, globalVars["filterCutoffsThetas"][i])
	globalVars["filterCutoffs"] = np.array(globalVars["filterCutoffs"])

	print(f"Variance Cutoff: {np.round(globalVars['filterCutoffs'])}")

	print("======  DEFINING REGIONS ...")

	globalVars["nullStd"] = np.sqrt(np.nanvar(diff))
	print(f"Null_std: {globalVars['nullStd']}")
	globalVars["regionCutoff"] = np.percentile(np.array(diff), 99)
	print(f"Region cutoff: {globalVars['regionCutoff']}")

	print("======  PERFORMING STATSTICAL TESTING FOR EACH REGION ...")
	tasks = [(region, globalVars) for region in globalVars["region"]]

	resultTTestFiles = pool.starmap(calculateRC.statTest, tasks)

	resultTTestFiles = [file for file in resultTTestFiles if file is not None]

	##### CHOOSING THETA
	resultTheta = calculateRC.selectTheta(resultTTestFiles, globalVars)

	theta = resultTheta[0]
	globalVars["adjFDR"] = resultTheta[1]
	selectRegionNum = resultTheta[2]
	totalRegionNum = resultTheta[3]
	thetaPvalues = resultTheta[4]


	##### FDR control
	print("======  CALLING PEAKS ...")
	print(f"Selected Variance Theta: {theta}")
	print(f"Total number of regions: {totalRegionNum}")
	print(f"The number of selected regions: {selectRegionNum}")
	print(f"Newly adjusted cutoff: {globalVars['adjFDR']}")


	##### Applying the selected theta
	pValueSimes = []

	### Apply the selected thata to the data
	for thetaPvalueList in thetaPvalues.values():
		for regionTheta, regionPvalue in thetaPvalueList:
			if regionTheta >= theta:
				pValueSimes.append(regionPvalue)

	pValueGroupBh = statsmodels.sandbox.stats.multicomp.multipletests(pValueSimes, alpha=globalVars["fdr"], method='fdr_bh')[0]


	##### Selecting windows
	taskCallPeak = []

	groupPvalueIdx = 0
	for inputFile in thetaPvalues:
		selectRegionIdx = []
		selectedIdx = 0

		for regionIdx, (regionTheta, regionPvalue) in enumerate(thetaPvalues[inputFile]):
			if regionTheta < theta:
				continue

			if pValueGroupBh[groupPvalueIdx + selectedIdx]:
				selectRegionIdx.append(regionIdx)
			selectedIdx += 1

		groupPvalueIdx += selectedIdx

		if len(selectRegionIdx) != 0:
			taskCallPeak.append((inputFile, selectRegionIdx, globalVars))
		else:
			os.remove(inputFile)

	if len(taskCallPeak) == 0:
		print("======= COMPLETED! ===========")
		print(f"There is no peak detected in {globalVars['outputDir']}.")
		return

	resultCallPeak = pool.starmap(calculateRC.doFDRprocedure, taskCallPeak)

	peakResult = []
	for inputFilename in resultCallPeak:
		with open(inputFilename) as inputStream:
			inputFile = inputStream.readlines()

			for line in inputFile:
				peakResult.append(line.split())

		os.remove(inputFilename)

	if len(peakResult) == 0:
		print("======= COMPLETED! ===========")
		print(f"There is no peak detected in {globalVars['outputDir']}.")
		return


	######## WRITE A RESULT FILE
	colNames = ["chr", "start", "end", "name", "score", "strand", "effectSize", "inputCount", "outputCount", "-log(pvalue)", "-log(qvalue)", "cohen's_d", "peusdoLog2FC"]
	mergedPeaks = mergePeaks(peakResult, globalVars)
	finalResult, maxNegLogPValue, maxNegLogQValue = filterSmallPeaks(mergedPeaks, globalVars)

	numActi = 0
	numRepress = 0

	outputFilename = os.path.join(globalVars["outputDir"], "CRADLE_peaks")
	outputStream = open(outputFilename, "w")
	outputStream.write('\t'.join([str(x) for x in colNames]) + "\n")

	for result in finalResult:
		if int(result[3]) == 1:
			numActi = numActi + 1
		else:
			numRepress = numRepress + 1

		## order in a common file ormat
		chromo = result[0]
		start = result[1]
		end = result[2]
		name = chromo + ":" + str(start) + "-" + str(end)
		score = "."
		strand = "."
		effectSize = result[6]
		inputCount = int(result[7])
		outputCount = int(result[8])
		neglogPvalue = float(result[4])
		cohensD = float(result[9])
		peusdoLog2FC = float(result[10])
		if np.isnan(neglogPvalue):
			if maxNegLogPValue == 1:
				neglogPvalue = "-log(0)"
			else:
				neglogPvalue = maxNegLogPValue
		neglogQvalue = float(result[5])
		if np.isnan(neglogQvalue):
			if maxNegLogQValue == 1:
				neglogQvalue = "-log(0)"
			else:
				neglogQvalue = maxNegLogQValue

		peakToAdd = [chromo, start, end, name, score, strand, effectSize, inputCount, outputCount, neglogPvalue, neglogQvalue, cohensD, peusdoLog2FC]

		outputStream.write('\t'.join([str(x) for x in peakToAdd]) + "\n")
	outputStream.close()

	print("======= COMPLETED! ===========")
	print(f"The peak result was saved in {globalVars['outputDir']}")
	print(f"Total number of peaks: {len(finalResult)}")
	print(f"Activated peaks: {numActi}")
	print(f"Repressed peaks: {numRepress}")
