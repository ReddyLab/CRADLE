# cython: language_level=3

import os
import tempfile
import warnings
import h5py
import numpy as np
import pyBigWig

from CRADLE.correctbiasutils.cython import writeBedFile

cpdef correctReadCounts(covariFileName, chromo, analysisStart, analysisEnd, lastBin, nBins, globalVars):
	warnings.filterwarnings('ignore', r'All-NaN slice encountered')
	warnings.filterwarnings('ignore', r'Mean of empty slice')

	regionStart = analysisStart
	if not lastBin:
		regionEnd = analysisStart + globalVars["binSize"] * nBins
		lastBinStart = None
		lastBinEnd = None
	else: # lastBin
		regionEnd = analysisStart + globalVars["binSize"] * (nBins-1)
		lastBinStart = regionEnd
		lastBinEnd = analysisEnd

	###### GET POSITIONS WHERE THE NUMBER OF FRAGMENTS > MIN_FRAG_FILTER_VALUE
	overallIdx, highReadCountIdx = selectIdx(chromo, regionStart, regionEnd, globalVars["ctrlbwNames"], globalVars["expbwNames"], lastBinStart, lastBinEnd, nBins, globalVars["binSize"], globalVars["highrc"], globalVars["min_frag_filter_value"])

	## OUTPUT FILES
	subfinalCtrlNames = [None] * globalVars["ctrlbwNum"]
	subfinalExperiNames = [None] * globalVars["expbwNum"]

	f = h5py.File(covariFileName, "r")

	for rep in range(globalVars["ctrlbwNum"]):
		if len(overallIdx) == 0:
			subfinalCtrlNames[rep] = None
			continue

		## observed read counts
		bw = pyBigWig.open(globalVars["ctrlbwNames"][rep])
		if lastBin:
			rcArr = np.array(bw.stats(chromo, regionStart, regionEnd, type="mean", nBins=(nBins-1)))
			rcArr[np.where(rcArr==None)] = float(0)
			rcArr = rcArr.tolist()

			last_value = bw.stats(chromo, lastBinStart, lastBinEnd, type="mean", nBins=1)[0]
			if last_value is None:
				last_value = float(0)
			rcArr.extend([last_value])
			rcArr = np.array(rcArr)
		else:
			rcArr = np.array(bw.stats(chromo, regionStart, regionEnd, type="mean", nBins=nBins))
			rcArr[np.where(rcArr==None)] = float(0)
		rcArr = rcArr / globalVars["ctrlScaler"][rep]
		bw.close()

		## predicted read counts
		prdvals = np.exp(np.sum(f['covari'][0:]* globalVars["coefctrl"][rep, 1:], axis=1) + globalVars["coefctrl"][rep, 0])
		prdvals[highReadCountIdx] = np.exp(np.sum(f['covari'][0:][highReadCountIdx] * globalVars["coefctrlHighrc"][rep, 1:], axis=1) + globalVars["coefctrlHighrc"][rep, 0])

		rcArr = rcArr - prdvals
		rcArr = rcArr[overallIdx]
		starts = np.array(list(range(analysisStart, analysisEnd)))[overallIdx]

		idx = np.where( (rcArr < np.finfo(np.float32).min) | (rcArr > np.finfo(np.float32).max))
		starts = np.delete(starts, idx)
		rcArr = np.delete(rcArr, idx)
		if len(rcArr) > 0:
			with tempfile.NamedTemporaryFile(mode="w+t", suffix=".bed", dir=globalVars["outputDir"], delete=False) as subfinalCtrlFile:
				subfinalCtrlNames[rep] = subfinalCtrlFile.name
				writeBedFile(subfinalCtrlFile, starts, rcArr, analysisEnd, globalVars["binSize"])
		else:
			subfinalCtrlNames[rep] = None

	for rep in range(globalVars["expbwNum"]):
		if len(overallIdx) == 0:
			subfinalExperiNames[rep] = None
			continue

		## observed read counts
		bw = pyBigWig.open(globalVars["expbwNames"][rep])
		if lastBin:
			rcArr = np.array(bw.stats(chromo, regionStart, regionEnd, type="mean", nBins=(nBins-1)))
			rcArr[np.where(rcArr==None)] = float(0)
			rcArr = rcArr.tolist()

			last_value = bw.stats(chromo, lastBinStart, lastBinEnd, type="mean", nBins=1)[0]
			if last_value is None:
				last_value = float(0)
			rcArr.extend([last_value])
			rcArr = np.array(rcArr)
		else:
			rcArr = np.array(bw.stats(chromo, regionStart, regionEnd, type="mean", nBins=nBins))
			rcArr[np.where(rcArr==None)] = float(0)
		rcArr = rcArr / globalVars["expScaler"][rep]
		bw.close()


		## predicted read counts
		prdvals = np.exp(np.sum(f['covari'][0:]* globalVars["coefexp"][rep, 1:], axis=1) + globalVars["coefexp"][rep, 0])
		prdvals[highReadCountIdx] = np.exp(np.sum(f['covari'][0:][highReadCountIdx] * globalVars["coefexpHighrc"][rep, 1:], axis=1) + globalVars["coefexpHighrc"][rep, 0])

		rcArr = rcArr - prdvals
		rcArr = rcArr[overallIdx]
		starts = np.array(list(range(analysisStart, analysisEnd)))[overallIdx]

		idx = np.where( (rcArr < np.finfo(np.float32).min) | (rcArr > np.finfo(np.float32).max))
		starts = np.delete(starts, idx)
		rcArr = np.delete(rcArr, idx)
		if len(rcArr) > 0:
			with tempfile.NamedTemporaryFile(mode="w+t", suffix=".bed", dir=globalVars["outputDir"], delete=False) as subfinalExperiFile:
				subfinalExperiNames[rep] = subfinalExperiFile.name
				writeBedFile(subfinalExperiFile, starts, rcArr, analysisEnd, globalVars["binSize"])
		else:
			subfinalExperiNames[rep] = None

	f.close()
	os.remove(covariFileName)

	return [subfinalCtrlNames, subfinalExperiNames, chromo]


def selectIdx(chromo, regionStart, regionEnd, ctrlBWNames, experiBWNames, lastBinStart, lastBinEnd, nBins, binSize, highRC, minFragFilterValue):
	meanMinFragFilterValue = int(np.round(minFragFilterValue / (len(ctrlBWNames) + len(experiBWNames))))

	for rep, bwName in enumerate(ctrlBWNames + experiBWNames):
		with pyBigWig.open(bwName) as bw:
			if lastBinStart is not None:
				temp = np.array(bw.stats(chromo, regionStart, regionEnd, type="mean", nBins=(nBins-1)))
				temp[np.where(temp == None)] = 0.0
				temp = temp.tolist()

				last_value = bw.stats(chromo, lastBinStart, lastBinEnd, type="mean", nBins=1)[0]
				if last_value is None:
					last_value = 0.0
				temp.extend([last_value])
				temp = np.array(temp)
			else:
				temp = np.array(bw.stats(chromo, regionStart, regionEnd, type="mean", nBins=nBins))
				temp[np.where(temp == None)] = 0.0
			temp = np.array(temp).astype(float)

			if rep == 0:
				rc_sum = temp
				highReadCountIdx = np.where(temp > highRC)[0]
				replicateIdx = np.where(temp >= meanMinFragFilterValue)[0]
			else:
				rc_sum = np.array(rc_sum)
				rc_sum += temp

				replicateIdx = selectReplicateIdx(temp, replicateIdx, meanMinFragFilterValue)

	idx = np.where(rc_sum > minFragFilterValue)[0].tolist()
	overallIdx = np.intersect1d(idx, replicateIdx)

	if len(overallIdx) == 0:
		highReadCountIdx = []
		return overallIdx, highReadCountIdx

	highReadCountIdx = np.intersect1d(highReadCountIdx, idx)

	return overallIdx, highReadCountIdx

def selectReplicateIdx(readCounts, prevReplicateIdx, meanMinFragFilterValue):
	currReplicateIdx = np.where(readCounts >= meanMinFragFilterValue)[0]
	currReplicateIdx = np.intersect1d(currReplicateIdx, prevReplicateIdx)

	return currReplicateIdx





