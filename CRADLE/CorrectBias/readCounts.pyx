# cython: language_level=3

import os
import tempfile
import warnings
import h5py
import numpy as np
import pyBigWig

from CRADLE.CorrectBias import vari
from CRADLE.correctbiasutils.cython import writeBedFile
from CRADLE.correctbiasutils import vari as commonVari

cpdef correctReadCounts(covariFileName, chromo, analysisStart, analysisEnd, lastBin, nBins):
	warnings.filterwarnings('ignore', r'All-NaN slice encountered')
	warnings.filterwarnings('ignore', r'Mean of empty slice')

	regionStart = analysisStart
	if not lastBin:
		regionEnd = analysisStart + vari.BINSIZE * nBins
		lastBinStart = None
		lastBinEnd = None
	else: # lastBin
		regionEnd = analysisStart + vari.BINSIZE * (nBins-1)
		lastBinStart = regionEnd
		lastBinEnd = analysisEnd

	###### GET POSITIONS WHERE THE NUMBER OF FRAGMENTS > MIN_FRAG_FILTER_VALUE
	overallIdx, highReadCountIdx = selectIdx(chromo, regionStart, regionEnd, commonVari.CTRLBW_NAMES, commonVari.EXPBW_NAMES, lastBinStart, lastBinEnd, nBins, vari.BINSIZE, vari.HIGHRC, vari.MIN_FRAG_FILTER_VALUE)

	## OUTPUT FILES
	subfinalCtrlNames = [None] * commonVari.CTRLBW_NUM
	subfinalExperiNames = [None] * commonVari.EXPBW_NUM

	f = h5py.File(covariFileName, "r")

	for rep in range(commonVari.CTRLBW_NUM):
		if len(overallIdx) == 0:
			subfinalCtrlNames[rep] = None
			continue

		## observed read counts
		bw = pyBigWig.open(commonVari.CTRLBW_NAMES[rep])
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
		rcArr = rcArr / commonVari.CTRLSCALER[rep]
		bw.close()

		## predicted read counts
		prdvals = np.exp(np.sum(f['covari'][0:]* vari.COEFCTRL[rep, 1:], axis=1) + vari.COEFCTRL[rep, 0])
		prdvals[highReadCountIdx] = np.exp(np.sum(f['covari'][0:][highReadCountIdx] * vari.COEFCTRL_HIGHRC[rep, 1:], axis=1) + vari.COEFCTRL_HIGHRC[rep, 0])

		rcArr = rcArr - prdvals
		rcArr = rcArr[overallIdx]
		starts = np.array(list(range(analysisStart, analysisEnd)))[overallIdx]

		idx = np.where( (rcArr < np.finfo(np.float32).min) | (rcArr > np.finfo(np.float32).max))
		starts = np.delete(starts, idx)
		rcArr = np.delete(rcArr, idx)
		if len(rcArr) > 0:
			with tempfile.NamedTemporaryFile(mode="w+t", suffix=".bed", dir=commonVari.OUTPUT_DIR, delete=False) as subfinalCtrlFile:
				subfinalCtrlNames[rep] = subfinalCtrlFile.name
				writeBedFile(subfinalCtrlFile, starts, rcArr, analysisEnd, vari.BINSIZE)
		else:
			subfinalCtrlNames[rep] = None

	for rep in range(commonVari.EXPBW_NUM):
		if len(overallIdx) == 0:
			subfinalExperiNames[rep] = None
			continue

		## observed read counts
		bw = pyBigWig.open(commonVari.EXPBW_NAMES[rep])
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
		rcArr = rcArr / commonVari.EXPSCALER[rep]
		bw.close()


		## predicted read counts
		prdvals = np.exp(np.sum(f['covari'][0:]* vari.COEFEXP[rep, 1:], axis=1) + vari.COEFEXP[rep, 0])
		prdvals[highReadCountIdx] = np.exp(np.sum(f['covari'][0:][highReadCountIdx] * vari.COEFEXP_HIGHRC[rep, 1:], axis=1) + vari.COEFEXP_HIGHRC[rep, 0])

		rcArr = rcArr - prdvals
		rcArr = rcArr[overallIdx]
		starts = np.array(list(range(analysisStart, analysisEnd)))[overallIdx]

		idx = np.where( (rcArr < np.finfo(np.float32).min) | (rcArr > np.finfo(np.float32).max))
		starts = np.delete(starts, idx)
		rcArr = np.delete(rcArr, idx)
		if len(rcArr) > 0:
			with tempfile.NamedTemporaryFile(mode="w+t", suffix=".bed", dir=commonVari.OUTPUT_DIR, delete=False) as subfinalExperiFile:
				subfinalExperiNames[rep] = subfinalExperiFile.name
				writeBedFile(subfinalExperiFile, starts, rcArr, analysisEnd, vari.BINSIZE)
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





