import os
import tempfile
import warnings
import h5py
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import statsmodels.api as sm
import py2bit
import pyBigWig

from CRADLE.CorrectBiasStored import vari

matplotlib.use('Agg')
warnings.filterwarnings('ignore', r'All-NaN slice encountered')
warnings.filterwarnings('ignore', r'Mean of empty slice')

cpdef performRegression(trainingSet):
	cdef int analysisStart
	cdef int analysisEnd
	newTrainingSet = []
	xRowCount = 0
	xColumnCount = vari.COVARI_NUM + 1

	with py2bit.open(vari.FA) as faFile:
		for trainIdx in range(len(trainingSet)):
			chromo = trainingSet[trainIdx][0]
			analysisStart = int(trainingSet[trainIdx][1])
			analysisEnd = int(trainingSet[trainIdx][2])
			chromoEnd = int(faFile.chroms(chromo))

			fragStart = analysisStart - vari.FRAGLEN + 1
			fragEnd = analysisEnd + vari.FRAGLEN - 1
			shearStart = fragStart - 2
			shearEnd = fragEnd + 2

			if shearStart < 1:
				shearStart = 1
				fragStart = 3
				analysisStart = max(analysisStart, fragStart)

			if shearEnd > chromoEnd:
				shearEnd = chromoEnd
				fragEnd = shearEnd - 2
				analysisEnd = min(analysisEnd, fragEnd)  # not included

			xRowCount = xRowCount + (analysisEnd - analysisStart)
			newTrainingSet.append([chromo, analysisStart, analysisEnd])

	#### Initialize COEF matrix
	COEFCTRL = np.zeros((vari.CTRLBW_NUM, 7), dtype=np.float64)
	COEFEXP = np.zeros((vari.EXPBW_NUM, 7), dtype=np.float64)

	#### Get X matrix
	cdef double [:,:] xView = np.ones((xRowCount, xColumnCount), dtype=np.float64)

	cdef int currentRow = 0
	cdef int pos
	cdef int j

	for trainIdx in range(len(newTrainingSet)):
		chromo = newTrainingSet[trainIdx][0]
		analysisStart = int(newTrainingSet[trainIdx][1])
		analysisEnd = int(newTrainingSet[trainIdx][2])

		hdfFileName = vari.COVARI_DIR + "/" + vari.COVARI_NAME + "_" + chromo + ".hdf5"
		with h5py.File(hdfFileName, "r") as hdfFile:
			pos = analysisStart
			while pos < analysisEnd:
				temp = hdfFile['covari'][pos - 3] * vari.SELECT_COVARI
				temp = temp[np.isnan(temp) == False]

				j = 0
				while j < vari.COVARI_NUM:
					xView[(currentRow + pos - analysisStart), j + 1] = temp[j]
					j = j + 1
				pos = pos + 1
			currentRow = currentRow + (analysisEnd - analysisStart)

	if xRowCount < 50000:
		idx = np.array(list(range(xRowCount)))
	else:
		idx = np.random.choice(np.array(list(range(xRowCount))), 50000, replace=False)

	#### Get Y matrix
	cdef double [:] yView = np.zeros(xRowCount, dtype=np.float64)
	cdef int ptr
	cdef int posIdx

	for rep in range(vari.CTRLBW_NUM):
		with pyBigWig.open(vari.CTRLBW_NAMES[rep]) as bwFile:
			ptr = 0
			for trainIdx in range(len(newTrainingSet)):
				chromo = newTrainingSet[trainIdx][0]
				analysisStart = int(newTrainingSet[trainIdx][1])
				analysisEnd = int(newTrainingSet[trainIdx][2])

				readCounts = np.array(bwFile.values(chromo, analysisStart, analysisEnd))
				readCounts[np.isnan(readCounts) == True] = float(0)
				readCounts = readCounts / vari.CTRLSCALER[rep]

				numPos = analysisEnd - analysisStart
				posIdx = 0
				while posIdx < numPos:
					yView[ptr + posIdx] = readCounts[posIdx]
					posIdx = posIdx + 1

				ptr = ptr + numPos

		del readCounts

		#### do regression
		model = sm.GLM(np.array(yView).astype(int), np.array(xView), family=sm.families.Poisson(link=sm.genmod.families.links.log)).fit()

		coef = model.params
		COEFCTRL[rep, 0] = coef[0]
		coefIdx = 1
		j = 1
		while j < 7:
			if np.isnan(vari.SELECT_COVARI[j-1]) == True:
				COEFCTRL[rep, j] = np.nan
				j = j + 1
			else:
				COEFCTRL[rep, j] = coef[coefIdx]
				j = j + 1
				coefIdx = coefIdx + 1

		corr = np.corrcoef(model.fittedvalues, np.array(yView))[0, 1]
		corr = np.round(corr, 2)

		## PLOT
		maxi1 = np.nanmax(model.fittedvalues[idx])
		maxi2 = np.nanmax(np.array(yView)[idx])
		maxi = max(maxi1, maxi2)

		bwName = '.'.join( vari.CTRLBW_NAMES[rep].rsplit('/', 1)[-1].split(".")[:-1])
		figName = vari.OUTPUT_DIR + "/fit_" + bwName + ".png"
		plt.plot(np.array(yView)[idx], model.fittedvalues[idx], color='g', marker='s', alpha=0.01)
		plt.text((maxi-25), 10, corr, ha='center', va='center')
		plt.xlabel("observed")
		plt.ylabel("predicted")
		plt.xlim(0, maxi)
		plt.ylim(0, maxi)
		plt.plot([0, maxi], [0, maxi], 'k-', color='r')
		plt.gca().set_aspect('equal', adjustable='box')
		plt.savefig(figName)
		plt.close()
		plt.clf()

	for rep in range(vari.EXPBW_NUM):
		with pyBigWig.open(vari.EXPBW_NAMES[rep]) as bwFile:
			ptr = 0
			for trainIdx in range(len(newTrainingSet)):
				chromo = newTrainingSet[trainIdx][0]
				analysisStart = int(newTrainingSet[trainIdx][1])
				analysisEnd = int(newTrainingSet[trainIdx][2])

				readCounts = np.array(bwFile.values(chromo, analysisStart, analysisEnd))
				readCounts[np.isnan(readCounts) == True] = float(0)
				readCounts = readCounts / vari.EXPSCALER[rep]

				numPos = analysisEnd - analysisStart
				posIdx = 0
				while posIdx < numPos:
					yView[ptr+posIdx] = readCounts[posIdx]
					posIdx = posIdx +1

				ptr = ptr + numPos

		del readCounts

		#### do regression
		model = sm.GLM(np.array(yView).astype(int), np.array(xView), family=sm.families.Poisson(link=sm.genmod.families.links.log)).fit()

		coef = model.params

		COEFEXP[rep, 0] = coef[0]
		coefIdx = 1
		j = 1
		while j < 7:
			if np.isnan(vari.SELECT_COVARI[j-1]) == True:
				COEFEXP[rep, j] = np.nan
				j = j + 1
			else:
				COEFEXP[rep, j] = coef[coefIdx]
				j = j + 1
				coefIdx = coefIdx + 1

		corr = np.corrcoef(model.fittedvalues, np.array(yView))[0, 1]
		corr = np.round(corr, 2)

		## PLOT
		maxi1 = np.nanmax(model.fittedvalues[idx])
		maxi2 = np.nanmax(np.array(yView)[idx])
		maxi = max(maxi1, maxi2)

		bwName = '.'.join( vari.EXPBW_NAMES[rep].rsplit('/', 1)[-1].split(".")[:-1])
		figName = vari.OUTPUT_DIR + "/fit_" + bwName + ".png"
		plt.plot(np.array(yView)[idx], model.fittedvalues[idx], color='g', marker='s', alpha=0.01)
		plt.text((maxi-25), 10, corr, ha='center', va='center')
		plt.xlabel("observed")
		plt.ylabel("predicted")
		plt.xlim(0, maxi)
		plt.ylim(0, maxi)
		plt.plot([0, maxi], [0, maxi], 'k-', color='r')
		plt.gca().set_aspect('equal', adjustable='box')
		plt.savefig(figName)
		plt.close()
		plt.clf()

	return COEFCTRL, COEFEXP


cpdef correctReadCount(args):
	chromo = args[0]
	analysisStart = int(args[1])  # Genomic coordinates(starts from 1)
	analysisEnd = int(args[2])

	with py2bit.open(vari.FA) as faFile:
		chromoEnd = int(faFile.chroms(chromo))

	###### GENERATE A RESULT MATRIX
	fragStart = analysisStart - vari.FRAGLEN + 1
	fragEnd = analysisEnd + vari.FRAGLEN - 1
	shearStart = fragStart - 2
	shearEnd = fragEnd + 2

	if shearStart < 1:
		shearStart = 1
		fragStart = 3
		analysisStart = max(analysisStart, fragStart)
	if shearEnd > chromoEnd:
		shearEnd = chromoEnd
		fragEnd = shearEnd - 2
		analysisEnd = min(analysisEnd, fragEnd)


	## OUTPUT FILES
	subfinalCtrl = [0] * vari.CTRLBW_NUM
	subfinalCtrlNames = [0] * vari.CTRLBW_NUM
	subfinalExp = [0] * vari.EXPBW_NUM
	subfinalExpNames = [0] * vari.EXPBW_NUM

	for i in range(vari.CTRLBW_NUM):
		subfinalCtrl[i] = tempfile.NamedTemporaryFile(mode="w+t", suffix=".bed", dir=vari.OUTPUT_DIR, delete=False)
		subfinalCtrlNames[i] = subfinalCtrl[i].name
		subfinalCtrl[i].close()

	for i in range(vari.EXPBW_NUM):
		subfinalExp[i] = tempfile.NamedTemporaryFile(mode="w+t", suffix=".bed", dir=vari.OUTPUT_DIR, delete=False)
		subfinalExpNames[i] = subfinalExp[i].name
		subfinalExp[i].close()

	###### GET POSITIONS WHERE THE NUMBER OF FRAGMENTS > FILTERVALUE
	selectedIdx, highReadCountIdx, starts = selectIdx(chromo, analysisStart, analysisEnd)

	if len(selectedIdx) == 0:
		for i in range(vari.CTRLBW_NUM):
			os.remove(subfinalCtrlNames[i])
		for i in range(vari.EXPBW_NUM):
			os.remove(subfinalExpNames[i])

		return [ [None] * vari.CTRLBW_NUM, [None] * vari.EXPBW_NUM, chromo ]


	hdfFileName = vari.COVARI_DIR + "/" + vari.COVARI_NAME + "_" + chromo + ".hdf5"
	with h5py.File(hdfFileName, "r") as hdfFile:
		for rep in range(vari.CTRLBW_NUM):
			with pyBigWig.open(vari.CTRLBW_NAMES[rep]) as bwFile:
				rcArr = np.array(bwFile.values(chromo, analysisStart, analysisEnd))
				rcArr[np.isnan(rcArr) == True] = float(0)
				rcArr = rcArr / vari.CTRLSCALER[rep]

			prdvals = np.exp(
				np.nansum(
					(hdfFile['covari'][(analysisStart-3):(analysisEnd-3)] * vari.SELECT_COVARI) * vari.COEFCTRL[rep, 1:],
					axis=1
				) + vari.COEFCTRL[rep, 0]
			)
			prdvals[highReadCountIdx] = np.exp(
				np.nansum(
					(hdfFile['covari'][(analysisStart-3):(analysisEnd-3)][highReadCountIdx] * vari.SELECT_COVARI) * vari.COEFCTRL_HIGHRC[rep, 1:],
					axis=1
				) + vari.COEFCTRL_HIGHRC[rep, 0]
			)

			rcArr = rcArr - prdvals
			rcArr = rcArr[selectedIdx]

			idx = np.where( (rcArr < np.finfo(np.float32).min) | (rcArr > np.finfo(np.float32).max))
			if len(idx[0]) > 0:
				tempStarts = np.delete(starts, idx)
				rcArr = np.delete(rcArr, idx)
				if len(rcArr) > 0:
					writeBedFile(subfinalCtrlNames[rep], tempStarts, rcArr, analysisEnd)
				else:
					os.remove(subfinalCtrlNames[rep])
					subfinalCtrlNames[rep] = None
			else:
				if len(rcArr) > 0:
					writeBedFile(subfinalCtrlNames[rep], starts, rcArr, analysisEnd)
				else:
					os.remove(subfinalCtrlNames[rep])
					subfinalCtrlNames[rep] = None

		for rep in range(vari.EXPBW_NUM):
			with pyBigWig.open(vari.EXPBW_NAMES[rep]) as bwFile:
				rcArr = np.array(bwFile.values(chromo, analysisStart, analysisEnd))
				rcArr[np.isnan(rcArr) == True] = float(0)
				rcArr = rcArr / vari.EXPSCALER[rep]

			prdvals = np.exp(
				np.nansum(
					(hdfFile['covari'][(analysisStart-3):(analysisEnd-3)] * vari.SELECT_COVARI) * vari.COEFEXP[rep, 1:],
					axis=1
				) + vari.COEFEXP[rep, 0]
			)
			prdvals[highReadCountIdx] = np.exp(
				np.nansum(
					(hdfFile['covari'][(analysisStart-3):(analysisEnd-3)][highReadCountIdx] * vari.SELECT_COVARI) * vari.COEFEXP_HIGHRC[rep, 1:],
					axis=1
				) + vari.COEFEXP_HIGHRC[rep, 0]
			)

			rcArr = rcArr - prdvals
			rcArr = rcArr[selectedIdx]

			idx = np.where( (rcArr < np.finfo(np.float32).min) | (rcArr > np.finfo(np.float32).max))
			if len(idx[0]) > 0:
				tempStarts = np.delete(starts, idx)
				rcArr = np.delete(rcArr, idx)
				if len(rcArr) > 0:
					writeBedFile(subfinalExpNames[rep], tempStarts, rcArr, analysisEnd)
				else:
					os.remove(subfinalExpNames[rep])
					subfinalExpNames[rep] = None
			else:
				if len(rcArr) > 0:
					writeBedFile(subfinalExpNames[rep], starts, rcArr, analysisEnd)
				else:
					os.remove(subfinalExpNames[rep])
					subfinalExpNames[rep] = None

	return [subfinalCtrlNames, subfinalExpNames, chromo]

cpdef selectIdx(chromo, analysisStart, analysisEnd):
	ctrlReadCounts = []
	for rep in range(vari.CTRLBW_NUM):
		with pyBigWig.open(vari.CTRLBW_NAMES[rep]) as bwFile:
			temp = np.array(bwFile.values(chromo, analysisStart, analysisEnd))
			temp[np.where(np.isnan(temp) == True)] = float(0)

		ctrlReadCounts.append(temp.tolist())

		if rep == 0:
			readCountSum = temp
			highReadCountIdx = np.where(temp > vari.HIGHRC)[0]
		else:
			readCountSum = readCountSum + temp

	ctrlReadCounts = np.nanmean(ctrlReadCounts, axis=0)
	idx1 = np.where(ctrlReadCounts > 0)[0].tolist()

	expRC = []
	for rep in range(vari.EXPBW_NUM):
		with pyBigWig.open(vari.EXPBW_NAMES[rep]) as bwFile:
			temp = np.array(bwFile.values(chromo, analysisStart, analysisEnd))
			temp[np.where(np.isnan(temp) == True)] = float(0)

		expRC.append(temp.tolist())

		readCountSum = readCountSum + temp

	expRC = np.nanmean(expRC, axis=0)
	idx2 = np.where(expRC > 0)[0].tolist()
	idx3 = np.where(readCountSum > vari.FILTERVALUE)[0].tolist()

	idxTemp = np.intersect1d(idx1, idx2)
	idx = np.intersect1d(idxTemp, idx3)

	if len(idx) == 0:
		return np.array([]), np.array([]), np.array([])

	starts = np.array(list(range(analysisStart, analysisEnd)))[idx]

	return idx, highReadCountIdx, starts

cpdef writeBedFile(subfileName, tempStarts, tempSignalvals, analysisEnd):
	subfile = open(subfileName, "w")

	tempSignalvals = tempSignalvals.astype(int)
	numIdx = len(tempSignalvals)

	idx = 0
	prevStart = tempStarts[idx]
	prevReadCount = tempSignalvals[idx]
	line = [prevStart, (prevStart + vari.BINSIZE), prevReadCount]
	if numIdx == 1:
		subfile.write('\t'.join([str(x) for x in line]) + "\n")
		subfile.close()
		return

	idx = 1
	while idx < numIdx:
		currStart = tempStarts[idx]
		currReadCount = tempSignalvals[idx]

		if (currStart == (prevStart + vari.BINSIZE)) and (currReadCount == prevReadCount):
			line[1] = currStart + vari.BINSIZE
			prevStart = currStart
			prevReadCount = currReadCount
			idx = idx + 1
		else:
			### End a current line
			subfile.write('\t'.join([str(x) for x in line]) + "\n")

			### Start a new line
			line = [currStart, (currStart+vari.BINSIZE), currReadCount]
			prevStart = currStart
			prevReadCount = currReadCount
			idx = idx + 1

		if idx == numIdx:
			line[1] = analysisEnd
			subfile.write('\t'.join([str(x) for x in line]) + "\n")
			subfile.close()
			break