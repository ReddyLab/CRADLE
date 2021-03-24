import os
import tempfile
import warnings
import h5py
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
import py2bit
import pyBigWig

matplotlib.use('Agg')

COEF_LEN = 7

cdef class TrainingRegion:
	cdef public str chromo
	cdef public int analysisStart, analysisEnd

	def __init__(self, chromo, analysisStart, analysisEnd):
		self.chromo = chromo
		self.analysisStart = analysisStart
		self.analysisEnd = analysisEnd

cdef class TrainingSet:
	cdef public int xRowCount
	cdef public list trainingRegions

	def __init__(self, list trainingRegions, int xRowCount):
		self.trainingRegions = trainingRegions
		self.xRowCount = xRowCount

	def __iter__(self):
		def regionGenerator():
			for region in self.trainingRegions:
				yield region

		return regionGenerator()

cpdef performRegression(trainingSet, scatterplotSamples, covariates, ctrlBWNames, ctrlScaler, experiBWNames, experiScaler, outputDir, outputLabel):
	xColumnCount = covariates.num + 1

	#### Get X matrix
	cdef double [:,:] xView = np.ones((trainingSet.xRowCount, xColumnCount), dtype=np.float64)

	cdef int currentRow = 0
	cdef int pos
	cdef int i, j

	for trainingRegion in trainingSet:
		hdfFileName = covariates.hdfFileName(trainingRegion.chromo)
		with h5py.File(hdfFileName, "r") as hdfFile:
			pos = trainingRegion.analysisStart
			while pos < trainingRegion.analysisEnd:
				temp = hdfFile['covari'][pos - 3] * covariates.selected
				temp = temp[np.isnan(temp) == False]

				j = 0
				i = currentRow + pos - trainingRegion.analysisStart
				while j < covariates.num:
					xView[i, j + 1] = temp[j]
					j += 1
				pos += 1
			currentRow += (trainingRegion.analysisEnd - trainingRegion.analysisStart)
	#### END Get X matrix

	#### Initialize COEF arrays
	COEFCTRL = np.zeros((len(ctrlBWNames), COEF_LEN), dtype=np.float64)
	COEFEXPR = np.zeros((len(experiBWNames), COEF_LEN), dtype=np.float64)

	for i, bwFileName in enumerate(ctrlBWNames):
		readCounts = getReadCounts(bwFileName, trainingSet, ctrlScaler[i])
		model = buildModel(readCounts, xView)

		COEFCTRL[i, :] = getCoefs(model, covariates)

		figName = figureFileName(outputDir, bwFileName, outputLabel)
		plot(readCounts, model.fittedvalues, figName, scatterplotSamples)

	for i, bwFileName in enumerate(experiBWNames):
		readCounts = getReadCounts(bwFileName, trainingSet, experiScaler[i])
		model = buildModel(readCounts, xView)

		COEFEXPR[i, :] = getCoefs(model, covariates)

		figName = figureFileName(outputDir, bwFileName, outputLabel)
		plot(readCounts, model.fittedvalues, figName, scatterplotSamples)

	return COEFCTRL, COEFEXPR

cpdef figureFileName(outputDir, bwFileName, outputLabel):
	bwName = '.'.join(bwFileName.rsplit('/', 1)[-1].split(".")[:-1])
	return os.path.join(outputDir, f"fit_{bwName}_{outputLabel}.png")

cpdef getReadCounts(bwFileName, trainingSet, scaler):
	cdef double [:] readCounts = np.zeros(trainingSet.xRowCount, dtype=np.float64)
	cdef int ptr
	cdef int posIdx

	with pyBigWig.open(bwFileName) as bwFile:
		ptr = 0
		for trainingRegion in trainingSet:
			regionReadCounts = np.array(
				bwFile.values(trainingRegion.chromo, trainingRegion.analysisStart, trainingRegion.analysisEnd)
			)
			regionReadCounts[np.isnan(regionReadCounts) == True] = 0.0
			regionReadCounts = regionReadCounts / scaler

			numPos = trainingRegion.analysisEnd - trainingRegion.analysisStart
			posIdx = 0
			while posIdx < numPos:
				readCounts[ptr + posIdx] = regionReadCounts[posIdx]
				posIdx += 1

			ptr += numPos

	return readCounts

cpdef buildModel(readCounts, xView):
	#### do regression
	return sm.GLM(np.array(readCounts).astype(int), np.array(xView), family=sm.families.Poisson(link=sm.genmod.families.links.log)).fit()

cpdef getCoefs(model, covariates):
	coef = np.zeros(COEF_LEN, dtype=np.float64)

	coef[0] = model.params[0]

	paramIdx = 1
	for j in range(1, COEF_LEN):
		if np.isnan(covariates.selected[j - 1]) == True:
			coef[j] = np.nan
		else:
			coef[j] = model.params[paramIdx]
			paramIdx += 1

	return coef

cpdef plot(yView, fittedvalues, figName, scatterplotSamples):
	corr = np.corrcoef(fittedvalues, np.array(yView))[0, 1]
	corr = np.round(corr, 2)
	maxi1 = np.nanmax(fittedvalues[scatterplotSamples])
	maxi2 = np.nanmax(np.array(yView)[scatterplotSamples])
	maxi = max(maxi1, maxi2)

	plt.plot(np.array(yView)[scatterplotSamples], fittedvalues[scatterplotSamples], color='g', marker='s', alpha=0.01)
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

cpdef correctReadCount(taskArgs, covariates, faFileName, ctrlBWNames, ctrlScaler, COEFCTRL, COEFCTRL_HIGHRC, experiBWNames, experiScaler, COEFEXP, COEFEXP_HIGHRC, highRC, minFragFilterValue, binsize, outputDir):
	chromo = taskArgs[0]
	analysisStart = int(taskArgs[1])  # Genomic coordinates(starts from 1)
	analysisEnd = int(taskArgs[2])

	ctrlBWNum = len(ctrlBWNames)
	experiBWNum = len(experiBWNames)

	with py2bit.open(faFileName) as faFile:
		chromoEnd = int(faFile.chroms(chromo))

	###### GENERATE A RESULT MATRIX
	fragStart = analysisStart - covariates.fragLen + 1
	fragEnd = analysisEnd + covariates.fragLen - 1
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
	subfinalCtrl = [0] * ctrlBWNum
	subfinalCtrlNames = [0] * ctrlBWNum
	subfinalExp = [0] * experiBWNum
	subfinalExpNames = [0] * experiBWNum

	for i in range(ctrlBWNum):
		subfinalCtrl[i] = tempfile.NamedTemporaryFile(mode="w+t", suffix=".bed", dir=outputDir, delete=False)
		subfinalCtrlNames[i] = subfinalCtrl[i].name
		subfinalCtrl[i].close()

	for i in range(experiBWNum):
		subfinalExp[i] = tempfile.NamedTemporaryFile(mode="w+t", suffix=".bed", dir=outputDir, delete=False)
		subfinalExpNames[i] = subfinalExp[i].name
		subfinalExp[i].close()

	###### GET POSITIONS WHERE THE NUMBER OF FRAGMENTS > MIN_FRAGNUM_FILTER_VALUE
	selectedIdx, highReadCountIdx, starts = selectIdx(chromo, analysisStart, analysisEnd, ctrlBWNames, experiBWNames, highRC, minFragFilterValue)

	if len(selectedIdx) == 0:
		for i in range(ctrlBWNum):
			os.remove(subfinalCtrlNames[i])
		for i in range(experiBWNum):
			os.remove(subfinalExpNames[i])

		return [ [None] * ctrlBWNum, [None] * experiBWNum, chromo ]


	hdfFileName = covariates.hdfFileName(chromo)
	with h5py.File(hdfFileName, "r") as hdfFile:
		for rep in range(ctrlBWNum):
			with pyBigWig.open(ctrlBWNames[rep]) as bwFile:
				rcArr = np.array(bwFile.values(chromo, analysisStart, analysisEnd))
				rcArr[np.isnan(rcArr) == True] = float(0)
				rcArr = rcArr / ctrlScaler[rep]

			prdvals = np.exp(
				np.nansum(
					(hdfFile['covari'][(analysisStart-3):(analysisEnd-3)] * covariates.selected) * COEFCTRL[rep, 1:],
					axis=1
				) + COEFCTRL[rep, 0]
			)
			prdvals[highReadCountIdx] = np.exp(
				np.nansum(
					(hdfFile['covari'][(analysisStart-3):(analysisEnd-3)][highReadCountIdx] * covariates.selected) * COEFCTRL_HIGHRC[rep, 1:],
					axis=1
				) + COEFCTRL_HIGHRC[rep, 0]
			)

			rcArr = rcArr - prdvals
			rcArr = rcArr[selectedIdx]

			idx = np.where( (rcArr < np.finfo(np.float32).min) | (rcArr > np.finfo(np.float32).max))
			if len(idx[0]) > 0:
				tempStarts = np.delete(starts, idx)
				rcArr = np.delete(rcArr, idx)
				if len(rcArr) > 0:
					writeBedFile(subfinalCtrlNames[rep], tempStarts, rcArr, analysisEnd, binsize)
				else:
					os.remove(subfinalCtrlNames[rep])
					subfinalCtrlNames[rep] = None
			else:
				if len(rcArr) > 0:
					writeBedFile(subfinalCtrlNames[rep], starts, rcArr, analysisEnd, binsize)
				else:
					os.remove(subfinalCtrlNames[rep])
					subfinalCtrlNames[rep] = None

		for rep in range(experiBWNum):
			with pyBigWig.open(experiBWNames[rep]) as bwFile:
				rcArr = np.array(bwFile.values(chromo, analysisStart, analysisEnd))
				rcArr[np.isnan(rcArr) == True] = float(0)
				rcArr = rcArr / experiScaler[rep]

			prdvals = np.exp(
				np.nansum(
					(hdfFile['covari'][(analysisStart-3):(analysisEnd-3)] * covariates.selected) * COEFEXP[rep, 1:],
					axis=1
				) + COEFEXP[rep, 0]
			)
			prdvals[highReadCountIdx] = np.exp(
				np.nansum(
					(hdfFile['covari'][(analysisStart-3):(analysisEnd-3)][highReadCountIdx] * covariates.selected) * COEFEXP_HIGHRC[rep, 1:],
					axis=1
				) + COEFEXP_HIGHRC[rep, 0]
			)

			rcArr = rcArr - prdvals
			rcArr = rcArr[selectedIdx]

			idx = np.where( (rcArr < np.finfo(np.float32).min) | (rcArr > np.finfo(np.float32).max))
			if len(idx[0]) > 0:
				tempStarts = np.delete(starts, idx)
				rcArr = np.delete(rcArr, idx)
				if len(rcArr) > 0:
					writeBedFile(subfinalExpNames[rep], tempStarts, rcArr, analysisEnd, binsize)
				else:
					os.remove(subfinalExpNames[rep])
					subfinalExpNames[rep] = None
			else:
				if len(rcArr) > 0:
					writeBedFile(subfinalExpNames[rep], starts, rcArr, analysisEnd, binsize)
				else:
					os.remove(subfinalExpNames[rep])
					subfinalExpNames[rep] = None

	return [subfinalCtrlNames, subfinalExpNames, chromo]

cpdef selectIdx(chromo, analysisStart, analysisEnd, ctrlBWNames, experiBWNames, highRC, minFragFilterValue):
	ctrlReadCounts = []
	for rep in range(len(ctrlBWNames)):
		with pyBigWig.open(ctrlBWNames[rep]) as bwFile:
			temp = np.array(bwFile.values(chromo, analysisStart, analysisEnd))
			temp[np.where(np.isnan(temp) == True)] = float(0)

		ctrlReadCounts.append(temp.tolist())

		if rep == 0:
			readCountSum = temp
			highReadCountIdx = np.where(temp > highRC)[0]
		else:
			readCountSum = readCountSum + temp

	ctrlReadCounts = np.nanmean(ctrlReadCounts, axis=0)
	idx1 = np.where(ctrlReadCounts > 0)[0].tolist()

	expRC = []
	for rep in range(len(experiBWNames)):
		with pyBigWig.open(experiBWNames[rep]) as bwFile:
			temp = np.array(bwFile.values(chromo, analysisStart, analysisEnd))
			temp[np.where(np.isnan(temp) == True)] = float(0)

		expRC.append(temp.tolist())

		readCountSum = readCountSum + temp

	expRC = np.nanmean(expRC, axis=0)
	idx2 = np.where(expRC > 0)[0].tolist()
	idx3 = np.where(readCountSum > minFragFilterValue)[0].tolist()

	idxTemp = np.intersect1d(idx1, idx2)
	idx = np.intersect1d(idxTemp, idx3)

	if len(idx) == 0:
		return np.array([]), np.array([]), np.array([])

	starts = np.array(list(range(analysisStart, analysisEnd)))[idx]

	return idx, highReadCountIdx, starts


cpdef writeBedFile(subfileName, tempStarts, tempSignalvals, analysisEnd, binsize):
	subfile = open(subfileName, "w")

	tempSignalvals = tempSignalvals.astype(int)
	numIdx = len(tempSignalvals)

	idx = 0
	prevStart = tempStarts[idx]
	prevReadCount = tempSignalvals[idx]
	line = [prevStart, (prevStart + binsize), prevReadCount]
	if numIdx == 1:
		subfile.write('\t'.join([str(x) for x in line]) + "\n")
		subfile.close()
		return

	idx = 1
	while idx < numIdx:
		currStart = tempStarts[idx]
		currReadCount = tempSignalvals[idx]

		if (currStart == (prevStart + binsize)) and (currReadCount == prevReadCount):
			line[1] = currStart + binsize
			prevStart = currStart
			prevReadCount = currReadCount
			idx = idx + 1
		else:
			### End a current line
			subfile.write('\t'.join([str(x) for x in line]) + "\n")

			### Start a new line
			line = [currStart, (currStart+binsize), currReadCount]
			prevStart = currStart
			prevReadCount = currReadCount
			idx = idx + 1

		if idx == numIdx:
			line[1] = analysisEnd
			subfile.write('\t'.join([str(x) for x in line]) + "\n")
			subfile.close()
			break
