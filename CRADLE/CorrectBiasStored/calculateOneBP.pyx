# cython: language_level=3

import os
import tempfile
import warnings
import h5py
import numpy as np
import statsmodels.api as sm
import py2bit
import pyBigWig

from CRADLE.correctbiasutils import TrainingRegion, TrainingSet, marshalFile
from CRADLE.correctbiasutils.cython import coalesceSections

COEF_LEN = 7

cpdef performRegression(trainingSet, covariates, ctrlBWNames, ctrlScaler, experiBWNames, experiScaler, scatterplotSamples):
	xColumnCount = covariates.num + 1

	#### Get X matrix
	xView = np.ones((trainingSet.xRowCount, xColumnCount), dtype=np.float64)

	cdef int currentRow = 0

	for trainingRegion in trainingSet:
		region_length = trainingRegion.analysisEnd - trainingRegion.analysisStart
		hdfFileName = covariates.hdfFileName(trainingRegion.chromo)
		with h5py.File(hdfFileName, "r") as hdfFile:
			non_selected_rows = np.where(np.isnan(covariates.selected))
			temp = hdfFile['covari'][trainingRegion.analysisStart - 3:trainingRegion.analysisEnd - 3]
			temp = np.delete(temp, non_selected_rows, 1)
			xView[currentRow:currentRow + region_length, 1:xColumnCount] = temp
			currentRow += region_length
	#### END Get X matrix

	#### Initialize COEF arrays
	COEFCTRL = np.zeros((len(ctrlBWNames), COEF_LEN), dtype=np.float64)
	COEFEXPR = np.zeros((len(experiBWNames), COEF_LEN), dtype=np.float64)

	ctrlPlotValues = {}
	experiPlotValues = {}

	for i, bwFileName in enumerate(ctrlBWNames):
		rawReadCounts = readCountData(bwFileName, trainingSet)
		readCounts = getReadCounts(rawReadCounts, trainingSet.xRowCount, ctrlScaler[i])
		model = buildModel(readCounts, xView)

		COEFCTRL[i, :] = getCoefs(model.params, covariates.selected)

		ctrlPlotValues[bwFileName] = (readCounts[scatterplotSamples], model.fittedvalues[scatterplotSamples])

	for i, bwFileName in enumerate(experiBWNames):
		rawReadCounts = readCountData(bwFileName, trainingSet)
		readCounts = getReadCounts(rawReadCounts, trainingSet.xRowCount, experiScaler[i])
		model = buildModel(readCounts, xView)

		COEFEXPR[i, :] = getCoefs(model.params, covariates.selected)

		experiPlotValues[bwFileName] = (readCounts[scatterplotSamples], model.fittedvalues[scatterplotSamples])

	return COEFCTRL, COEFEXPR, ctrlPlotValues, experiPlotValues

def readCountData(bwFileName, trainingSet):
	with pyBigWig.open(bwFileName) as bwFile:
		if pyBigWig.numpy == 1:
			for trainingRegion in trainingSet:
				regionReadCounts = bwFile.values(trainingRegion.chromo, trainingRegion.analysisStart, trainingRegion.analysisEnd, numpy=True)
				regionLength = trainingRegion.analysisEnd - trainingRegion.analysisStart
				yield regionReadCounts, regionLength
		else:
			for trainingRegion in trainingSet:
				regionReadCounts = np.array(
					bwFile.values(trainingRegion.chromo, trainingRegion.analysisStart, trainingRegion.analysisEnd)
				)
				regionLength = trainingRegion.analysisEnd - trainingRegion.analysisStart
				yield regionReadCounts, regionLength

cpdef getReadCounts(rawReadCounts, rowCount, scaler):
	cdef double [:] readCountsView
	cdef int ptr
	cdef int posIdx

	readCounts = np.zeros(rowCount, dtype=np.float64)
	readCountsView = readCounts

	ptr = 0
	for regionReadCounts, regionLength in rawReadCounts:
		regionReadCounts[np.isnan(regionReadCounts)] = 0.0
		regionReadCounts = regionReadCounts / scaler

		posIdx = 0
		while posIdx < regionLength:
			readCountsView[ptr + posIdx] = regionReadCounts[posIdx]
			posIdx += 1

		ptr += regionLength

	return readCounts

cpdef buildModel(readCounts, xView):
	#### do regression
	return sm.GLM(np.array(readCounts).astype(int), np.array(xView), family=sm.families.Poisson(link=sm.genmod.families.links.log)).fit()

cpdef getCoefs(modelParams, selectedCovariates):
	coef = np.zeros(COEF_LEN, dtype=np.float64)

	coef[0] = modelParams[0]

	paramIdx = 1
	for j in range(1, COEF_LEN):
		if np.isnan(selectedCovariates[j - 1]):
			coef[j] = np.nan
		else:
			coef[j] = modelParams[paramIdx]
			paramIdx += 1

	return coef

cpdef correctReadCount(tasks, covariates, faFileName, ctrlBWNames, ctrlScaler, COEFCTRL, COEFCTRL_HIGHRC, experiBWNames, experiScaler, COEFEXP, COEFEXP_HIGHRC, highRC, minFragFilterValue, binsize, outputDir):
	correctedCtrlReadCounts = [[] for _ in range(len(ctrlBWNames))]
	correctedExprReadCounts = [[] for _ in range(len(experiBWNames))]

	for taskArgs in tasks:
		chromo = taskArgs[0]
		analysisStart = int(taskArgs[1])  # Genomic coordinates(starts from 1)
		analysisEnd = int(taskArgs[2])

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

		###### GET POSITIONS WHERE THE NUMBER OF FRAGMENTS > MIN_FRAGNUM_FILTER_VALUE
		ctrlSpecificIdx, experiSpecificIdx, highReadCountIdx = selectIdx(chromo, analysisStart, analysisEnd, ctrlBWNames, experiBWNames, highRC, minFragFilterValue)

		## OUTPUT FILES
		subfinalCtrlReadCounts = [None] * len(ctrlBWNames)
		subfinalExperiReadCounts = [None] * len(experiBWNames)
		hdfFileName = covariates.hdfFileName(chromo)
		with h5py.File(hdfFileName, "r") as hdfFile:
			for rep, bwName in enumerate(ctrlBWNames):
				if len(ctrlSpecificIdx[rep]) == 0:
					correctedCtrlReadCounts[rep].append((chromo, None))
					continue

				with pyBigWig.open(bwName) as bwFile:
					rcArr = np.array(bwFile.values(chromo, analysisStart, analysisEnd))
					rcArr[np.isnan(rcArr)] = 0.0
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
				rcArr = rcArr[ctrlSpecificIdx[rep]]
				starts = np.array(list(range(analysisStart, analysisEnd)))[ctrlSpecificIdx[rep]]
				ctrlSpecificIdx[rep] = []

				idx = np.where( (rcArr < np.finfo(np.float32).min) | (rcArr > np.finfo(np.float32).max))
				starts = np.delete(starts, idx)
				rcArr = np.delete(rcArr, idx)
				if len(rcArr) > 0:
					subfinalReadCounts = coalesceSections(starts, rcArr, analysisEnd, binsize)
					correctedCtrlReadCounts[rep].append((chromo, subfinalReadCounts))
				else:
					correctedCtrlReadCounts[rep].append((chromo, None))

			for rep, bwName in enumerate(experiBWNames):
				if len(experiSpecificIdx[rep]) == 0:
					correctedExprReadCounts[rep].append((chromo, None))
					continue

				with pyBigWig.open(bwName) as bwFile:
					rcArr = np.array(bwFile.values(chromo, analysisStart, analysisEnd))
					rcArr[np.isnan(rcArr)] = 0.0
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
				rcArr = rcArr[experiSpecificIdx[rep]]
				starts = np.array(list(range(analysisStart, analysisEnd)))[experiSpecificIdx[rep]]
				experiSpecificIdx[rep] = []

				idx = np.where( (rcArr < np.finfo(np.float32).min) | (rcArr > np.finfo(np.float32).max))
				starts = np.delete(starts, idx)
				rcArr = np.delete(rcArr, idx)
				if len(rcArr) > 0:
					subfinalReadCounts = coalesceSections(starts, rcArr, analysisEnd, binsize)
					correctedExprReadCounts[rep].append( (chromo, subfinalReadCounts) )
				else:
					correctedExprReadCounts[rep].append((chromo, None) )

	ctrlNames = [marshalFile(outputDir, readCounts) for readCounts in correctedCtrlReadCounts]
	exprNames = [marshalFile(outputDir, readCounts) for readCounts in correctedExprReadCounts]

	return [ctrlNames, exprNames]

cpdef selectIdx(chromo, analysisStart, analysisEnd, ctrlBWNames, experiBWNames, highRC, minFragFilterValue):
	readCountSums = np.zeros(analysisEnd - analysisStart, dtype=np.float64)
	meanMinFragFilterValue = int(np.round(minFragFilterValue / (len(ctrlBWNames) + len(experiBWNames))))

	ctrlSpecificIdx = []
	experiSpecificIdx = []
	for rep, bwName in enumerate(ctrlBWNames):
		with pyBigWig.open(bwName) as bwFile:
			readCounts = np.array(bwFile.values(chromo, analysisStart, analysisEnd))
			readCounts[np.isnan(readCounts)] = 0.0

		if rep == 0:
			highReadCountIdx = np.where(readCounts > highRC)[0]

		readCountSums += readCounts

		ctrlSpecificIdx.append( list(np.where(readCounts >= meanMinFragFilterValue)[0]) )

	for rep, bwName in enumerate(experiBWNames):
		with pyBigWig.open(bwName) as bwFile:
			readCounts = np.array(bwFile.values(chromo, analysisStart, analysisEnd))
			readCounts[np.isnan(readCounts)] = 0.0

		readCountSums += readCounts

		experiSpecificIdx.append( list(np.where(readCounts >= meanMinFragFilterValue)[0])  )

	idx = np.where(readCountSums > minFragFilterValue)[0].tolist()
	if len(idx) == 0:
		ctrlSpecificIdx = [ [] for _ in range(len(ctrlBWNames)) ]
		experiSpecificIdx = [ [] for _ in range(len(experiBWNames)) ]
		highReadCountIdx = []
		return ctrlSpecificIdx, experiSpecificIdx, highReadCountIdx

	highReadCountIdx = np.intersect1d(highReadCountIdx, idx)

	for rep in range(len(ctrlBWNames)):
		ctrlSpecificIdx[rep] = list(np.intersect1d(ctrlSpecificIdx[rep], idx))

	for rep in range(len(experiBWNames)):
		experiSpecificIdx[rep] = list(np.intersect1d(experiSpecificIdx[rep], idx))

	return ctrlSpecificIdx, experiSpecificIdx, highReadCountIdx


