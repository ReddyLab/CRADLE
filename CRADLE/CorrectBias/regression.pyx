# cython: language_level=3

import os
import h5py
import numpy as np
import statsmodels.api as sm


cpdef performRegression(covariFiles, scatterplotSamples, globalVars):
	### Read covariates values (X)
	xNumRows = 0
	for i in range(len(covariFiles)):
		xNumRows = xNumRows + int(covariFiles[i][1])
	xNumCols = globalVars["covariNum"] + 1

	cdef double [:,:] XView = np.ones((xNumRows, xNumCols), dtype=np.float64)

	cdef int rowPtr = 0
	cdef int rowIdx
	cdef int colPtr

	for fileIdx in range(len(covariFiles)):
		subfileName = covariFiles[fileIdx][0]
		f = h5py.File(subfileName, "r")

		rowIdx = 0
		while rowIdx < f['covari'].shape[0]:
			temp = f['covari'][rowIdx]
			colPtr = 0
			while colPtr < globalVars["covariNum"]:
				XView[rowIdx + rowPtr, colPtr+1] = float(temp[colPtr])
				colPtr = colPtr + 1
			rowIdx = rowIdx + 1
		rowPtr = rowPtr + int(covariFiles[fileIdx][1])

		f.close()
		os.remove(subfileName)


	### COEFFICIENTS
	COEFCTRL = np.zeros((globalVars["ctrlbwNum"], (globalVars["covariNum"]+1)), dtype=np.float64)
	COEFEXP = np.zeros((globalVars["expbwNum"], (globalVars["covariNum"]+1)), dtype=np.float64)

	readCounts = np.zeros(xNumRows, dtype=np.float64)
	cdef double [:] readCountsView = readCounts

	cdef int ptr
	cdef int rcIdx

	ctrlPlotValues = {}
	experiPlotValues = {}

	for rep in range(globalVars["ctrlbwNum"]):
		ptr = 0

		for fileIdx in range(len(covariFiles)):
			subfileName = covariFiles[fileIdx][rep+2]
			f = h5py.File(subfileName, "r")

			rcIdx = 0
			while rcIdx < f['Y'].shape[0]:
				readCountsView[rcIdx+ptr] = float(f['Y'][rcIdx])
				rcIdx = rcIdx + 1

			ptr = ptr + int(f['Y'].shape[0])

			f.close()
			os.remove(subfileName)


		deleteIdx = np.where( (readCounts < np.finfo(np.float32).min) | (readCounts > np.finfo(np.float32).max))[0]
		loglink = sm.genmod.families.links.log()
		poisson = sm.families.Poisson(link=loglink)
		if len(deleteIdx) != 0:
			model = sm.GLM(
				np.delete(readCounts.astype(int), deleteIdx),
				np.delete(np.array(XView), deleteIdx, axis=0),
				family=poisson
			).fit()
		else:
			model = sm.GLM(readCounts.astype(int), np.array(XView), family=poisson).fit()

		coef = model.params
		COEFCTRL[rep, ] = coef

		ctrlPlotValues[globalVars["ctrlbwNames"][rep]] = (readCounts[scatterplotSamples], model.fittedvalues[scatterplotSamples])

	for rep in range(globalVars["expbwNum"]):
		ptr = 0
		for fileIdx in range(len(covariFiles)):
			subfileName = covariFiles[fileIdx][rep + 2 + globalVars["ctrlbwNum"]]
			f = h5py.File(subfileName, "r")

			rcIdx = 0
			while rcIdx < f['Y'].shape[0]:
				readCountsView[rcIdx+ptr] = float(f['Y'][rcIdx])
				rcIdx = rcIdx + 1

			ptr = ptr + int(f['Y'].shape[0])

			f.close()
			os.remove(subfileName)

		deleteIdx = np.where( (readCounts < np.finfo(np.float32).min) | (readCounts > np.finfo(np.float32).max))[0]
		loglink = sm.genmod.families.links.log()
		poisson = sm.families.Poisson(link=loglink)
		if len(deleteIdx) != 0:
			model = sm.GLM(
				np.delete(readCounts.astype(int), deleteIdx),
				np.delete(np.array(XView), deleteIdx, axis=0),
				family=poisson
			).fit()
		else:
			model = sm.GLM(
				readCounts.astype(int),
				np.array(XView),
				family=poisson
			).fit()

		coef = model.params
		COEFEXP[rep, ] = coef

		experiPlotValues[globalVars["expbwNames"][rep]] = (readCounts[scatterplotSamples], model.fittedvalues[scatterplotSamples])

	return COEFCTRL, COEFEXP, ctrlPlotValues, experiPlotValues