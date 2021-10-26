# cython: language_level=3

import os
import h5py
import numpy as np
import statsmodels.api as sm

from CRADLE.CorrectBias import vari
from CRADLE.correctbiasutils import vari as commonVari

cpdef performRegression(covariFiles, scatterplotSamples):
	### Read covariates values (X)
	xNumRows = 0
	for i in range(len(covariFiles)):
		xNumRows = xNumRows + int(covariFiles[i][1])
	xNumCols = vari.COVARI_NUM + 1

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
			while colPtr < vari.COVARI_NUM:
				XView[rowIdx + rowPtr, colPtr+1] = float(temp[colPtr])
				colPtr = colPtr + 1
			rowIdx = rowIdx + 1
		rowPtr = rowPtr + int(covariFiles[fileIdx][1])

		f.close()
		os.remove(subfileName)


	### COEFFICIENTS
	COEFCTRL = np.zeros((commonVari.CTRLBW_NUM, (vari.COVARI_NUM+1)), dtype=np.float64)
	COEFEXP = np.zeros((commonVari.EXPBW_NUM, (vari.COVARI_NUM+1)), dtype=np.float64)

	readCounts = np.zeros(xNumRows, dtype=np.float64)
	cdef double [:] readCountsView = readCounts

	cdef int ptr
	cdef int rcIdx

	ctrlPlotValues = {}
	experiPlotValues = {}

	for rep in range(commonVari.CTRLBW_NUM):
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

		ctrlPlotValues[commonVari.CTRLBW_NAMES[rep]] = (readCounts[scatterplotSamples], model.fittedvalues[scatterplotSamples])

	for rep in range(commonVari.EXPBW_NUM):
		ptr = 0
		for fileIdx in range(len(covariFiles)):
			subfileName = covariFiles[fileIdx][rep + 2 + commonVari.CTRLBW_NUM]
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

		experiPlotValues[commonVari.EXPBW_NAMES[rep]] = (readCounts[scatterplotSamples], model.fittedvalues[scatterplotSamples])

	return COEFCTRL, COEFEXP, ctrlPlotValues, experiPlotValues