# cython: language_level=3

import struct
import tempfile
import h5py
import numpy as np
import pyBigWig

from CRADLE.correctbiasutils import CORRECTED_RC_TEMP_FILE_STRUCT_FORMAT, SONICATION_SHEAR_BIAS_OFFSET, START_INDEX_ADJUSTMENT, ChromoRegion, ChromoRegionSet, marshalFile
from CRADLE.correctbiasutils.cython import coalesceSections

# The covariate values stored in the HDF files start at index 0 (0-index, obviously)
# The lowest start point for an analysis region is 3 (1-indexed), so we need to subtract
# 3 from the analysis start and end points to match them up with correct covariate values
# in the HDF files.
cdef int COVARIATE_FILE_INDEX_OFFSET = 3

cpdef correctReadCounts(regions, covariates, chromoEnds, ctrlBWNames, ctrlScaler, COEFCTRL, COEFCTRL_HIGHRC, experiBWNames, experiScaler, COEFEXP, COEFEXP_HIGHRC, highRC, minFragFilterValue, binsize, outputDir):
	cdef int reg = 0 # region index
	cdef int regCount = len(regions) # region count
	cdef int analysisStart
	cdef int analysisEnd
	cdef int fragLen = covariates.fragLen
	cdef int fragRegionStart
	cdef int fragRegionEnd
	cdef int shearStart
	cdef int shearEnd
	cdef int chromoEnd
	cdef int startIndexAdjustment = START_INDEX_ADJUSTMENT
	cdef int soncationShearBiasOffset = SONICATION_SHEAR_BIAS_OFFSET

	cdef int rep = 0 # replicate index
	cdef int ctrlBWNameCount = len(ctrlBWNames)
	cdef int experiBWNameCount = len(experiBWNames)

	cdef int outIdx # index used for iterating over read counts when outputting them to a temp file
	cdef int coalescedSectionCount

	correctedCtrlReadCountFiles = [tempfile.NamedTemporaryFile(mode="wb", suffix=".msl", dir=outputDir, delete=False) for _ in ctrlBWNames]
	correctedExprReadCountFiles = [tempfile.NamedTemporaryFile(mode="wb", suffix=".msl", dir=outputDir, delete=False) for _ in experiBWNames]

	while reg < regCount:
		chromo = regions[reg][0]
		analysisStart = regions[reg][1]  # Genomic coordinates(starts from 1)
		analysisEnd = regions[reg][2]
		chromoEnd = chromoEnds[chromo]

		chromoBytes = bytes(chromo, "utf-8") # used later when writing values to a file

		###### GENERATE A RESULT MATRIX

		#
		# Adjust analysis boundaries for use with covariate values file
		#
		# Define a region of fragments of length fragLen
		fragRegionStart = analysisStart - fragLen + startIndexAdjustment
		fragRegionEnd = analysisEnd + fragLen - startIndexAdjustment

		# Define a region that includes base pairs used to model shearing/sonication bias
		shearStart = fragRegionStart - soncationShearBiasOffset
		shearEnd = fragRegionEnd + soncationShearBiasOffset

		# Make sure the analysisStart and analysisEnd fall within the boundaries of the region
		# covariates have been precomputed for
		if shearStart < 1:
			fragRegionStart = soncationShearBiasOffset + startIndexAdjustment
			analysisStart = max(analysisStart, fragRegionStart)

		if shearEnd > chromoEnd:
			fragRegionEnd = chromoEnd - soncationShearBiasOffset
			analysisEnd = min(analysisEnd, fragRegionEnd)
		#
		# End adjust boundaries
		#
		###### GET POSITIONS WHERE THE NUMBER OF FRAGMENTS > MIN_FRAGNUM_FILTER_VALUE
		ctrlSpecificIdx, experiSpecificIdx, highReadCountIdx = selectIdx(chromo, analysisStart, analysisEnd, ctrlBWNames, experiBWNames, highRC, minFragFilterValue)

		## OUTPUT FILES
		covariateFileName = covariates.covariateFileName(chromo)
		with h5py.File(covariateFileName, "r") as covariateValues:
			rep = 0
			while rep < ctrlBWNameCount:
				if len(ctrlSpecificIdx[rep]) == 0:
					rep += 1
					continue

				with pyBigWig.open(ctrlBWNames[rep]) as bwFile:
					rcArr = np.array(bwFile.values(chromo, analysisStart, analysisEnd))
					rcArr[np.isnan(rcArr)] = 0.0
					rcArr = rcArr / ctrlScaler[rep]

				rawValues = covariateValues['covari'][(analysisStart - COVARIATE_FILE_INDEX_OFFSET):(analysisEnd - COVARIATE_FILE_INDEX_OFFSET)]
				rawCovariateValues = rawValues * covariates.selected

				prdvals = np.exp(
					np.nansum((rawCovariateValues) * COEFCTRL[rep, 1:], axis=1) + COEFCTRL[rep, 0]
				)
				prdvals[highReadCountIdx] = np.exp(
					np.nansum((rawCovariateValues[highReadCountIdx]) * COEFCTRL_HIGHRC[rep, 1:], axis=1) + COEFCTRL_HIGHRC[rep, 0]
				)

				rcArr = rcArr - prdvals
				rcArr = rcArr[ctrlSpecificIdx[rep]]
				starts = np.array(list(range(analysisStart, analysisEnd)))[ctrlSpecificIdx[rep]]
				ctrlSpecificIdx[rep] = []

				idx = np.where( (rcArr < np.finfo(np.float32).min) | (rcArr > np.finfo(np.float32).max))
				starts = np.delete(starts, idx)
				rcArr = np.delete(rcArr, idx)

				if len(rcArr) > 0:
					rcArr = np.rint(rcArr)
					coalescedSectionCount, startEntries, endEntries, valueEntries = coalesceSections(starts, rcArr, analysisEnd, binsize)
					writeCorrectedReads(correctedCtrlReadCountFiles[rep], chromoBytes, coalescedSectionCount, startEntries, endEntries, valueEntries)
				rep += 1

			rep = 0
			while rep < experiBWNameCount:
				if len(experiSpecificIdx[rep]) == 0:
					rep += 1
					continue

				with pyBigWig.open(experiBWNames[rep]) as bwFile:
					rcArr = np.array(bwFile.values(chromo, analysisStart, analysisEnd))
					rcArr[np.isnan(rcArr)] = 0.0
					rcArr = rcArr / experiScaler[rep]

				prdvals = np.exp(
					np.nansum(
						(covariateValues['covari'][(analysisStart - COVARIATE_FILE_INDEX_OFFSET):(analysisEnd - COVARIATE_FILE_INDEX_OFFSET)] * covariates.selected) * COEFEXP[rep, 1:],
						axis=1
					) + COEFEXP[rep, 0]
				)
				prdvals[highReadCountIdx] = np.exp(
					np.nansum(
						(covariateValues['covari'][(analysisStart - COVARIATE_FILE_INDEX_OFFSET):(analysisEnd - COVARIATE_FILE_INDEX_OFFSET)][highReadCountIdx] * covariates.selected) * COEFEXP_HIGHRC[rep, 1:],
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
					rcArr = np.rint(rcArr)
					coalescedSectionCount, startEntries, endEntries, valueEntries = coalesceSections(starts, rcArr, analysisEnd, binsize)
					writeCorrectedReads(correctedExprReadCountFiles[rep], chromoBytes, coalescedSectionCount, startEntries, endEntries, valueEntries)
				rep += 1

		reg += 1

	ctrlNames = [tempFile.name for tempFile in correctedCtrlReadCountFiles]
	exprNames = [tempFile.name for tempFile in correctedExprReadCountFiles]

	[tempFile.close() for tempFile in correctedCtrlReadCountFiles]
	[tempFile.close() for tempFile in correctedExprReadCountFiles]

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

cpdef writeCorrectedReads(outFile, chrom, sectionCount, starts, ends, values):
	cdef int outIdx = 0
	while outIdx < sectionCount:
		outFile.write(struct.pack(CORRECTED_RC_TEMP_FILE_STRUCT_FORMAT, chrom, starts[outIdx], ends[outIdx], values[outIdx]))
		outIdx += 1
