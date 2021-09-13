# cython: language_level=3

import numpy as np

cpdef arraySplit(values, numBins, fillValue=np.nan):
	"""Splits a numpy array of values into numBins "bins". If len(values) is not evenly
	divisible by numBins the last bin gets the extra values.

	Example:
		arraySplit(np.arange(0, 16), 3) = [array(0, 1, 2, 3, 4), array(5, 6, 7, 8, 9), array(10, 11, 12, 13, 14, 15)]
	"""

	cdef int binCount = numBins
	cdef int valueCount = len(values)
	cdef int extraValues
	cdef int smallBinSize
	cdef int largeBinSize
	cdef int start
	cdef int end
	cdef int smallBinEndIndex
	cdef int largeBinEndIndex
	cdef int i

	if binCount < 1:
		return [values]
	elif binCount > valueCount:
		binCount = valueCount

	smallBinSize = max(1, valueCount // binCount)
	largeBinSize = smallBinSize + 1
	extraValues = valueCount - (binCount * smallBinSize)
	smallBinEndIndex = binCount - extraValues
	largeBinEndIndex = smallBinEndIndex + extraValues

	bins = [fillValue] * binCount

	start = 0
	end = smallBinSize
	i = 0
	while i < smallBinEndIndex:
		bins[i] = values[start:end]
		start = end
		end += smallBinSize
		i += 1

	end = start + largeBinSize
	while i < largeBinEndIndex:
		bins[i] = values[start:end]
		start = end
		end += largeBinSize
		i += 1

	return bins

cpdef writeBedFile(subfile, tempStarts, tempSignalvals, analysisEnd, binsize):
	tempSignalvals = tempSignalvals.astype(int)
	numIdx = len(tempSignalvals)

	idx = 0
	prevStart = tempStarts[idx]
	prevReadCount = tempSignalvals[idx]
	line = [prevStart, (prevStart + binsize), prevReadCount]
	if numIdx == 1:
		line[1] = min(line[1], analysisEnd)
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
			line[1] = min(line[1], analysisEnd)
			subfile.write('\t'.join([str(x) for x in line]) + "\n")
			subfile.close()
			break

cpdef coalesceSections(starts, values, analysisEnd=None, stepSize=1):
	""" Coalesce adjacent sections with the same values into a single sections.
	Note: This also coerces all the values to integers.
	"""
	cdef long [:] startsView
	cdef long [:] valuesView
	cdef long cStepSize = stepSize
	cdef int i
	cdef int numIdx
	cdef long currStart
	cdef long nextStart
	cdef long currEnd
	cdef long currValue
	cdef long nextValue
	cdef int coalescedSectionCount

	values = values.astype(int)
	numIdx = values.size

	startEntries = []
	endEntries = []
	valueEntries = []

	if numIdx == 0:
		return 0, startEntries, endEntries, valueEntries

	startsView = starts
	valuesView = values

	currStart = startsView[0]
	currValue = valuesView[0]
	currEnd = currStart + cStepSize

	coalescedSectionCount = 1
	i = 1
	while i < numIdx:
		nextStart = startsView[i]
		nextValue = valuesView[i]

		if nextValue == currValue and nextStart == currEnd:
			currEnd += cStepSize
		else:
			### End a current line
			startEntries.append(currStart)
			endEntries.append(currEnd)
			valueEntries.append(float(currValue))

			### Start a new line
			currStart = nextStart
			currEnd = nextStart + cStepSize
			currValue = nextValue
			coalescedSectionCount += 1

		i += 1

	if analysisEnd is not None:
		currEnd = min(currEnd, analysisEnd)

	startEntries.append(currStart)
	endEntries.append(currEnd)
	valueEntries.append(float(currValue))

	return coalescedSectionCount, startEntries, endEntries, valueEntries
