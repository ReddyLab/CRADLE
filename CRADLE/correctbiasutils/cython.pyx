import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pyBigWig

matplotlib.use('Agg')

cpdef array_split(values, numBins, fillValue=np.nan):
	"""Splits a numpy array of values into numBins "bins". If len(values) is not evenly
	divisible by numBins the last bin gets the extra values.

	Example:
		array_split(np.arange(0, 16), 3) = [array(0, 1, 2, 3, 4), array(5, 6, 7, 8, 9), array(10, 11, 12, 13, 14, 15)]
	"""

	cdef float [:] valuesView
	cdef int binCount
	cdef int smallBinSize
	cdef int valueCount
	cdef int start
	cdef int end
	cdef int smallBinCount
	cdef int i

	binCount = numBins
	if binCount < 1:
		return values

	valuesView = values
	valueCount = len(valuesView)
	smallBinSize = max(1, valueCount // binCount)
	start = 0
	end = smallBinSize
	smallBinCount = binCount - 1
	bins = [fillValue] * binCount

	i = 0
	while i < smallBinCount and end < valueCount:
		bins[i] = valuesView[start:end]
		start = end
		end += smallBinSize
		i += 1

	if start < valueCount:
		bins[i] = valuesView[start:]

	return bins

cpdef generateNormalizedObBWs(bwHeader, scaler, regions, observedBWName, normObBWName):
	cdef int currStart
	cdef int currEnd
	cdef float currRC
	cdef int nextStart
	cdef int nextEnd
	cdef float nextRC
	cdef int numIdx
	cdef int i
	cdef int coalescedSections
	cdef long [:] startsView
	cdef long [:] valuesView

	normObBW = pyBigWig.open(normObBWName, "w")
	normObBW.addHeader(bwHeader)

	obBW = pyBigWig.open(observedBWName)
	for region in regions:
		chromo = region[0]
		start = int(region[1])
		end = int(region[2])

		starts = np.arange(start, end)
		if pyBigWig.numpy == 1:
			values = obBW.values(chromo, start, end, numpy=True)
		else:
			values = np.array(obBW.values(chromo, start, end))

		idx = np.where( (np.isnan(values) == False) & (values > 0))[0]
		starts = starts[idx]

		if len(starts) == 0:
			continue

		values = values[idx]
		values = values / scaler

		## merge positions with the same values
		values = values.astype(int)
		numIdx = len(values)

		startsView = starts
		valuesView = values

		currStart = startsView[0]
		currRC = valuesView[0]
		currEnd = currStart + 1

		startEntries = []
		endEntries = []
		valueEntries = []

		coalescedSections = 1
		i = 1
		while i < numIdx:
			nextStart = startsView[i]
			nextRC = valuesView[i]

			if nextRC == currRC and nextStart == currEnd:
				currEnd += 1
			else:
				### End a current line
				startEntries.append(currStart)
				endEntries.append(currEnd)
				valueEntries.append(float(currRC))

				### Start a new line
				currStart = nextStart
				currEnd = nextStart + 1
				currRC = nextRC
				coalescedSections += 1

			i += 1

		startEntries.append(currStart)
		endEntries.append(currEnd)
		valueEntries.append(float(currRC))

		normObBW.addEntries([chromo] * coalescedSections, startEntries, ends=endEntries, values=valueEntries)
	normObBW.close()
	obBW.close()

	return normObBWName

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

cpdef writeBedFile(subfile, tempStarts, tempSignalvals, analysisEnd, binsize):
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