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