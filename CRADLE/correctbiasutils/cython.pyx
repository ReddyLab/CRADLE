import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use('Agg')

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