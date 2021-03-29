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
