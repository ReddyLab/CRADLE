import numpy as np

class BigWig:
	'''A mock for interacting with bigwig files. Don't use large regions/coordinates
	as this will create an array large enough to contain the entire thing.

	E.g., `b.addEntries(['chr17'], [8_000_000], [8_000_001], [1.0])` will create
	an array 8,000,001 entries long, mostly np.nan.'''

	def __init__(self, data):
		self.data = data
		self.header = None

	def addHeader(self, header):
		self.header = header

	def values(self, chrom, start, end, numpy=False):
		allData = self.data.get(chrom, None)
		data = [np.nan] * (end - start)
		if allData is not None and start < len(allData):
			stop = min(end, len(allData))
			data[0:stop - start] = allData[start:stop]

		if numpy:
			return np.array(data)
		else:
			return data

	def addEntries(self, chroms, starts, ends=None, values=None):
		'''This is not an exact replica of pyBigWig's behavior. That
		version of addEntries has some more arguments and accepts inputs
		in several formats not supported here.

		For more information see https://github.com/deeptools/pyBigWig#adding-entries-to-a-bigwig-file
		and https://github.com/deeptools/pyBigWig/blob/master/pyBigWig.c#L1674'''

		if values is None:
			return
		if ends is None:
			return
		if len(chroms) != len(starts):
			return
		if len(chroms) != len(values):
			return
		if len(chroms) != len(ends):
			return

		for i, chrom in enumerate(chroms):
			currentChrom = self.data.get(chrom, None)
			start = starts[i]
			end = ends[i]
			readCount = values[i]

			if currentChrom is None:
				currentChrom = [np.nan] * (end - 1)
			else:
				if len(currentChrom) < end:
					currentChrom += [np.nan] * (end - len(currentChrom))

			currentChrom[start:end] = [readCount] * (end - start)
			self.data[chrom] = currentChrom

	def __eq__(self, other):
		return self.header == other.header and self.data == other.data

	def __repr__(self):
		return f"{self.header}: {self.data}"
