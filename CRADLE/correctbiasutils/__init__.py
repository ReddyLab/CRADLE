from __future__ import annotations

import io
import linecache
import marshal
import math
import multiprocessing
import os
import os.path
import struct
import tempfile
import matplotlib # type: ignore
import matplotlib.pyplot as plt # type: ignore
import numpy as np
import pyBigWig # type: ignore
import statsmodels.api as sm # type: ignore

from shutil import copyfile
from typing import Iterator, List, Type

from CRADLE.correctbiasutils.cython import arraySplit, coalesceSections # type: ignore
from CRADLE.logging import timer

matplotlib.use('Agg')

TRAINING_BIN_SIZE = 1_000
SCATTERPLOT_SAMPLE_COUNT = 10_000
SONICATION_SHEAR_BIAS_OFFSET = 2

CORRECTED_RC_TEMP_FILE_STRUCT_FORMAT = "=LLf"

# 10_000_000 here is a somewhat arbitrary choice, but I've tried 1_000_000 and
# 100_000_000 on a significant run and not much changed.
# 100_000_000:  --  Completed Merging Temp Files .... : 27.71366087388333 min(s)
# 10_000_000:   --  Completed Merging Temp Files .... : 27.868539819183333 min(s)
# 1_000_000:    --  Completed Merging Temp Files .... : 28.09171124881666 min(s)
# I think 10_000_000 is a good memory tradeoff.
MERGE_FILES_BUFFER_SIZE = 10_000_000

# Used to adjust coordinates between 0 and 1-based systems.
START_INDEX_ADJUSTMENT = 1

class ChromoRegionMergeException(Exception):
	pass

class ChromoRegion:
	__slots__ = ["chromo", "start", "_end", "_length"]

	"""Describes a Chromosome region. Index is 0-based and half-closed"""
	def __init__(self, chromo: str, start: int, end: int) -> None:
		assert start <= end
		self.chromo = chromo
		self.start = start
		self._end = end
		self._length = end - start

	@property
	def end(self) -> int:
		return self._end

	@end.setter
	def end(self, value: int) -> None:
		assert value >= self.start

		self._end = value
		self._length = self._end - self.start

	def contiguousWith(self, o: ChromoRegion) -> bool:
		if self.chromo != o.chromo:
			return False

		if self.start < o.start:
			return self.end >= o.start
		else:
			return o.end >= self.start

	def __add__(self, o: ChromoRegion) -> ChromoRegion:
		if not self.contiguousWith(o):
			raise ChromoRegionMergeException(f"Regions are not contiguous: {self.chromo}:{self.start}-{self.end}, {o.chromo}:{o.start}-{o.end}")

		return ChromoRegion(self.chromo, min(self.start, o.start), max(self.end, o.end))

	def __sub__(self, o: ChromoRegion) -> List[ChromoRegion]:
		if not self.contiguousWith(o):
			return [ChromoRegion(self.chromo, self.start, self.end)]

		if o.start <= self.start and o.end >= self.end:
			return []
		elif o.start > self.start and o.end < self.end:
			return [ChromoRegion(self.chromo, self.start, o.start), ChromoRegion(self.chromo, o.end, self.end)]
		elif o.start <= self.start and o.end < self.end:
			return [ChromoRegion(self.chromo, o.end, self.end)]
		elif o.start > self.start and o.end >= self.end:
			return [ChromoRegion(self.chromo, self.start, o.start)]

		# Should not happen, the above conditions are exhaustive
		return []

	def __len__(self) -> int:
		return self._length

	def __eq__(self, o: object) -> bool:
		if not isinstance(o, ChromoRegion):
			return NotImplemented

		return (self.chromo == o.chromo
			and self.start == o.start
			and self.end == o.end
			and self._length == o._length)

	# __lt__ has been implemented specifically for sorting -- both the "list.sort()" method
	# and "sorted" function use __lt__ for comparing objects.
	def __lt__(self, o: object) -> bool:
		if not isinstance(o, ChromoRegion):
			return NotImplemented

		if self.chromo == o.chromo:
			return self.start < o.start

		selfChromo = self.chromo
		if selfChromo.startswith("chr"):
			selfChromo = selfChromo[3:]

		otherChromo = o.chromo
		if otherChromo.startswith("chr"):
			otherChromo = otherChromo[3:]

		if selfChromo.isdigit() and otherChromo.isdigit():
			return int(selfChromo) < int(otherChromo)

		if not selfChromo.isdigit() and otherChromo.isdigit():
			return False # sort letters higher than numbers

		if selfChromo.isdigit() and not otherChromo.isdigit():
			return True # sort letters higher than numbers

		return selfChromo < otherChromo # sort lexigraphically

	def __repr__(self) -> str:
		return f"({self.chromo}:{self.start}-{self.end})"


# Not quite a set in the mathematical sense.
class ChromoRegionSet:
	__slots__ = ["regions", "_chromoSet", "_chromos", "cumulativeRegionSize", "_chromoOrderDirty"]

	regions: list[ChromoRegion]
	_chromoSet: set[str]
	_chromos: list[str]
	cumulativeRegionSize: int
	_chromoOrderDirty: bool

	def __init__(self, regions: List[ChromoRegion]=None) -> None:
		self.regions = []
		self._chromoSet = set()
		self._chromos = []
		self.cumulativeRegionSize = 0
		self._chromoOrderDirty = False
		if regions is not None:
			self.regions = regions
			for region in self.regions:
				if region.chromo not in self._chromoSet:
					self._chromoSet.add(region.chromo)
					self._chromos.append(region.chromo)
				self.cumulativeRegionSize += len(region)

	def addRegion(self, region: ChromoRegion) -> None:
		self._chromoSet.add(region.chromo)
		self.regions.append(region)
		self.cumulativeRegionSize += len(region)
		self._chromoOrderDirty = True

	@property
	def chromos(self):
		if self._chromoOrderDirty:
			chromoSet = set()
			chromos = []
			for region in self.regions:
				if region.chromo not in chromoSet:
					chromoSet.add(region.chromo)
					chromos.append(region.chromo)
			self._chromos = chromos
		self._chromoOrderDirty = False
		return self._chromos

	def sortRegions(self) -> None:
		self.regions.sort()
		self._chromoOrderDirty = True

	def mergeRegions(self) -> None:
		if len(self.regions) == 1:
			return

		self.sortRegions()

		mergedRegions = []
		currentRegion = self.regions[0]

		for region in self.regions[1:]:
			if currentRegion.contiguousWith(region):
				currentRegion = currentRegion + region
			else:
				mergedRegions.append(currentRegion)
				currentRegion = region
		mergedRegions.append(currentRegion)

		mergedCumulativeLength = 0
		for region in mergedRegions:
			mergedCumulativeLength += len(region)

		self.regions = mergedRegions
		self.cumulativeRegionSize = mergedCumulativeLength

	def __add__(self, o: ChromoRegionSet) -> ChromoRegionSet:
		"""Simple conacatenation of sets. No region merging is done."""
		newRegionSet = ChromoRegionSet(self.regions + o.regions)
		newRegionSet.sortRegions()

		return newRegionSet

	def __sub__(self, o: ChromoRegionSet) -> ChromoRegionSet:
		regionWorkingSet = self.regions
		for removeRegion in o:
			tempRegions = []
			for region in regionWorkingSet:
				tempRegions.extend(region - removeRegion)
			regionWorkingSet = tempRegions

		return ChromoRegionSet(regionWorkingSet)

	def __len__(self) -> int:
		return len(self.regions)

	def __iter__(self) ->  Iterator[ChromoRegion]:
		for region in self.regions:
			yield region

	def __eq__(self, o: object) -> bool:
		if not isinstance(o, ChromoRegionSet):
			return NotImplemented

		if len(self.regions) != len(o.regions):
			return False

		if self.cumulativeRegionSize != o.cumulativeRegionSize:
			return False

		for selfRegion, oRegion in zip(sorted(self), sorted(o)):
			if selfRegion != oRegion:
				return False

		return True

	def __repr__(self) -> str:
		return f"[{', '.join([str(region) for region in self.regions])}]"

	@classmethod
	def loadBed(cls: Type[ChromoRegionSet], filename: str) -> ChromoRegionSet:
		regionSet = ChromoRegionSet()

		with open(filename) as regionFile:
			regionLines = regionFile.readlines()

		for line in regionLines:
			temp = line.split()
			regionSet.addRegion(ChromoRegion(temp[0], int(temp[1]), int(temp[2])))

		return regionSet


def process(poolSize, function, argumentLists, context="spawn"):
	pool = multiprocessing.get_context(context).Pool(poolSize)
	results = pool.starmap(function, argumentLists)
	pool.close()
	pool.join()

	return results


def getResultBWHeader(regions, ctrlBWName):
	with pyBigWig.open(ctrlBWName) as bwFile:
		resultBWHeader = []
		for chromo in regions.chromos:
			chromoSize = bwFile.chroms(chromo)
			resultBWHeader.append((chromo, chromoSize))

	return resultBWHeader


def outputBWFile(outputDir, filename):
	signalBWName = '.'.join(filename.rsplit('/', 1)[-1].split(".")[:-1])
	return os.path.join(outputDir, signalBWName + "_corrected.bw")


def outputCorrectedTmpFile(outputDir, chromo, chromoId, filename):
	signalBWName = '.'.join(filename.rsplit('/', 1)[-1].split(".")[:-1])
	return os.path.join(outputDir, f"{signalBWName}_{chromo}_{chromoId}_corrected.tmp")


def outputNormalizedTmpFile(outputDir, filename):
	normObBWName = '.'.join(filename.rsplit('/', 1)[-1].split(".")[:-1])
	return os.path.join(outputDir, normObBWName + "_normalized.tmp")


def mergeBWFiles(outputDir, header, fileChromoInfo, ctrlBWNames, experiBWNames):
	jobList = []
	for ctrlFile in ctrlBWNames:
		jobList.append((ctrlFile, header, fileChromoInfo, outputBWFile(outputDir, ctrlFile), outputDir))
	for experiFile in experiBWNames:
		jobList.append((experiFile, header, fileChromoInfo, outputBWFile(outputDir, experiFile), outputDir))

	return process(len(ctrlBWNames) + len(experiBWNames), mergeCorrectedFilesToBW, jobList)


def mergeCorrectedFilesToBW(replicateFile, bwHeader, fileChromoInfo, signalBWName, outputDir):
	signalBW = pyBigWig.open(signalBWName, "w")
	signalBW.addHeader(bwHeader)
	dataReadSize = struct.calcsize(CORRECTED_RC_TEMP_FILE_STRUCT_FORMAT) * MERGE_FILES_BUFFER_SIZE
	starts = np.zeros(MERGE_FILES_BUFFER_SIZE, dtype=np.int32)
	ends = np.zeros(MERGE_FILES_BUFFER_SIZE, dtype=np.int32)
	values = np.zeros(MERGE_FILES_BUFFER_SIZE, dtype=np.float32)

	for chromo, chromoId in fileChromoInfo:
		chromos = [chromo] * MERGE_FILES_BUFFER_SIZE
		tempFile = outputCorrectedTmpFile(outputDir, chromo, chromoId, replicateFile)
		with open(tempFile, "rb") as dataFile:
			data = dataFile.read(dataReadSize)
			while data != b'':
				total = 0
				for start, end, value in struct.iter_unpack(CORRECTED_RC_TEMP_FILE_STRUCT_FORMAT, data):
					starts[total] = start
					ends[total] = end
					values[total] = value
					total += 1
				signalBW.addEntries(chromos[:total], starts[:total], ends=ends[:total], values=values[:total])
				data = dataFile.read(dataReadSize)
		os.remove(tempFile)
	signalBW.close()

	return signalBWName


def divideGenome(regions, baseBinSize=1, genomeBinSize=50000):
	"""Splits regions larger than ~genomeBinSize into several regions genomeBinSize big."""

	# Adjust the genome bin size to be a multiple of base bin Size
	genomeBinSize -= genomeBinSize % baseBinSize

	# Return an list of tuples of values instead of a ChromoRegionSet because this will be used in Cython code.
	# And the less "python" interaction Cython has, the better. Python tuples are lighter weight
	# than regular objects.
	newRegions = []
	for region in regions:
		binStart = region.start
		while binStart < region.end:
			binEnd = binStart + genomeBinSize
			newRegions.append((region.chromo, binStart, min(binEnd, region.end)))
			binStart = binEnd

	return newRegions


def genNormalizedObBWs(outputDir, header, regions, ctrlBWNames, ctrlScaler, experiBWNames, experiScaler):
	# copy the first replicate
	observedBWName = ctrlBWNames[0]
	copyfile(observedBWName, outputNormalizedBWFile(outputDir, observedBWName))

	jobList = []
	for i, ctrlBWName in enumerate(ctrlBWNames[1:], start=1):
		jobList.append((header, float(ctrlScaler[i]), regions, ctrlBWName, outputNormalizedBWFile(outputDir, ctrlBWName)))
	for i, experiBWName in enumerate(experiBWNames):
		jobList.append((header, float(experiScaler[i]), regions, experiBWName, outputNormalizedBWFile(outputDir, experiBWName)))

	return process(len(ctrlBWNames) + len(experiBWNames) - 1, generateNormalizedObBWs, jobList)


def outputNormalizedBWFile(outputDir, filename):
	normObBWName = '.'.join(filename.rsplit('/', 1)[-1].split(".")[:-1])
	return os.path.join(outputDir, normObBWName + "_normalized.bw")


def generateNormalizedObBWs(bwHeader, scaler, regions, observedBWName, normObBWName):
	with pyBigWig.open(observedBWName) as obBW, pyBigWig.open(normObBWName, "w") as normObBW:
		_generateNormalizedObBWs(bwHeader, scaler, regions, obBW, normObBW)
	return normObBWName


def _generateNormalizedObBWs(bwHeader, scaler, regions, observedBW, normObBW):
	normObBW.addHeader(bwHeader)

	for region in regions:
		starts = np.arange(region.start, region.end)
		if pyBigWig.numpy == 1:
			values = observedBW.values(region.chromo, region.start, region.end, numpy=True)
		else:
			values = np.array(observedBW.values(region.chromo, region.start, region.end), dtype=np.float32)

		idx = np.where( (np.isnan(values) == False) & (values > 0))[0]
		starts = starts[idx]

		if len(starts) == 0:
			continue

		values = values[idx]
		values = values / scaler
		values = np.rint(values)

		coalescedSectionCount, startEntries, endEntries, valueEntries = coalesceSections(starts, values)
		normObBW.addEntries([region.chromo] * coalescedSectionCount, startEntries, ends=endEntries, values=valueEntries)


def getReadCounts(trainingSet, fileName):
	values = []

	with pyBigWig.open(fileName) as bwFile:
		for region in trainingSet:
			if pyBigWig.numpy == 1:
				temp = bwFile.values(region.chromo, region.start, region.end, numpy=True)
			else:
				temp = np.array(bwFile.values(region.chromo, region.start, region.end), dtype=np.float32)
			idx = np.where(np.isnan(temp) == True)
			temp[idx] = 0
			temp = temp.tolist()
			values.extend(temp)

	return values


def getScalerTasks(trainingSet, observedReadCounts, ctrlBWNames, experiBWNames):
	tasks = []
	ctrlBWCount = len(ctrlBWNames)
	sampleSetCount = ctrlBWCount + len(experiBWNames)
	for i in range(1, sampleSetCount):
		if i < ctrlBWCount:
			bwName = ctrlBWNames[i]
		else:
			bwName = experiBWNames[i - ctrlBWCount]
		tasks.append((trainingSet, observedReadCounts, bwName))

	return tasks


def getScalerForEachSample(trainingSet, observedReadCounts1Values, bwFileName):
	observedReadCounts2Values = getReadCounts(trainingSet, bwFileName)
	model = sm.OLS(observedReadCounts2Values, observedReadCounts1Values).fit()
	scaler = model.params[0]

	return scaler


@timer("Selecting Training Sets from trainingSetMetas", 1, "m")
def selectTrainingSetFromMeta(trainingSetMetas, rc99Percentile):
	trainSet1 = ChromoRegionSet()
	trainSet2 = ChromoRegionSet()

	### trainSet1
	for binIdx in range(5):
		if trainingSetMetas[binIdx] is None:
			continue

		regionNum = int(trainingSetMetas[binIdx][2])
		candiRegionNum = int(trainingSetMetas[binIdx][3])
		candiRegionFile = trainingSetMetas[binIdx][4]

		if candiRegionNum < regionNum:
			subfileStream = open(candiRegionFile)
			subfileLines = subfileStream.readlines()

			for line in subfileLines:
				temp = line.split()
				trainSet1.addRegion(ChromoRegion(temp[0], int(temp[1]), int(temp[2])))
		else:
			selectRegionIdx = np.random.choice(list(range(candiRegionNum)), regionNum, replace=False)

			for idx in selectRegionIdx:
				temp = linecache.getline(candiRegionFile, idx+1).split()
				trainSet1.addRegion(ChromoRegion(temp[0], int(temp[1]), int(temp[2])))
		os.remove(candiRegionFile)

	### trainSet2
	for binIdx in range(5, len(trainingSetMetas)):
		if trainingSetMetas[binIdx] is None:
			continue

		# downLimit = int(trainingSetMetas[binIdx][0])
		regionNum = int(trainingSetMetas[binIdx][2])
		candiRegionNum = int(trainingSetMetas[binIdx][3])
		candiRegionFile = trainingSetMetas[binIdx][4]

		# if downLimit == rc99Percentile:
		# 	subfileStream = open(candiRegionFile)
		# 	subfile = subfileStream.readlines()

		# 	i = len(subfile) - 1
		# 	while regionNum > 0 and i >= 0:
		# 		temp = subfile[i].split()
		# 		trainSet2.addRegion(ChromoRegion(temp[0], int(temp[1]), int(temp[2])))
		# 		i = i - 1
		# 		regionNum = regionNum - 1

		# else:

		if candiRegionNum < regionNum:
			subfileStream = open(candiRegionFile)
			subfileLines = subfileStream.readlines()

			for line in subfileLines:
				temp = line.split()
				trainSet2.addRegion(ChromoRegion(temp[0], int(temp[1]), int(temp[2])))
		else:
			selectRegionIdx = np.random.choice(list(range(candiRegionNum)), regionNum, replace=False)

			for idx in selectRegionIdx:
				temp = linecache.getline(candiRegionFile, idx+1).split()
				trainSet2.addRegion(ChromoRegion(temp[0], int(temp[1]), int(temp[2])))

		os.remove(candiRegionFile)

	return trainSet1, trainSet2


def regionMeans(bwFile, binCount, chromo, start, end):
	if pyBigWig.numpy == 1:
		values = bwFile.values(chromo, start, end, numpy=True)
	else:
		values = np.array(bwFile.values(chromo, start, end), dtype=np.float32)

	if binCount == 1:
		means = [np.mean(values)]
	else:
		binnedValues = arraySplit(values, binCount, fillValue=np.nan)
		means = [np.mean(x) for x in binnedValues]

	return means


@timer("Getting Candidate Training Sets", 1, "m")
def getCandidateTrainingSet(rcPercentile, regions, ctrlBWName, outputDir):
	trainRegionNum = math.pow(10, 6) / float(TRAINING_BIN_SIZE)

	meanRC = []
	totalBinNum = 0
	with pyBigWig.open(ctrlBWName) as ctrlBW:
		for region in regions:
			numBin = max(1, len(region) // TRAINING_BIN_SIZE)
			totalBinNum += numBin

			means = regionMeans(ctrlBW, numBin, region.chromo, region.start, region.end)

			meanRC.extend(means)

	if totalBinNum < trainRegionNum:
		trainRegionNum = totalBinNum

	meanRC = np.array(meanRC)
	meanRC = meanRC[np.where((np.isnan(meanRC) == False) & (meanRC > 0))]

	trainingRegionNum1 = int(np.round(trainRegionNum * 0.5 / 5))
	trainingRegionNum2 = int(np.round(trainRegionNum * 0.5 / 9))

	rc00pctl = int(np.percentile(meanRC, rcPercentile[0]))
	rc20pctl = int(np.percentile(meanRC, rcPercentile[1]))
	rc40pctl = int(np.percentile(meanRC, rcPercentile[2]))
	rc60pctl = int(np.percentile(meanRC, rcPercentile[3]))
	rc80pctl = int(np.percentile(meanRC, rcPercentile[4]))
	rc90pctl = int(np.percentile(meanRC, rcPercentile[5]))
	rc92pctl = int(np.percentile(meanRC, rcPercentile[6]))
	rc94pctl = int(np.percentile(meanRC, rcPercentile[7]))
	rc96pctl = int(np.percentile(meanRC, rcPercentile[8]))
	rc98pctl = int(np.percentile(meanRC, rcPercentile[9]))
	rc99pctl = int(np.percentile(meanRC, rcPercentile[10]))
	rc100pctl = int(np.percentile(meanRC, rcPercentile[11]))

	trainingSetMeta = [
		(rc00pctl, rc20pctl, trainingRegionNum1, regions, ctrlBWName, outputDir),
		(rc20pctl, rc40pctl, trainingRegionNum1, regions, ctrlBWName, outputDir),
		(rc40pctl, rc60pctl, trainingRegionNum1, regions, ctrlBWName, outputDir),
		(rc60pctl, rc80pctl, trainingRegionNum1, regions, ctrlBWName, outputDir),
		(rc80pctl, rc90pctl, trainingRegionNum1, regions, ctrlBWName, outputDir),

		(rc90pctl, rc92pctl, trainingRegionNum2, regions, ctrlBWName, outputDir),
		(rc92pctl, rc94pctl, trainingRegionNum2, regions, ctrlBWName, outputDir),
		(rc94pctl, rc96pctl, trainingRegionNum2, regions, ctrlBWName, outputDir),
		(rc96pctl, rc98pctl, trainingRegionNum2, regions, ctrlBWName, outputDir),
		(rc98pctl, rc99pctl, trainingRegionNum2, regions, ctrlBWName, outputDir),

		(rc99pctl, rc100pctl, 3*trainingRegionNum2, regions, ctrlBWName, outputDir)
	]

	return trainingSetMeta, rc90pctl, rc99pctl


def fillTrainingSetMeta(downLimit, upLimit, trainingRegionNum, regions, ctrlBWName, outputDir):
	ctrlBW = pyBigWig.open(ctrlBWName)

	resultLine = [downLimit, upLimit, trainingRegionNum]
	numOfCandiRegion = 0

	resultFile = tempfile.NamedTemporaryFile(mode="w+t", suffix=".txt", dir=outputDir, delete=False)

	fileBuffer = io.StringIO()
	for region in regions:
		numBin = len(region) // TRAINING_BIN_SIZE
		if numBin == 0:
			numBin = 1

			meanValue = regionMeans(ctrlBW, numBin, region.chromo, region.start, region.end)[0]

			if np.isnan(meanValue) or meanValue == 0:
				continue

			if (meanValue >= downLimit) and (meanValue < upLimit):
				fileBuffer.write(f"{region.chromo}\t{region.start}\t{region.end}\n")
				numOfCandiRegion += 1
		else:
			regionEnd = numBin * TRAINING_BIN_SIZE + region.start

			meanValues = np.array(regionMeans(ctrlBW, numBin, region.chromo, region.start, regionEnd))

			pos = np.arange(0, numBin) * TRAINING_BIN_SIZE + region.start

			idx = np.where((meanValues != None) & (meanValues > 0))
			meanValues = meanValues[idx]
			pos = pos[idx]

			idx = np.where((meanValues >= downLimit) & (meanValues < upLimit))

			if len(idx[0]) == 0:
				continue

			starts = pos[idx]

			numOfCandiRegion += len(starts)
			for start in starts:
				fileBuffer.write(f"{region.chromo}\t{start}\t{start + TRAINING_BIN_SIZE}\n")

	resultFile.write(fileBuffer.getvalue())
	ctrlBW.close()
	resultFile.close()
	fileBuffer.close()

	if numOfCandiRegion == 0:
		os.remove(resultFile.name)
		return None

	resultLine.extend([numOfCandiRegion, resultFile.name])
	return resultLine


def alignCoordinatesToCovariateFileBoundaries(chromoEnds, trainingSet, fragLen):
	newTrainingSet = ChromoRegionSet()

	for trainingRegion in trainingSet:
		chromoEnd = chromoEnds[trainingRegion.chromo]
		analysisStart = trainingRegion.start
		analysisEnd = trainingRegion.end

		# Define a region of fragments of length fragLen
		fragRegionStart = analysisStart - fragLen + START_INDEX_ADJUSTMENT
		fragRegionEnd = analysisEnd + fragLen - START_INDEX_ADJUSTMENT

		# Define a region that includes base pairs used to model shearing/sonication bias
		shearStart = fragRegionStart - SONICATION_SHEAR_BIAS_OFFSET
		shearEnd = fragRegionEnd + SONICATION_SHEAR_BIAS_OFFSET

		# Make sure the analysisStart and analysisEnd fall within the boundaries of the region
		# covariates have been precomputed for
		if shearStart < 1:
			fragRegionStart = SONICATION_SHEAR_BIAS_OFFSET + START_INDEX_ADJUSTMENT
			analysisStart = max(analysisStart, fragRegionStart)

		if shearEnd > chromoEnd:
			fragRegionEnd = chromoEnd - SONICATION_SHEAR_BIAS_OFFSET
			analysisEnd = min(analysisEnd, fragRegionEnd)

		newTrainingSet.addRegion(ChromoRegion(trainingRegion.chromo, analysisStart, analysisEnd))

	return newTrainingSet


def getScatterplotSampleIndices(populationSize):
	if populationSize <= SCATTERPLOT_SAMPLE_COUNT:
		return np.arange(0, populationSize)
	else:
		return np.random.choice(np.arange(0, populationSize), SCATTERPLOT_SAMPLE_COUNT, replace=False)


def figureFileName(outputDir, bwFilename):
	bwName = '.'.join(bwFilename.rsplit('/', 1)[-1].split(".")[:-1])
	return os.path.join(outputDir, f"fit_{bwName}.png")


def plot(regRCs, regRCFittedValues, highRCs, highRCFittedValues, figName):
	corr = np.corrcoef(regRCFittedValues, regRCs)[0, 1]
	corr = np.round(corr, 2)
	maxi1 = np.nanmax(regRCFittedValues)
	maxi2 = np.nanmax(regRCs)
	maxiRegRC = max(maxi1, maxi2)
	plt.plot(regRCs, regRCFittedValues, color='g', marker='s', alpha=0.01)

	corr = np.corrcoef(highRCFittedValues, highRCs)[0, 1]
	corr = np.round(corr, 2)
	maxi1 = np.nanmax(highRCFittedValues)
	maxi2 = np.nanmax(highRCs)
	maxiHighRC = max(maxi1, maxi2)
	plt.plot(highRCs, highRCFittedValues, color='g', marker='s', alpha=0.01)

	maxi = max(maxiRegRC, maxiHighRC)

	plt.text((maxi-25), 10, corr, ha='center', va='center')
	plt.xlabel("observed")
	plt.ylabel("predicted")
	plt.xlim(0, maxi)
	plt.ylim(0, maxi)
	plt.plot([0, maxi], [0, maxi], 'k-')
	plt.gca().set_aspect('equal', adjustable='box')
	plt.savefig(figName)
	plt.close()
	plt.clf()
