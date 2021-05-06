import io
import linecache
import marshal
import math
import multiprocessing
import os
import tempfile
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pyBigWig
import statsmodels.api as sm

from shutil import copyfile

from CRADLE.correctbiasutils.cython import arraySplit, generateNormalizedObBWs

matplotlib.use('Agg')

TRAINING_BIN_SIZE = 1000

class TrainingRegion:
	def __init__(self, chromo, analysisStart, analysisEnd):
		self.chromo = chromo
		self.analysisStart = analysisStart
		self.analysisEnd = analysisEnd

class TrainingSet:
	def __init__(self, trainingRegions, xRowCount):
		self.trainingRegions = trainingRegions
		self.xRowCount = xRowCount

	def __iter__(self):
		def regionGenerator():
			for region in self.trainingRegions:
				yield region

		return regionGenerator()

def process(poolSize, function, argumentLists):
	pool = multiprocessing.Pool(poolSize)
	results = pool.starmap_async(function, argumentLists).get()
	pool.close()
	pool.join()

	return results

def getResultBWHeader(region, ctrlBWNames):
	chromoInData = np.array(region)[:,0]

	chromoInDataUnique = set()
	bwFile = pyBigWig.open(ctrlBWNames[0])

	resultBWHeader = []
	for chromo in chromoInData:
		if chromo in chromoInDataUnique:
			continue
		chromoInDataUnique.add(chromo)
		chromoSize = bwFile.chroms(chromo)
		resultBWHeader.append( (chromo, chromoSize) )

	bwFile.close()

	return resultBWHeader

def marshalFile(outputDir, data):
	with tempfile.NamedTemporaryFile(mode="wb", suffix=".msl", dir=outputDir, delete=False) as outputFile:
		name = outputFile.name
		marshal.dump(data, outputFile)

	return name

def rotateBWFileArrays(tempFiles, ctrlBWNames, experiBWNames):
	# The tempFiles list isn't in a useful shape. It's a list of pairs of lists of files
	# Example:
	# [
	#   # Results of Job 0
	#	[['ctrlFile0Temp0.msl', 'ctrlFile1Temp0.msl'], ['experiFile0Temp0.msl', 'experiFile1Temp0.msl'], ],
	#   # Results of Job 1
	#	[['ctrlFile0Temp1.msl', 'ctrlFile1Temp1.msl'], ['experiFile0Temp1.msl', 'experiFile1Temp1.msl'], ],
	#   # Results of Job 2
	#	[['ctrlFile0Temp2.msl', 'ctrlFile1Temp2.msl'], ['experiFile0Temp2.msl', 'experiFile1Temp2.msl'], ],
	# ]
	#
	# The following code rearranges the file names so they are in this shape:
	# [
	#	[
	# 		['ctrlFile0Temp0.msl', 'ctrlFile0Temp1.msl', 'ctrlFile0Temp2.msl'],
	# 		['ctrlFile1Temp0.msl', 'ctrlFile1Temp1.msl', 'ctrlFile1Temp2.msl']
	# 	],
	#	[
	# 		['experiFile0Temp0.msl', 'experiFile0Temp1.msl', 'experiFile0Temp2.msl'],
	# 		['experiFile1Temp0.msl', 'experiFile1Temp1.msl', 'experiFile1Temp2.msl']
	# 	],
	# ]
	#
	# Which matches what we want to do with the files better -- combine all temp files for a particular file together
	# into a single BigWig file

	ctrlFiles = [[] for _ in range(len(ctrlBWNames))]
	experiFiles = [[] for _ in range(len(experiBWNames))]
	for jobFiles in tempFiles:
		jobCtrl = jobFiles[0]
		jobExperi = jobFiles[1]
		for i, ctrlFile in enumerate(jobCtrl):
			ctrlFiles[i].append(ctrlFile)
		for i, experiFile in enumerate(jobExperi):
			experiFiles[i].append(experiFile)

	return ctrlFiles, experiFiles

def mergeBWFiles(outputDir, header, tempFiles, ctrlBWNames, experiBWNames):
	ctrlFiles, experiFiles = rotateBWFileArrays(tempFiles, ctrlBWNames, experiBWNames)

	jobList = []
	for i, ctrlFile in enumerate(ctrlBWNames):
		jobList.append([ctrlFiles[i], header, outputBWFile(outputDir, ctrlFile)])
	for i, experiFile in enumerate(experiBWNames):
		jobList.append([experiFiles[i], header, outputBWFile(outputDir, experiFile)])

	return process(len(ctrlBWNames) + len(experiBWNames), mergeCorrectedFilesToBW, jobList)

def outputBWFile(outputDir, filename):
	signalBWName = '.'.join(filename.rsplit('/', 1)[-1].split(".")[:-1])
	return outputDir + "/" + signalBWName + "_corrected.bw"

def mergeCorrectedFilesToBW(tempFiles, bwHeader, signalBWName):
	signalBW = pyBigWig.open(signalBWName, "w")
	signalBW.addHeader(bwHeader)

	for tempFile in tempFiles:
		with open(tempFile, 'rb') as dataFile:
			correctedReadCounts = marshal.load(dataFile)

		for counts in correctedReadCounts:
			chromo, readCounts = counts
			if readCounts is None:
				continue

			sectionCount, starts, ends, values = readCounts
			signalBW.addEntries([chromo] * sectionCount, starts, ends=ends, values=values)

		os.remove(tempFile)

	signalBW.close()

	return signalBWName

def divideGenome(regions, baseBinSize=1, genomeBinSize=50000):

	# Adjust the genome bin size to be a multiple of base bin Size
	genomeBinSize -= genomeBinSize % baseBinSize

	tasks = []
	for region in regions:
		regionChromo = region[0]
		regionStart = int(region[1])
		regionEnd = int(region[2])

		binStart = regionStart
		while binStart < regionEnd:
			binEnd = binStart + genomeBinSize
			tasks.append((regionChromo, binStart, min(binEnd, regionEnd)))
			binStart = binEnd

	return tasks

def genNormalizedObBWs(outputDir, header, regions, ctrlBWNames, ctrlScaler, experiBWNames, experiScaler):
	# copy the first replicate
	observedBWName = ctrlBWNames[0]
	copyfile(observedBWName, outputNormalizedBWFile(outputDir, observedBWName))

	jobList = []
	for i, ctrlBWName in enumerate(ctrlBWNames[1:], start=1):
		jobList.append([header, float(ctrlScaler[i]), regions, ctrlBWName, outputNormalizedBWFile(outputDir, ctrlBWName)])
	for i, experiBWName in enumerate(experiBWNames):
		jobList.append([header, float(experiScaler[i]), regions, experiBWName, outputNormalizedBWFile(outputDir, experiBWName)])


	return process(len(ctrlBWNames) + len(experiBWNames) - 1, generateNormalizedObBWs, jobList)

def outputNormalizedBWFile(outputDir, filename):
	normObBWName = '.'.join(filename.rsplit('/', 1)[-1].split(".")[:-1])
	return outputDir + "/" + normObBWName + "_normalized.bw"

def getScalerTasks(trainingSets, ctrlBWNames, experiBWNames, sampleCount):
	###### OBTAIN READ COUNTS OF THE FIRST REPLICATE OF CTRLBW.

	ob1 = pyBigWig.open(ctrlBWNames[0])

	ob1Values = []
	for trainingSet in trainingSets:
		chromo = trainingSet[0]
		start = int(trainingSet[1])
		end = int(trainingSet[2])

		temp = np.array(ob1.values(chromo, start, end))
		idx = np.where(np.isnan(temp))
		temp[idx] = 0
		temp = temp.tolist()
		ob1Values.extend(temp)

	tasks = []
	for i in range(1, sampleCount):
		if i < len(ctrlBWNames):
			bwName = ctrlBWNames[i]
		else:
			bwName = experiBWNames[i - len(ctrlBWNames)]
		tasks.append([trainingSets, ob1Values, bwName])

	return tasks

def getScalerForEachSample(trainingSets, ob1Values, bwName):
	ob2 = pyBigWig.open(bwName)
	ob2Values = []
	for trainingSet in trainingSets:
		chromo = trainingSet[0]
		start = int(trainingSet[1])
		end = int(trainingSet[2])

		temp = np.array(ob2.values(chromo, start, end))
		idx = np.where(np.isnan(temp) == True)
		temp[idx] = 0
		temp = temp.tolist()
		ob2Values.extend(temp)
	ob2.close()

	model = sm.OLS(ob2Values, ob1Values).fit()
	scaler = model.params[0]

	return scaler

def selectTrainingSetFromMeta(trainingSetMetas, rc99Percentile):
	trainSet1 = []
	trainSet2 = []

	### trainSet1
	for binIdx in range(5):
		if trainingSetMetas[binIdx] is None:
			continue

		regionNum = int(trainingSetMetas[binIdx][2])
		candiRegionFile = trainingSetMetas[binIdx][3]
		candiRegionNum = int(trainingSetMetas[binIdx][4])

		if candiRegionNum < regionNum:
			subfileStream = open(candiRegionFile)
			subfileLines = subfileStream.readlines()

			for line in subfileLines:
				temp = line.split()
				trainSet1.append([temp[0], int(temp[1]), int(temp[2])])
		else:
			selectRegionIdx = np.random.choice(list(range(candiRegionNum)), regionNum, replace=False)

			for idx in selectRegionIdx:
				temp = linecache.getline(candiRegionFile, idx+1).split()
				trainSet1.append([temp[0], int(temp[1]), int(temp[2])])
		os.remove(candiRegionFile)

	### trainSet2
	for binIdx in range(5, len(trainingSetMetas)):
		if trainingSetMetas[binIdx] is None:
			continue

		# downLimit = int(trainingSetMetas[binIdx][0])
		regionNum = int(trainingSetMetas[binIdx][2])
		candiRegionFile = trainingSetMetas[binIdx][3]
		candiRegionNum = int(trainingSetMetas[binIdx][4])

		# if downLimit == rc99Percentile:
		# 	subfileStream = open(candiRegionFile)
		# 	subfile = subfileStream.readlines()

		# 	i = len(subfile) - 1
		# 	while regionNum > 0 and i >= 0:
		# 		temp = subfile[i].split()
		# 		trainSet2.append([ temp[0], int(temp[1]), int(temp[2])])
		# 		i = i - 1
		# 		regionNum = regionNum - 1

		# else:

		if candiRegionNum < regionNum:
			subfileStream = open(candiRegionFile)
			subfileLines = subfileStream.readlines()

			for line in subfileLines:
				temp = line.split()
				trainSet2.append([ temp[0], int(temp[1]), int(temp[2])])
		else:
			selectRegionIdx = np.random.choice(list(range(candiRegionNum)), regionNum, replace=False)

			for idx in selectRegionIdx:
				temp = linecache.getline(candiRegionFile, idx+1).split()
				trainSet2.append([ temp[0], int(temp[1]), int(temp[2])])

		os.remove(candiRegionFile)

	return trainSet1, trainSet2

def regionMeans(bwFile, binCount, chromo, start, end):
	if pyBigWig.numpy == 1:
		values = bwFile.values(chromo, start, end, numpy=True)
	else:
		values = np.array(bwFile.values(chromo, start, end))

	if binCount == 1:
		means = [np.mean(values)]
	else:
		binnedValues = arraySplit(values, binCount, fillValue=np.nan)
		means = [np.mean(x) for x in binnedValues]

	return means

def getCandidateTrainingSet(rcPercentile, regions, ctrlBWName, outputDir):
	trainRegionNum = math.pow(10, 6) / float(TRAINING_BIN_SIZE)

	meanRC = []
	totalBinNum = 0
	with pyBigWig.open(ctrlBWName) as ctrlBW:
		for region in regions:
			regionChromo = region[0]
			regionStart = int(region[1])
			regionEnd = int(region[2])

			numBin = max(1, (regionEnd - regionStart) // TRAINING_BIN_SIZE)
			totalBinNum += numBin

			means = regionMeans(ctrlBW, numBin, regionChromo, regionStart, regionEnd)

			meanRC.extend(means)

	if totalBinNum < trainRegionNum:
		trainRegionNum = totalBinNum

	meanRC = np.array(meanRC)
	meanRC = meanRC[np.where((np.isnan(meanRC) == False) & (meanRC > 0))]

	trainingRegionNum1 = int(np.round(trainRegionNum * 0.5 / 5))
	trainingRegionNum2 = int(np.round(trainRegionNum * 0.5 / 9))
	trainingSetMeta = []
	rc90Percentile = None
	rc99Percentile = None

	for i in range(5):
		rc1 = int(np.percentile(meanRC, int(rcPercentile[i])))
		rc2 = int(np.percentile(meanRC, int(rcPercentile[i+1])))
		temp = [rc1, rc2, trainingRegionNum1, regions, ctrlBWName, outputDir, rc90Percentile, rc99Percentile]
		trainingSetMeta.append(temp)  ## RC criteria1(down), RC criteria2(up), # of bases, candidate regions

	for i in range(5, 11):
		rc1 = int(np.percentile(meanRC, int(rcPercentile[i])))
		rc2 = int(np.percentile(meanRC, int(rcPercentile[i+1])))
		if i == 10:
			temp = [rc1, rc2, 3*trainingRegionNum2, regions, ctrlBWName, outputDir, rc90Percentile, rc99Percentile]
			rc99Percentile = rc1
		else:
			temp = [rc1, rc2, trainingRegionNum2, regions, ctrlBWName, outputDir, rc90Percentile, rc99Percentile] # RC criteria1(down), RC criteria2(up), # of bases, candidate regions
		if i == 5:
			rc90Percentile = rc1

		trainingSetMeta.append(temp)

	return trainingSetMeta, rc90Percentile, rc99Percentile

def fillTrainingSetMeta(downLimit, upLimit, trainingRegionNum, regions, ctrlBWName, outputDir, rc90Percentile = None, rc99Percentile = None):
	ctrlBW = pyBigWig.open(ctrlBWName)

	resultLine = [downLimit, upLimit, trainingRegionNum]
	numOfCandiRegion = 0

	resultFile = tempfile.NamedTemporaryFile(mode="w+t", suffix=".txt", dir=outputDir, delete=False)
	'''
	if downLimit == rc99Percentile:
		result = []

		for region in regions:
			regionChromo = region[0]
			regionStart = int(region[1])
			regionEnd = int(region[2])

			numBin = int( (regionEnd - regionStart) / TRAINING_BIN_SIZE )
			if numBin == 0:
				numBin = 1
				meanValue = np.array(ctrlBW.stats(regionChromo, regionStart, regionEnd, nBins=numBin, type="mean"))[0]

				if meanValue is None:
					continue

				if (meanValue >= downLimit) and (meanValue < upLimit):
					result.append([regionChromo, regionStart, regionEnd, meanValue])

			else:
				regionEnd = numBin * TRAINING_BIN_SIZE + regionStart
				meanValues = np.array(ctrlBW.stats(regionChromo, regionStart, regionEnd, nBins=numBin, type="mean"))
				pos = np.array(list(range(0, numBin))) * TRAINING_BIN_SIZE + regionStart

				idx = np.where(meanValues != None)
				meanValues = meanValues[idx]
				pos = pos[idx]

				idx = np.where((meanValues >= downLimit) & (meanValues < upLimit))
				start = pos[idx]
				end = start + TRAINING_BIN_SIZE
				meanValues = meanValues[idx]
				chromoArray = [regionChromo] * len(start)
				result.extend(np.column_stack((chromoArray, start, end, meanValues)).tolist())

		if len(result) == 0:
			return None

		result = np.array(result)
		result = result[result[:,3].astype(float).astype(int).argsort()][:,0:3].tolist()

		numOfCandiRegion = len(result)
		for i in range(len(result)):
			resultFile.write('\t'.join([str(x) for x in result[i]]) + "\n")
	'''
	fileBuffer = io.StringIO()
	for region in regions:
		regionChromo = region[0]
		regionStart = int(region[1])
		regionEnd = int(region[2])

		numBin = (regionEnd - regionStart) // TRAINING_BIN_SIZE
		if numBin == 0:
			numBin = 1

			meanValue = regionMeans(ctrlBW, numBin, regionChromo, regionStart, regionEnd)[0]

			if np.isnan(meanValue) or meanValue == 0:
				continue

			if (meanValue >= downLimit) and (meanValue < upLimit):
				fileBuffer.write(f"{regionChromo}\t{regionStart}\t{regionEnd}\n")
				numOfCandiRegion += 1
		else:
			regionEnd = numBin * TRAINING_BIN_SIZE + regionStart

			meanValues = np.array(regionMeans(ctrlBW, numBin, regionChromo, regionStart, regionEnd))

			pos = np.arange(0, numBin) * TRAINING_BIN_SIZE + regionStart

			idx = np.where((meanValues != None) & (meanValues > 0))
			meanValues = meanValues[idx]
			pos = pos[idx]

			idx = np.where((meanValues >= downLimit) & (meanValues < upLimit))

			if len(idx[0]) == 0:
				continue

			starts = pos[idx]

			numOfCandiRegion += len(starts)
			for start in starts:
				fileBuffer.write(f"{regionChromo}\t{start}\t{start + TRAINING_BIN_SIZE}\n")

	resultFile.write(fileBuffer.getvalue())
	ctrlBW.close()
	resultFile.close()
	fileBuffer.close()

	if numOfCandiRegion != 0:
		resultLine.extend([ resultFile.name, numOfCandiRegion ])
		return resultLine

	os.remove(resultFile.name)
	return None

def alignCoordinatesToHDF(faFile, oldTrainingSet, fragLen):
	trainingSet = []
	xRowCount = 0

	for trainingRegion in oldTrainingSet:
		chromo = trainingRegion[0]
		analysisStart = int(trainingRegion[1])
		analysisEnd = int(trainingRegion[2])
		chromoEnd = faFile.chroms(chromo)

		fragStart = analysisStart - fragLen + 1
		fragEnd = analysisEnd + fragLen - 1
		shearStart = fragStart - 2
		shearEnd = fragEnd + 2

		if shearStart < 1:
			shearStart = 1
			fragStart = 3
			analysisStart = max(analysisStart, fragStart)

		if shearEnd > chromoEnd:
			shearEnd = chromoEnd
			fragEnd = shearEnd - 2
			analysisEnd = min(analysisEnd, fragEnd)  # not included

		xRowCount += (analysisEnd - analysisStart)
		trainingSet.append(TrainingRegion(chromo, analysisStart, analysisEnd))

	return TrainingSet(trainingRegions=trainingSet, xRowCount=xRowCount)

def getScatterplotSamples(trainingSet):
	if trainingSet.xRowCount <= 10_000:
		return np.arange(0, trainingSet.xRowCount)
	else:
		return np.random.choice(np.arange(0, trainingSet.xRowCount), 10_000, replace=False)

def figureFileName(outputDir, bwFileName):
	bwName = '.'.join(bwFileName.rsplit('/', 1)[-1].split(".")[:-1])
	return os.path.join(outputDir, f"fit_{bwName}.png")

def plot(regRCs, regRCFittedValues, highRCs, highRCFittedValues, figName):
	fig, ax = plt.subplots()

	corr = np.corrcoef(regRCFittedValues, regRCs)[0, 1]
	corr = np.round(corr, 2)
	maxi1 = np.nanmax(regRCFittedValues)
	maxi2 = np.nanmax(regRCs)
	maxiRegRC = max(maxi1, maxi2)
	ax.plot(regRCs, regRCFittedValues, color='g', marker='s', alpha=0.01)

	corr = np.corrcoef(highRCFittedValues, highRCs)[0, 1]
	corr = np.round(corr, 2)
	maxi1 = np.nanmax(highRCFittedValues)
	maxi2 = np.nanmax(highRCs)
	maxiHighRC = max(maxi1, maxi2)
	ax.plot(highRCs, highRCFittedValues, color='g', marker='s', alpha=0.01)

	maxi = max(maxiRegRC, maxiHighRC)

	ax.text((maxi-25), 10, corr, ha='center', va='center')
	ax.set_xlabel("observed")
	ax.set_ylabel("predicted")
	ax.set_xlim(0, maxi)
	ax.set_ylim(0, maxi)
	ax.plot([0, maxi], [0, maxi], 'k-', color='r')
	ax.set_aspect('equal', adjustable='box')
	fig.savefig(figName)
