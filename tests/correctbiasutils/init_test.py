import numpy as np
import pyBigWig
import pytest

import CRADLE.correctbiasutils as utils

from tests.mocks.BigWig import BigWig
from tests.mocks.TwoBit import TwoBit

from CRADLE.correctbiasutils import ChromoRegion, ChromoRegionSet

@pytest.mark.parametrize("outputDir,filename,result", [
	('test/', 'foo.bw', 'test/foo_corrected.bw'),
	('test', 'foo.bw', 'test/foo_corrected.bw'),
	('test/', 'bar/foo.bw', 'test/foo_corrected.bw'),
	('test', 'bar/foo.bw', 'test/foo_corrected.bw'),
	('test1/test2', 'bar/foo.bw', 'test1/test2/foo_corrected.bw'),
	('test1/test2', 'bar/foo.bw', 'test1/test2/foo_corrected.bw'),
])
def testOutputBWFile(outputDir, filename, result):
	assert utils.outputBWFile(outputDir, filename) == result

@pytest.mark.parametrize("outputDir,filename,result", [
	('test/', 'foo.bw', 'test/foo_normalized.bw'),
	('test', 'foo.bw', 'test/foo_normalized.bw'),
	('test/', 'bar/foo.bw', 'test/foo_normalized.bw'),
	('test', 'bar/foo.bw', 'test/foo_normalized.bw'),
	('test1/test2', 'bar/foo.bw', 'test1/test2/foo_normalized.bw'),
	('test1/test2', 'bar/foo.bw', 'test1/test2/foo_normalized.bw'),
])
def testOutputNormalizedBWFile(outputDir, filename, result):
	assert utils.outputNormalizedBWFile(outputDir, filename) == result

@pytest.mark.parametrize("outputDir,bwFilename,result", [
	('test/', 'foo.bw', 'test/fit_foo.png'),
	('test', 'foo.bw', 'test/fit_foo.png'),
	('test/', 'bar/foo.bw', 'test/fit_foo.png'),
	('test', 'bar/foo.bw', 'test/fit_foo.png'),
	('test1/test2', 'bar/foo.bw', 'test1/test2/fit_foo.png'),
	('test1/test2', 'bar/foo.bw', 'test1/test2/fit_foo.png'),
])
def testFigureFileName(outputDir, bwFilename, result):
	assert utils.figureFileName(outputDir, bwFilename) == result

@pytest.mark.parametrize("observedData,header,scaler,regions,resultData", [
	(
		{'chr17': [np.nan, np.nan, 0., 1., 1., 5., 6., 6., 0., 7., 8., 10.]},
		[('chr17', 12)],
		2.0,
		ChromoRegionSet([ChromoRegion('chr17', 1, 5), ChromoRegion('chr17', 8, 12)]),
		{'chr17': [np.nan, np.nan, np.nan, 0., 0., np.nan, np.nan, np.nan, np.nan, 3., 4., 5.]}
	),
	(
		{
			'chr17': [np.nan, np.nan, 0., 1., 1., 5., 6., 6., 0., 7., 8., 10.],
			'chr18': [np.nan, np.nan, 0., 1., 0., 0., 0., 5., 5., 5.]
		},
		[('chr17', 12), ('chr18', 10)],
		2.0,
		ChromoRegionSet([ChromoRegion('chr17', 1, 5), ChromoRegion('chr17', 8, 11), ChromoRegion('chr18', 8, 12)]),
		{
			'chr17': [np.nan, np.nan, np.nan, 0., 0., np.nan, np.nan, np.nan, np.nan, 3., 4.],
			'chr18': [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 2., 2.]
		}
	),
])
def test_generateNormalizedObBWs(observedData, header, scaler, regions, resultData):
	normObBW = BigWig({})
	observedBW = BigWig(observedData)
	utils._generateNormalizedObBWs(
		header,
		scaler,
		regions,
		observedBW,
		normObBW
	)

	resultBW = BigWig(resultData)
	resultBW.header = header
	assert normObBW == resultBW

@pytest.mark.parametrize('regions,bwFile,result', [
	(ChromoRegionSet([ChromoRegion('chr17', 8086770, 8086780), ChromoRegion('chr17', 8086790, 8086800)]), 'tests/files/test_ctrl.bw', [('chr17', 83_257_441)]),
	(ChromoRegionSet([ChromoRegion('chr17', 8086770, 8086780), ChromoRegion('chr17', 8086790, 8086800)]), 'tests/files/test_exp.bw', [('chr17', 83_257_441)]),
])
def testGetResultHeader(regions, bwFile, result):
	assert utils.getResultBWHeader(regions, bwFile) == result

@pytest.mark.parametrize('trainingSets,bwFile,result', [
	(
		ChromoRegionSet([ChromoRegion('chr17', 8086770, 8086780), ChromoRegion('chr17', 8086790, 8086800)]),
		'tests/files/test_ctrl.bw',
		[0., 0., 0., 130., 130., 128., 128., 128., 130., 130., 122., 118., 122., 122., 122., 122., 122., 122., 122., 120.]),
	(
		ChromoRegionSet([ChromoRegion('chr17', 8086770, 8086780), ChromoRegion('chr17', 8086790, 8086800)]),
		'tests/files/test_exp.bw',
		[0., 0., 0., 130., 130., 128., 128., 128., 130., 130., 122., 118., 122., 122., 122., 122., 122., 122., 122., 120.])
])
def testGetReadCounts(trainingSets, bwFile, result):
	assert utils.getReadCounts(trainingSets, bwFile) == result

@pytest.mark.parametrize("trainingSets,observedReadCounts,ctrlBWNames,experiBWNames,resultBWList", [
	([[1, 2, 3], [1, 2, 3], [1, 2, 3]], [1, 2, 3, 4, 5, 6, 7], ['a', 'b'], ['c'], ['b', 'c']),
	([[1, 2, 3], [1, 2, 3], [1, 2, 3]], [1, 2, 3, 4, 5, 6, 7], ['a'], ['b', 'c'], ['b', 'c']),
	([[1, 2, 3], [1, 2, 3], [1, 2, 3]], [1, 2, 3, 4, 5, 6, 7], ['a', 'b'], ['c', 'd'], ['b', 'c', 'd']),
])
def testGetScalerTasks(trainingSets, observedReadCounts, ctrlBWNames, experiBWNames, resultBWList):
	result = utils.getScalerTasks(trainingSets, observedReadCounts, ctrlBWNames, experiBWNames)
	expectedResult = [(trainingSets, observedReadCounts, fileName) for fileName in resultBWList]
	assert result == expectedResult

@pytest.mark.parametrize("bwFileName,binCount,chromo,start,end,result", [
	# chr17 8,088,775-8,088,787
	# [28., 28., 28., 28., 28., 28., 28., 28., 30., 30., 30., 30.]
	('tests/files/test_ctrl.bw', 1, 'chr17', 8_088_775, 8_088_787, [28.666666666666668]),
	('tests/files/test_ctrl.bw', 2, 'chr17', 8_088_775, 8_088_787, [28., 29.333333333333332]),
	('tests/files/test_ctrl.bw', 3, 'chr17', 8_088_775, 8_088_787, [28., 28., 30.]),
	('tests/files/test_ctrl.bw', 4, 'chr17', 8_088_775, 8_088_787, [28., 28., 28.666666666666668, 30.]),
	('tests/files/test_ctrl.bw', 5, 'chr17', 8_088_775, 8_088_787, [28., 28., 28., 28.666666666666668, 30.]),
	('tests/files/test_ctrl.bw', 6, 'chr17', 8_088_775, 8_088_787, [28., 28., 28., 28., 30., 30.]),
	('tests/files/test_ctrl.bw', 7, 'chr17', 8_088_775, 8_088_787, [28., 28., 28., 28., 28., 30., 30.]),
	# chr17 8,086,770-8,086,785
	# [nan,  nan,  nan, 130., 130., 128., 128., 128., 130., 130., 130., 130., 130., 126., 126.]
	('tests/files/test_ctrl.bw', 1, 'chr17', 8_086_770, 8_086_785, [np.nan]),
	('tests/files/test_ctrl.bw', 2, 'chr17', 8_086_770, 8_086_785, [np.nan, 128.75]),
])
def testRegionMeans(bwFileName, binCount, chromo, start, end, result):
	with pyBigWig.open(bwFileName) as bwFile:
		assert np.allclose(utils.regionMeans(bwFile, binCount, chromo, start, end), [np.float32(x) for x in result], atol=0.00001, equal_nan=True)

def testAlignCoordinatesToCovariateFileBoundaries():
	genome = TwoBit({ "chr1": "actgtcgattcgctctcgatatagcatagctac", "chr2": "tctcgatcgctctcgcgctagagatccgag" })
	trainingSet = ChromoRegionSet([
		ChromoRegion("chr1", 2, 20),
		ChromoRegion("chr1", 4, 20),
		ChromoRegion("chr1", 20, 29),
		ChromoRegion("chr1", 20, 32),
		ChromoRegion("chr2", 10, 15)])
	fragLen = 5
	results = ChromoRegionSet([
		ChromoRegion("chr1", 3, 20),
		ChromoRegion("chr1", 4, 20),
		ChromoRegion("chr1", 20, 29),
		ChromoRegion("chr1", 20, 31),
		ChromoRegion("chr2", 10, 15),
	])
	assert utils.alignCoordinatesToCovariateFileBoundaries(genome, trainingSet, fragLen) == results

@pytest.mark.parametrize("populationSize", [(0), (100), (1_000), (10_000), (100_000)])
def testGetScatterplotSampleIndices(populationSize):
	sampleIndices = utils.getScatterplotSampleIndices(populationSize)
	assert len(set(sampleIndices)) == len(sampleIndices)
	assert len(sampleIndices) == (populationSize if populationSize < utils.SCATTERPLOT_SAMPLE_COUNT else utils.SCATTERPLOT_SAMPLE_COUNT)
