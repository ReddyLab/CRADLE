import pytest

import CRADLE.correctbiasutils as utils

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

@pytest.mark.parametrize('regions,bwFile,result', [
	([('chr17', 8086770, 8086780), ('chr17', 8086790, 8086800)], 'tests/files/test_ctrl.bw', [('chr17', 83_257_441)]),
	([('chr17', 8086770, 8086780), ('chr17', 8086790, 8086800)], 'tests/files/test_exp.bw', [('chr17', 83_257_441)]),
])
def testGetResultHeader(regions, bwFile, result):
	assert utils.getResultBWHeader(regions, bwFile) == result

@pytest.mark.parametrize('trainingSets,bwFile,result', [
	(
		[('chr17', 8086770, 8086780), ('chr17', 8086790, 8086800)],
		'tests/files/test_ctrl.bw',
		[0., 0., 0., 130., 130., 128., 128., 128., 130., 130., 122., 118., 122., 122., 122., 122., 122., 122., 122., 120.]),
	(
		[('chr17', 8086770, 8086780), ('chr17', 8086790, 8086800)],
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
	expectedResult = [[trainingSets, observedReadCounts, fileName] for fileName in resultBWList]
	assert result == expectedResult

@pytest.mark.parametrize("populationSize", [(0), (100), (1_000), (10_000), (100_000)])
def testGetScatterplotSampleIndices(populationSize):
	sampleIndices = utils.getScatterplotSampleIndices(populationSize)
	assert len(set(sampleIndices)) == len(sampleIndices)
	assert len(sampleIndices) == (populationSize if populationSize < utils.SCATTERPLOT_SAMPLE_COUNT else utils.SCATTERPLOT_SAMPLE_COUNT)
