import pytest

from CRADLE.correctbiasutils import outputBWFile, outputNormalizedBWFile, figureFileName

@pytest.mark.parametrize("outputDir,filename,result", [
    ('test/', 'foo.bw', 'test/foo_corrected.bw'),
    ('test', 'foo.bw', 'test/foo_corrected.bw'),
    ('test/', 'bar/foo.bw', 'test/foo_corrected.bw'),
    ('test', 'bar/foo.bw', 'test/foo_corrected.bw'),
    ('test1/test2', 'bar/foo.bw', 'test1/test2/foo_corrected.bw'),
    ('test1/test2', 'bar/foo.bw', 'test1/test2/foo_corrected.bw'),
])
def testOutputBWFile(outputDir, filename, result):
    assert outputBWFile(outputDir, filename) == result

@pytest.mark.parametrize("outputDir,filename,result", [
    ('test/', 'foo.bw', 'test/foo_normalized.bw'),
    ('test', 'foo.bw', 'test/foo_normalized.bw'),
    ('test/', 'bar/foo.bw', 'test/foo_normalized.bw'),
    ('test', 'bar/foo.bw', 'test/foo_normalized.bw'),
    ('test1/test2', 'bar/foo.bw', 'test1/test2/foo_normalized.bw'),
    ('test1/test2', 'bar/foo.bw', 'test1/test2/foo_normalized.bw'),
])
def testOutputNormalizedBWFile(outputDir, filename, result):
    assert outputNormalizedBWFile(outputDir, filename) == result

@pytest.mark.parametrize("outputDir,bwFilename,result", [
    ('test/', 'foo.bw', 'test/fit_foo.png'),
    ('test', 'foo.bw', 'test/fit_foo.png'),
    ('test/', 'bar/foo.bw', 'test/fit_foo.png'),
    ('test', 'bar/foo.bw', 'test/fit_foo.png'),
    ('test1/test2', 'bar/foo.bw', 'test1/test2/fit_foo.png'),
    ('test1/test2', 'bar/foo.bw', 'test1/test2/fit_foo.png'),
])
def testFigureFileName(outputDir, bwFilename, result):
    assert figureFileName(outputDir, bwFilename) == result