import numpy as np
import pytest
import pyximport; pyximport.install()

from CRADLE.correctbiasutils.cython import coalesceSections

@pytest.mark.parametrize("starts,values,analysisEnd,stepSize,sectionCount,startEntries,endEntries,valueEntries", [
	(
		np.arange(0, 0),
		np.array([]),
		1,
		1,
		0,
		[],
		[],
		[]
	),
	(
		np.arange(0, 10),
		np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
		11,
		1,
		1,
		[0],
		[10],
		[1.0]
	),
	(
		np.arange(0, 30, 3),
		np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
		31,
		3,
		1,
		[0],
		[30],
		[1.0]
	),
	(
		np.arange(10, 31),
		np.array([1, 2, np.nan, np.nan, 1, 1, 1, 2, 2, np.nan, 0, 1, 1, np.nan, 9, 8, 7, 7, 6, 5, 4]),
		32,
		1,
		12,
		[10, 11, 14, 17, 20, 21, 24, 25, 26, 28, 29, 30],
		[11, 12, 17, 19, 21, 23, 25, 26, 28, 29, 30, 31],
		[1.0, 2.0, 1.0, 2.0, 0.0, 1.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0]
	),
	(
		np.arange(10, 52, 2),
		np.array([1, 2, np.nan, np.nan, 1, 1, 1, 2, 2, np.nan, 0, 1, 1, np.nan, 9, 8, 7, 7, 6, 5, 4]),
		53,
		2,
		12,
		[10, 12, 18, 24, 30, 32, 38, 40, 42, 46, 48, 50],
		[12, 14, 24, 28, 32, 36, 40, 42, 46, 48, 50, 52],
		[1.0, 2.0, 1.0, 2.0, 0.0, 1.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0]
	),
])
def testCoalesceSections(starts, values, analysisEnd, stepSize, sectionCount, startEntries, endEntries, valueEntries):
	idx = np.where(np.isnan(values) == False)[0]
	starts = starts[idx]
	values = values[idx]
	result =  coalesceSections(starts, values, analysisEnd, stepSize)
	assert result[0] == sectionCount
	assert result[1] == startEntries
	assert result[2] == endEntries
	assert result[3] == valueEntries
