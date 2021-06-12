import numpy as np
import pytest
import pyximport; pyximport.install()

from CRADLE.CorrectBiasStored.calculateOneBP import getReadCounts

@pytest.mark.parametrize("rawReadCounts,rowCount,scaler,result", [
	([(np.array([1, 2, 3, 4, 5]), 5)], 5, 2, [0.5, 1., 1.5, 2., 2.5]),
	([
		(np.array([1, 2, 3, 4, 5]), 5),
		(np.array([2, 4, 6, 8, 10]), 5)
	], 10, 2, [0.5, 1., 1.5, 2., 2.5, 1., 2., 3., 4., 5.]),
	([
		(np.array([1, 2, 3, 4, 5]), 5),
		(np.array([2, np.nan, 6, 8, 0]), 5)
	], 10, 2, [0.5, 1., 1.5, 2., 2.5, 1., 0., 3., 4., 0.])
])
def testGetCoefs(rawReadCounts, rowCount, scaler, result):
	try:
		np.testing.assert_equal(
			getReadCounts(rawReadCounts, rowCount, scaler),
			np.array(result)
		)
	except AssertionError as error:
		raise error
	else:
		assert True
