import numpy as np
import pytest
import pyximport; pyximport.install()

from CRADLE.CorrectBiasStored.calculateOneBP import getCoefs

@pytest.mark.parametrize("modelParams,selectedCovariates,result", [
	([1, 2, 3, 4, 5, 6, 7], [np.nan, 1, np.nan, 1, 1, 1], [1, np.nan, 2, np.nan, 3, 4, 5]),
	([10, 2, 3, 4, 5, 6, 7], [1, 1, 1, 1, 1, 1], [10, 2, 3, 4, 5, 6, 7]),
	([10, 2, 3, 4, 5, 6, 7], [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan], [10, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]),
])
def testGetCoefs(modelParams, selectedCovariates, result):
	try:
		np.testing.assert_equal(getCoefs(modelParams, selectedCovariates), np.array(result))
	except AssertionError as error:
		raise error
	else:
		assert True
