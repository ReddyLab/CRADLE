import numpy as np
import pytest

from tests.mocks.BigWig import BigWig

def testAddHeader():
	data = {'chr1': [np.nan, np.nan, 0., 1., 1., 6., 6., 6., 0., 7., 8., 10.]}
	bw = BigWig(data)
	bw.addHeader([('chr1', len(data['chr1']))])
	assert bw.header == [('chr1', 12)]

@pytest.mark.parametrize("data,location,numpy,result", [
	({'chr1': [np.nan, np.nan, 0., 1., 1., 5., 6., 6., 0., 7., 8., 10.]}, ('chr1', 3, 6), False, [1., 1., 5.]),
	({'chr1': [np.nan, np.nan, 0., 1., 1., 5., 6., 6., 0., 7., 8., 10.]}, ('chr1', 3, 6), True, np.array([1., 1., 5.])),
	({'chr1': [np.nan, np.nan, 0., 1., 1., 5., 6., 6., 0., 7., 8., 10.]}, ('chr1', 0, 6), True, np.array([np.nan, np.nan, 0., 1., 1., 5.])),
	({'chr1': [np.nan, np.nan, 0., 1., 1., 5., 6., 6., 0., 7., 8., 10.]}, ('chr5', 3, 6), False, [np.nan, np.nan, np.nan]),
	({'chr1': [np.nan, np.nan, 0., 1., 1., 5., 6., 6., 0., 7., 8., 10.]}, ('chr1', 10, 15), False, [8., 10., np.nan, np.nan, np.nan]),
])
def testValues(data, location, numpy, result):
	bw = BigWig(data)
	assert np.array_equal(bw.values(location[0], location[1], location[2], numpy), result, equal_nan=True)

@pytest.mark.parametrize("data,chroms,starts,ends,values,results", [
	(
		{'chr1': [np.nan, np.nan, 0., 1., 1., 5., 6., 6., 0., 7., 8., 10.]},
		['chr1'] * 6, [10, 11, 12, 13, 14, 15], [11, 12, 13, 14, 15, 16], [10., 11., 12., 13., 14., 15.],
		[{'chr1': [np.nan, np.nan, 0., 1., 1., 5., 6., 6., 0., 7., 10., 11., 12., 13., 14., 15.]}]
	),
	(
		{'chr1': [np.nan, np.nan, 0., 1., 1., 5., 6., 6., 0., 7., 8., 10.]},
		['chr1'] * 4, [1, 2, 3, 4], [2, 3, 4, 5], [10., 11., 12.,13.],
		[{'chr1': [np.nan, 10., 11., 12., 13., 5., 6., 6., 0., 7., 8., 10.]}]
	),
	(
		{'chr1': [np.nan, np.nan, 0., 1., 1., 5.]},
		['chr1'] * 3, [6, 7, 8], [7, 8, 9], [6., 7., 8.],
		[{'chr1': [np.nan, np.nan, 0., 1., 1., 5., 6., 7., 8.]}]
	),
	(
		{'chr1': [np.nan, np.nan, 0., 1., 1., 5.]},
		['chr1'] * 3, [8, 9, 10], [9, 10, 11], [6., 7., 8.],
		[{'chr1': [np.nan, np.nan, 0., 1., 1., 5., np.nan, np.nan, 6., 7., 8.]}]
	),
	(
		{'chr1': [np.nan, np.nan, 0., 1., 1., 5.]},
		['chr2'] * 3, [8, 9, 10], [9, 10, 11], [6., 7., 8.],
		[
			{
				'chr1': [np.nan, np.nan, 0., 1., 1., 5.],
				'chr2': [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 6., 7., 8.]
			}
		]
	),
	(
		{'chr1': [np.nan, np.nan, 0., 1., 1., 5., 6., 6., 0., 7., 8., 10.]},
		['chr1'], [10], [16], [15.],
		[{'chr1': [np.nan, np.nan, 0., 1., 1., 5., 6., 6., 0., 7., 15., 15., 15., 15., 15., 15.]}]
	),
	(
		{'chr1': [np.nan, np.nan, 0., 1., 1., 5., 6., 6., 0., 7., 8., 10.]},
		['chr1'], [1], [5], [10.],
		[{'chr1': [np.nan, 10., 10., 10., 10., 5., 6., 6., 0., 7., 8., 10.]}]
	),
	(
		{'chr1': [np.nan, np.nan, 0., 1., 1., 5.]},
		['chr1'], [6], [9], [6.],
		[{'chr1': [np.nan, np.nan, 0., 1., 1., 5., 6., 6., 6.]}]
	),
	(
		{'chr1': [np.nan, np.nan, 0., 1., 1., 5.]},
		['chr1'], [8], [11], [7.],
		[{'chr1': [np.nan, np.nan, 0., 1., 1., 5., np.nan, np.nan, 7., 7., 7.]}]
	),
	(
		{'chr1': [np.nan, np.nan, 0., 1., 1., 5.]},
		['chr2'], [8], [11], [8.],
		[
			{
				'chr1': [np.nan, np.nan, 0., 1., 1., 5.],
				'chr2': [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 8., 8., 8.]
			}
		]
	),
])
def testAddEntries(data, chroms, starts, ends, values, results):
	bw = BigWig(data)
	bw.addEntries(chroms, starts, ends, values)
	allChroms = set(data.keys())
	allChroms = allChroms.intersection(set(chroms))
	for chrom, result in zip(allChroms, results):
		assert bw.data[chrom] == result[chrom]

@pytest.mark.parametrize("data1,data2", [
	({'chr1': [np.nan, np.nan, 0., 1., 1., 5., 6., 6., 0., 7., 8., 10.]}, {'chr1': [np.nan, np.nan, 0., 1., 1., 5., 6., 6., 0., 7., 8., 10.]}),
	(
		{
			'chr1': [np.nan, np.nan, 0., 1., 1., 5.],
			'chr2': [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 6., 7., 8.]
		},
		{
			'chr1': [np.nan, np.nan, 0., 1., 1., 5.],
			'chr2': [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 6., 7., 8.]
		}
	)
])
def testEqual(data1, data2):
	bw1 = BigWig(data1)
	bw2 = BigWig(data2)
	assert bw1 == bw2

@pytest.mark.parametrize("data1,data2", [
	({'chr1': [np.nan, np.nan, 0., 1., 1., 5., 6., 6., 0., 7., 8., 11.]}, {'chr1': [np.nan, np.nan, 0., 1., 1., 5., 6., 6., 0., 7., 8., 10.]}),
	(
		{
			'chr1': [np.nan, np.nan, 0., 1., 1., 5.],
			'chr2': [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 6., 7., 8.]
		},
		{
			'chr1': [np.nan, np.nan, 0., 1., 1., 5.],
			'chr3': [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 6., 7., 8.]
		}
	)
])
def testNotEqual(data1, data2):
	bw1 = BigWig(data1)
	bw2 = BigWig(data2)
	assert bw1 != bw2

@pytest.mark.parametrize("data", [
	({'chr1': [np.nan, np.nan, 0., 1., 1., 5., 6., 6., 0., 7., 8., 11.]}),
	(
		{
			'chr1': [np.nan, np.nan, 0., 1., 1., 5.],
			'chr2': [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 6., 7., 8.]
		},
	)
])
def testEqualSame(data):
	bw = BigWig(data)
	assert bw == bw
