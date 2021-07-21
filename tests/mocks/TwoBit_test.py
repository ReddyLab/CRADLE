import pytest

from tests.mocks.TwoBit import TwoBit

@pytest.mark.parametrize("twoBit,chrom,result", [
	(TwoBit({}), None, {}),
	(
		TwoBit({
			'chr1': 'agctagtatcgatagcatcgtcgatcgatcgctcgatcgac',
			'chr2': 'tgcatcgatcgcgcgcgccc'
		}),
		None,
		{'chr1': 41, 'chr2': 20}
	),
	(
		TwoBit({}),
		'chr1',
		None
	),
	(
		TwoBit({
			'chr1': 'agctagtatcgatagcatcgtcgatcgatcgctcgatcgac',
			'chr2': 'tgcatcgatcgcgcgcgccc'
		}),
		'chr1',
		41
	),
	(
		TwoBit({
			'chr1': 'agctagtatcgatagcatcgtcgatcgatcgctcgatcgac',
			'chr2': 'tgcatcgatcgcgcgcgccc'
		}),
		'chr17',
		None
	)
])
def testChroms(twoBit, chrom, result):
	assert twoBit.chroms(chrom) == result

@pytest.mark.parametrize('twoBit, chrom, start, end, result', [
	(
		TwoBit({
			'chr1': 'agctagtatcgatagcatcgtcgatcgatcgctcgatcgac',
			'chr2': 'tgcatcgatcgcgcgcgccc'
		}), 'chr1', 3, 10, 'tagtatc'
	),
	(
		TwoBit({
			'chr1': 'agctagtatcgatagcatcgtcgatcgatcgctcgatcgac',
			'chr2': 'tgcatcgatcgcgcgcgccc'
		}), 'chr2', None, None, 'tgcatcgatcgcgcgcgccc'
	)
])
def testSequence(twoBit, chrom, start, end, result):
	assert twoBit.sequence(chrom, start, end) == result
