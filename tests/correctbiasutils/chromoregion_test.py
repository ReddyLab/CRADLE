import pytest

from CRADLE.correctbiasutils import ChromoRegion, ChromoRegionMergeException

@pytest.mark.parametrize("region1,region2,result", [
	(ChromoRegion("chr1", 10, 100), ChromoRegion("chr2", 90, 200), False),
	(ChromoRegion("chr1", 10, 100), ChromoRegion("chr1", 90, 200), True),
	(ChromoRegion("chr1", 10, 100), ChromoRegion("chr1", 100, 200), True),
	(ChromoRegion("chr1", 10, 100), ChromoRegion("chr1", 101, 200), False),
	(ChromoRegion("chr1", 10, 100), ChromoRegion("chr1", 200, 300), False),
	(ChromoRegion("chr2", 90, 200), ChromoRegion("chr1", 10, 100), False),
	(ChromoRegion("chr1", 90, 200), ChromoRegion("chr1", 10, 100), True),
	(ChromoRegion("chr1", 100, 200), ChromoRegion("chr1", 10, 100), True),
	(ChromoRegion("chr1", 101, 200), ChromoRegion("chr1", 10, 100), False),
	(ChromoRegion("chr1", 200, 300), ChromoRegion("chr1", 10, 100), False),
])
def testChromoRegionContiguous(region1, region2, result):
	assert region1.contiguousWith(region2) == result

@pytest.mark.parametrize("region,length", [
	(ChromoRegion("chr1", 10, 100), 90),
	(ChromoRegion("chr2", 90, 200), 110),
])
def testChromoRegionLength(region, length):
	assert len(region) == length

@pytest.mark.parametrize("region1,region2,result", [
	(ChromoRegion("chr1", 10, 100), ChromoRegion("chr1", 10, 100), True),
	(ChromoRegion("chr1", 10, 100), ChromoRegion("chr2", 10, 100), False),
	(ChromoRegion("chr1", 10, 100), ChromoRegion("chr1", 90, 200), False),
])
def testChromoRegionEqual(region1, region2, result):
	assert (region1 == region2) == result

@pytest.mark.parametrize("region1,region2,result", [
	(ChromoRegion("chr1", 10, 100), ChromoRegion("chr1", 10, 100), False),
	(ChromoRegion("chr1", 10, 100), ChromoRegion("chr2", 10, 100), True),
	(ChromoRegion("chr1", 10, 100), ChromoRegion("chr1", 90, 200), True),
	(ChromoRegion("chr2", 10, 100), ChromoRegion("chr1", 10, 100), False),
	(ChromoRegion("chr1", 90, 200), ChromoRegion("chr1", 10, 100), False),
	(ChromoRegion("chr1", 90, 200), ChromoRegion("chr1", 100, 200), True),
	(ChromoRegion("chr1", 100, 200), ChromoRegion("chr1", 90, 200), False),
])
def testChromoRegionLessThan(region1, region2, result):
	assert (region1 < region2) == result

@pytest.mark.parametrize("region1,region2,result,exception", [
	(ChromoRegion("chr1", 10, 100), ChromoRegion("chr1", 10, 100), ChromoRegion("chr1", 10, 100), None),
	(ChromoRegion("chr1", 10, 100), ChromoRegion("chr1", 50, 100), ChromoRegion("chr1", 10, 100), None),
	(ChromoRegion("chr1", 10, 100), ChromoRegion("chr1", 50, 200), ChromoRegion("chr1", 10, 200), None),
	(ChromoRegion("chr1", 10, 100), ChromoRegion("chr1", 50, 200), ChromoRegion("chr1", 10, 200), None),
	(ChromoRegion("chr1", 50, 100), ChromoRegion("chr1", 10, 100), ChromoRegion("chr1", 10, 100), None),
	(ChromoRegion("chr1", 50, 200), ChromoRegion("chr1", 10, 100), ChromoRegion("chr1", 10, 200), None),
	(ChromoRegion("chr1", 50, 200), ChromoRegion("chr1", 10, 100), ChromoRegion("chr1", 10, 200), None),
	(ChromoRegion("chr1", 50, 200), ChromoRegion("chr2", 10, 100), None, ChromoRegionMergeException),
	(ChromoRegion("chr1", 50, 200), ChromoRegion("chr2", 10, 30), None, ChromoRegionMergeException),
	(ChromoRegion("chr1", 50, 200), ChromoRegion("chr1", 10, 30), None, ChromoRegionMergeException),
	(ChromoRegion("chr1", 10, 30), ChromoRegion("chr1", 50, 200), None, ChromoRegionMergeException),
])
def testChromoRegionAdd(region1, region2, result, exception):
	if exception is None:
		assert region1 + region2 == result
	else:
		with pytest.raises(exception):
			_ = region1 + region2

@pytest.mark.parametrize("region1,region2,result", [
	(ChromoRegion("chr1", 10, 100), ChromoRegion("chr1", 1, 101), []),
	(ChromoRegion("chr1", 10, 100), ChromoRegion("chr1", 20, 90), [ChromoRegion("chr1", 10, 20), ChromoRegion("chr1", 90, 100)]),
	(ChromoRegion("chr1", 50, 100), ChromoRegion("chr1", 10, 60), [ChromoRegion("chr1", 60, 100)]),
	(ChromoRegion("chr1", 50, 100), ChromoRegion("chr1", 60, 150), [ChromoRegion("chr1", 50, 60)]),
	(ChromoRegion("chr1", 50, 100), ChromoRegion("chr1", 1, 50), [ChromoRegion("chr1", 50, 100)]),
	(ChromoRegion("chr1", 50, 100), ChromoRegion("chr1", 100, 150), [ChromoRegion("chr1", 50, 100)]),
	(ChromoRegion("chr1", 50, 100), ChromoRegion("chr2", 10, 80), [ChromoRegion("chr1", 50, 100)]),
])
def testChromoRegionSub(region1, region2, result):
	assert region1 - region2 == result

@pytest.mark.parametrize("region, newEnd, result, exception", [
	(ChromoRegion("chr1", 10, 100), 90, ChromoRegion("chr1", 10, 90), None),
	(ChromoRegion("chr1", 10, 100), 100, ChromoRegion("chr1", 10, 100), None),
	(ChromoRegion("chr1", 10, 100), 110, ChromoRegion("chr1", 10, 110), None),
	(ChromoRegion("chr1", 10, 100), 9, None, AssertionError),
])
def testChromoRegionSetEnd(region, newEnd, result, exception):
	if exception is None:
		region.end = newEnd
		assert region == result
	else:
		with pytest.raises(exception):
			region.end = newEnd
