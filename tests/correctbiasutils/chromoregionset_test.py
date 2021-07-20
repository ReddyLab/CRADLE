import pytest

from CRADLE.correctbiasutils import ChromoRegion, ChromoRegionSet

@pytest.mark.parametrize("regionSet, region, result", [
	(ChromoRegionSet(), ChromoRegion("chr1", 10, 100), ChromoRegionSet([ChromoRegion("chr1", 10, 100)])),
	(ChromoRegionSet([ChromoRegion("chr1", 10, 100)]), ChromoRegion("chr1", 110, 200), ChromoRegionSet([ChromoRegion("chr1", 10, 100), ChromoRegion("chr1", 110, 200)]))
])
def testChromoRegionSetAddRegion(regionSet, region, result):
	regionSet.addRegion(region)
	assert regionSet == result

@pytest.mark.parametrize("regionSet, result", [
	(ChromoRegionSet([ChromoRegion("chr1", 10, 100)]), ChromoRegionSet([ChromoRegion("chr1", 10, 100)])),
	(ChromoRegionSet([ChromoRegion("chr1", 10, 100), ChromoRegion("chr1", 20, 200)]), ChromoRegionSet([ChromoRegion("chr1", 10, 100), ChromoRegion("chr1", 20, 200)])),
	(ChromoRegionSet([ChromoRegion("chr1", 20, 200), ChromoRegion("chr1", 10, 100)]), ChromoRegionSet([ChromoRegion("chr1", 10, 100), ChromoRegion("chr1", 20, 200)])),
	(ChromoRegionSet([ChromoRegion("chr1", 20, 200), ChromoRegion("chr2", 10, 100)]), ChromoRegionSet([ChromoRegion("chr1", 20, 200), ChromoRegion("chr2", 10, 100)])),
	(ChromoRegionSet([ChromoRegion("chr1", 20, 200), ChromoRegion("chr2", 10, 100), ChromoRegion("chr1", 10, 100)]), ChromoRegionSet([ChromoRegion("chr1", 10, 100), ChromoRegion("chr1", 20, 200), ChromoRegion("chr2", 10, 100)])),
])
def testChromoRegionSetSortRegions(regionSet, result):
	regionSet.sortRegions()
	assert regionSet == result

@pytest.mark.parametrize("regionSet, result", [
	(ChromoRegionSet([ChromoRegion("chr1", 10, 100)]), ChromoRegionSet([ChromoRegion("chr1", 10, 100)])),
	(ChromoRegionSet([ChromoRegion("chr1", 10, 100), ChromoRegion("chr1", 20, 200)]), ChromoRegionSet([ChromoRegion("chr1", 10, 200)])),
	(ChromoRegionSet([ChromoRegion("chr1", 20, 200), ChromoRegion("chr2", 10, 100)]), ChromoRegionSet([ChromoRegion("chr1", 20, 200), ChromoRegion("chr2", 10, 100)])),
	(ChromoRegionSet([ChromoRegion("chr1", 20, 200), ChromoRegion("chr2", 10, 100), ChromoRegion("chr1", 10, 100)]), ChromoRegionSet([ChromoRegion("chr1", 10, 200), ChromoRegion("chr2", 10, 100)])),
	(ChromoRegionSet([ChromoRegion("chr1", 20, 200), ChromoRegion("chr2", 10, 100), ChromoRegion("chr1", 10, 100), ChromoRegion("chr2", 10, 100)]), ChromoRegionSet([ChromoRegion("chr1", 10, 200), ChromoRegion("chr2", 10, 100)])),
	(ChromoRegionSet([ChromoRegion("chr1", 20, 200), ChromoRegion("chr2", 10, 100), ChromoRegion("chr1", 10, 100), ChromoRegion("chr2", 100, 200)]), ChromoRegionSet([ChromoRegion("chr1", 10, 200), ChromoRegion("chr2", 10, 200)])),
	(ChromoRegionSet([ChromoRegion("chr1", 20, 200), ChromoRegion("chr2", 10, 100), ChromoRegion("chr1", 10, 100), ChromoRegion("chr2", 101, 200)]), ChromoRegionSet([ChromoRegion("chr1", 10, 200), ChromoRegion("chr2", 10, 100), ChromoRegion("chr2", 101, 200)])),
])
def testChromoRegionMergeRegions(regionSet, result):
	regionSet.mergeRegions()
	assert regionSet == result

@pytest.mark.parametrize("regionSet1, regionSet2, result", [
	(
		ChromoRegionSet([ChromoRegion("chr1", 10, 100), ChromoRegion("chr2", 10, 200)]),
		ChromoRegionSet([ChromoRegion("chr1", 20, 200)]),
		ChromoRegionSet([ChromoRegion("chr1", 10, 100), ChromoRegion("chr1", 20, 200), ChromoRegion("chr2", 10, 200)])
	),
	(
		ChromoRegionSet([ChromoRegion("chr1", 10, 100), ChromoRegion("chr1", 10, 100)]),
		ChromoRegionSet([ChromoRegion("chr1", 20, 200)]),
		ChromoRegionSet([ChromoRegion("chr1", 10, 100), ChromoRegion("chr1", 20, 200), ChromoRegion("chr1", 10, 100)])
	)
])
def testChromoRegionSetAdd(regionSet1, regionSet2, result):
	assert (regionSet1 + regionSet2) == result

@pytest.mark.parametrize("regionSet1, regionSet2, result", [
	(
		ChromoRegionSet([ChromoRegion("chr1", 10, 100)]),
		ChromoRegionSet([ChromoRegion("chr1", 20, 200)]),
		ChromoRegionSet([ChromoRegion("chr1", 10, 20)])
	),
	(
		ChromoRegionSet([ChromoRegion("chr1", 10, 300), ChromoRegion("chr1", 30, 100)]),
		ChromoRegionSet([ChromoRegion("chr1", 20, 200)]),
		ChromoRegionSet([ChromoRegion("chr1", 10, 20), ChromoRegion("chr1", 200, 300)])
	),
	(
		ChromoRegionSet([ChromoRegion("chr1", 10, 300), ChromoRegion("chr1", 30, 100)]),
		ChromoRegionSet([ChromoRegion("chr1", 20, 40), ChromoRegion("chr1", 50, 60)]),
		ChromoRegionSet([ChromoRegion("chr1", 10, 20), ChromoRegion("chr1", 40, 50), ChromoRegion("chr1", 60, 300), ChromoRegion("chr1", 40, 50), ChromoRegion("chr1", 60, 100)])
	),
	(
		ChromoRegionSet([ChromoRegion("chr1", 10, 300), ChromoRegion("chr2", 30, 100)]),
		ChromoRegionSet([ChromoRegion("chr1", 20, 200)]),
		ChromoRegionSet([ChromoRegion("chr1", 10, 20), ChromoRegion("chr1", 200, 300), ChromoRegion("chr2", 30, 100)])
	),
])
def testChromoRegionSetSubtract(regionSet1, regionSet2, result):
    assert (regionSet1 - regionSet2) == result

@pytest.mark.parametrize("regionSet, length", [
	(ChromoRegionSet([ChromoRegion("chr1", 10, 100), ChromoRegion("chr1", 110, 200)]), 2),
	(ChromoRegionSet([ChromoRegion("chr1", 10, 100), ChromoRegion("chr1", 110, 200), ChromoRegion("chr2", 110, 200)]), 3)
])
def testChromoRegionSetLength(regionSet, length):
	assert len(regionSet) == length

@pytest.mark.parametrize("regionSet1, regionSet2, result", [
	(ChromoRegionSet([ChromoRegion("chr1", 10, 100)]), ChromoRegionSet([ChromoRegion("chr1", 10, 100)]), True),
	(ChromoRegionSet([ChromoRegion("chr1", 10, 100)]), ChromoRegionSet([ChromoRegion("chr2", 10, 100)]), False),
	(ChromoRegionSet([ChromoRegion("chr1", 10, 100), ChromoRegion("chr1", 20, 200)]), ChromoRegionSet([ChromoRegion("chr1", 10, 100), ChromoRegion("chr1", 20, 200)]), True),
	(ChromoRegionSet([ChromoRegion("chr1", 10, 100), ChromoRegion("chr1", 20, 200)]), ChromoRegionSet([ChromoRegion("chr1", 20, 200), ChromoRegion("chr1", 10, 100)]), True),
	(ChromoRegionSet([ChromoRegion("chr1", 10, 100), ChromoRegion("chr2", 20, 200)]), ChromoRegionSet([ChromoRegion("chr2", 20, 200), ChromoRegion("chr1", 10, 100)]), True),
	(ChromoRegionSet([ChromoRegion("chr1", 10, 100), ChromoRegion("chr1", 20, 200)]), ChromoRegionSet([ChromoRegion("chr1", 10, 100), ChromoRegion("chr2", 20, 200)]), False),

])
def testChromoRegionSetEqual(regionSet1, regionSet2, result):
	assert (regionSet1 == regionSet2) == result
