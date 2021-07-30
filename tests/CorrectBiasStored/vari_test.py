import pytest
import pyximport; pyximport.install()

from CRADLE.CorrectBiasStored.vari import setAnlaysisRegion
from CRADLE.correctbiasutils import ChromoRegion, ChromoRegionSet
from tests.mocks.BigWig import BigWig

@pytest.mark.parametrize("regionSet1, blacklistRegionSet, bigWig, result", [
	(
		ChromoRegionSet([ChromoRegion("chr1", 10, 100)]),
		ChromoRegionSet([ChromoRegion("chr1", 20, 200)]),
		BigWig({'chr1': [float(1) for _ in range(400)]}),
		ChromoRegionSet([ChromoRegion("chr1", 10, 20)])
	),
	(
		ChromoRegionSet([ChromoRegion("chr1", 10, 300), ChromoRegion("chr1", 30, 100)]),
		ChromoRegionSet([ChromoRegion("chr1", 20, 200)]),
		BigWig({'chr1': [float(1) for _ in range(150)]}),
		ChromoRegionSet([ChromoRegion("chr1", 10, 20)])
	),
	(
		ChromoRegionSet([ChromoRegion("chr1", 10, 300), ChromoRegion("chr1", 30, 100)]),
		ChromoRegionSet([ChromoRegion("chr1", 20, 40), ChromoRegion("chr1", 50, 60)]),
		BigWig({'chr1': [float(1) for _ in range(200)]}),
		ChromoRegionSet([ChromoRegion("chr1", 10, 20), ChromoRegion("chr1", 40, 50), ChromoRegion("chr1", 60, 200)])
	),
	(
		ChromoRegionSet([ChromoRegion("chr1", 10, 300), ChromoRegion("chr2", 30, 100)]),
		ChromoRegionSet([ChromoRegion("chr1", 20, 200)]),
		BigWig({'chr1': [float(1) for _ in range(400)]}),
		ChromoRegionSet([ChromoRegion("chr1", 10, 20), ChromoRegion("chr1", 200, 300)])
	),
	(
		ChromoRegionSet([ChromoRegion("chr1", 10, 300), ChromoRegion("chr2", 30, 100)]),
		None,
		BigWig({'chr1': [float(1) for _ in range(400)]}),
		ChromoRegionSet([ChromoRegion("chr1", 10, 300),])
	),
])
def testSetAnlaysisRegion(regionSet1, blacklistRegionSet, bigWig, result):
	assert setAnlaysisRegion(regionSet1, blacklistRegionSet, bigWig) == result
