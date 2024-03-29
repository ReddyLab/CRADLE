import pytest

from CRADLE.correctbiasutils import divideGenome, ChromoRegion, ChromoRegionSet

# Semi-random regions
rand_regions = ChromoRegionSet([ChromoRegion('chr1', 100, 30789), ChromoRegion('chr1', 100, 123123), ChromoRegion('chr1', 124434, 473629), ChromoRegion('chr1', 1234756, 1657483)])

# Regions that are exactly, or 1 off of, a multiple of genomeBinSize long
gbs_regions = ChromoRegionSet([ChromoRegion('chr2', 100, 200100), ChromoRegion('chr2', 124434, 524435), ChromoRegion('chr2', 624434, 824433)])

@pytest.mark.parametrize("regions,baseBinSize,genomeBinSize,result", [
	(rand_regions, 1, 50_000, [
		('chr1', 100, 30789), ('chr1', 100, 50100), ('chr1', 50100, 100100), ('chr1', 100100, 123123),
		('chr1', 124434, 174434), ('chr1', 174434, 224434), ('chr1', 224434, 274434), ('chr1', 274434, 324434),
		('chr1', 324434, 374434), ('chr1', 374434, 424434), ('chr1', 424434, 473629), ('chr1', 1234756, 1284756),
		('chr1', 1284756, 1334756), ('chr1', 1334756, 1384756), ('chr1', 1384756, 1434756), ('chr1', 1434756, 1484756),
		('chr1', 1484756, 1534756), ('chr1', 1534756, 1584756), ('chr1', 1584756, 1634756), ('chr1', 1634756, 1657483)
		]),
	(rand_regions, 30, 50_000, [
		('chr1', 100, 30789), ('chr1', 100, 50080), ('chr1', 50080, 100060), ('chr1', 100060, 123123),
		('chr1', 124434, 174414), ('chr1', 174414, 224394), ('chr1', 224394, 274374), ('chr1', 274374, 324354),
		('chr1', 324354, 374334), ('chr1', 374334, 424314), ('chr1', 424314, 473629), ('chr1', 1234756, 1284736),
		('chr1', 1284736, 1334716), ('chr1', 1334716, 1384696), ('chr1', 1384696, 1434676), ('chr1', 1434676, 1484656),
		('chr1', 1484656, 1534636), ('chr1', 1534636, 1584616), ('chr1', 1584616, 1634596), ('chr1', 1634596, 1657483)]),
	(rand_regions, 33, 11234, [
		('chr1', 100, 11320), ('chr1', 11320, 22540), ('chr1', 22540, 30789), ('chr1', 100, 11320),
		('chr1', 11320, 22540), ('chr1', 22540, 33760), ('chr1', 33760, 44980), ('chr1', 44980, 56200),
		('chr1', 56200, 67420), ('chr1', 67420, 78640), ('chr1', 78640, 89860), ('chr1', 89860, 101080),
		('chr1', 101080, 112300), ('chr1', 112300, 123123), ('chr1', 124434, 135654), ('chr1', 135654, 146874),
		('chr1', 146874, 158094), ('chr1', 158094, 169314), ('chr1', 169314, 180534), ('chr1', 180534, 191754),
		('chr1', 191754, 202974), ('chr1', 202974, 214194), ('chr1', 214194, 225414), ('chr1', 225414, 236634),
		('chr1', 236634, 247854), ('chr1', 247854, 259074), ('chr1', 259074, 270294), ('chr1', 270294, 281514),
		('chr1', 281514, 292734), ('chr1', 292734, 303954), ('chr1', 303954, 315174), ('chr1', 315174, 326394),
		('chr1', 326394, 337614), ('chr1', 337614, 348834), ('chr1', 348834, 360054), ('chr1', 360054, 371274),
		('chr1', 371274, 382494), ('chr1', 382494, 393714), ('chr1', 393714, 404934), ('chr1', 404934, 416154),
		('chr1', 416154, 427374), ('chr1', 427374, 438594), ('chr1', 438594, 449814), ('chr1', 449814, 461034),
		('chr1', 461034, 472254), ('chr1', 472254, 473629), ('chr1', 1234756, 1245976), ('chr1', 1245976, 1257196),
		('chr1', 1257196, 1268416), ('chr1', 1268416, 1279636), ('chr1', 1279636, 1290856), ('chr1', 1290856, 1302076),
		('chr1', 1302076, 1313296), ('chr1', 1313296, 1324516), ('chr1', 1324516, 1335736), ('chr1', 1335736, 1346956),
		('chr1', 1346956, 1358176), ('chr1', 1358176, 1369396), ('chr1', 1369396, 1380616), ('chr1', 1380616, 1391836),
		('chr1', 1391836, 1403056), ('chr1', 1403056, 1414276), ('chr1', 1414276, 1425496), ('chr1', 1425496, 1436716),
		('chr1', 1436716, 1447936), ('chr1', 1447936, 1459156), ('chr1', 1459156, 1470376), ('chr1', 1470376, 1481596),
		('chr1', 1481596, 1492816), ('chr1', 1492816, 1504036), ('chr1', 1504036, 1515256), ('chr1', 1515256, 1526476),
		('chr1', 1526476, 1537696), ('chr1', 1537696, 1548916), ('chr1', 1548916, 1560136), ('chr1', 1560136, 1571356),
		('chr1', 1571356, 1582576), ('chr1', 1582576, 1593796), ('chr1', 1593796, 1605016), ('chr1', 1605016, 1616236),
		('chr1', 1616236, 1627456), ('chr1', 1627456, 1638676), ('chr1', 1638676, 1649896), ('chr1', 1649896, 1657483)
		]),
	(gbs_regions, 1, 50_000, [
		('chr2', 100, 50100), ('chr2', 50100, 100100), ('chr2', 100100, 150100), ('chr2', 150100, 200100),
		('chr2', 124434, 174434), ('chr2', 174434, 224434), ('chr2', 224434, 274434), ('chr2', 274434, 324434),
		('chr2', 324434, 374434), ('chr2', 374434, 424434), ('chr2', 424434, 474434), ('chr2', 474434, 524434),
		('chr2', 524434, 524435), ('chr2', 624434, 674434), ('chr2', 674434, 724434), ('chr2', 724434, 774434),
		('chr2', 774434, 824433)]),
	(gbs_regions, 30, 50_000, [
		('chr2', 100, 50080), ('chr2', 50080, 100060), ('chr2', 100060, 150040), ('chr2', 150040, 200020),
		('chr2', 200020, 200100), ('chr2', 124434, 174414), ('chr2', 174414, 224394), ('chr2', 224394, 274374),
		('chr2', 274374, 324354), ('chr2', 324354, 374334), ('chr2', 374334, 424314), ('chr2', 424314, 474294),
		('chr2', 474294, 524274), ('chr2', 524274, 524435), ('chr2', 624434, 674414), ('chr2', 674414, 724394),
		('chr2', 724394, 774374), ('chr2', 774374, 824354), ('chr2', 824354, 824433)]),
])
def testDivideGenomeRand(regions, baseBinSize, genomeBinSize, result):
	assert divideGenome(regions, baseBinSize, genomeBinSize) == result
