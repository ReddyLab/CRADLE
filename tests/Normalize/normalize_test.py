import numpy as np
import pytest

from CRADLE.Normalize.normalize import (RegionDtype, getNonOverlappingRegions,
                                        mergeRegions)


def testMergeRegions():
	mergedRegions, overlaps = mergeRegions('tests/files/regions/normalize_merge_regions.bed')

	expectedMergedRegions = np.array([
		("chr10", 68885301, 68885351, False),
		("chr10", 68885359, 68885399, False),
		("chr11", 92913994, 93412926, False),
		("chr21", 40804100, 40804149, False),
		("chr21", 40804159, 40804249, False),
		("chr5", 95895849, 96522862, False),
	], dtype=RegionDtype)
	expectedOverlaps = np.array([
		("chr10", 68885310, 68885349, True),
		("chr21", 40804109, 40804140, True),
		("chr21", 40804179, 40804209, True),
	], dtype=RegionDtype)

	assert np.array_equal(mergedRegions, expectedMergedRegions)
	assert np.array_equal(overlaps, expectedOverlaps)

def testGetNonOverlappingRegions():
	mergedRegions, overlaps = mergeRegions('tests/files/regions/normalize_merge_regions.bed')
	nonOverlaps = getNonOverlappingRegions(mergedRegions, overlaps)

	expectedNonOverlaps = np.array([
		("chr10", 68885301, 68885310, False),
		("chr10", 68885349, 68885351, False),
		("chr10", 68885359, 68885399, False),
		("chr11", 92913994, 93412926, False),
		("chr21", 40804100, 40804109, False),
		("chr21", 40804140, 40804149, False),
		("chr21", 40804159, 40804179, False),
		("chr21", 40804209, 40804249, False),
		("chr5", 95895849, 96522862, False)
	], dtype=RegionDtype)
	print(nonOverlaps)

	assert np.array_equal(nonOverlaps, expectedNonOverlaps)
