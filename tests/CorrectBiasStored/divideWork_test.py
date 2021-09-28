import pytest
import pyximport; pyximport.install()


from CRADLE.correctbiasutils import ChromoRegionSet
from CRADLE.correctbiasutils import divideGenome
from CRADLE.CorrectBiasStored.correctBias import divideWork, divideWorkByChrom

@pytest.mark.parametrize("regionFile,blacklistFile,processCount,jobGroupCount", [
	('tests/files/regions/actual_probes_hg38_merged.bed', 'tests/files/regions/hg38_filter_out.bed', 100, 100),
	('tests/files/regions/actual_probes_hg38_merged.bed', 'tests/files/regions/hg38_filter_out.bed', 50, 50),
	('tests/files/regions/actual_probes_hg38_merged.bed', 'tests/files/regions/hg38_filter_out.bed', 20, 20),
	('tests/files/regions/actual_probes_hg38_merged.bed', 'tests/files/regions/hg38_filter_out.bed', 10, 10),
	('tests/files/regions/actual_probes_hg38_merged.bed', 'tests/files/regions/hg38_filter_out.bed', 1, 1),
	('tests/files/regions/regionfile_correctionModel.bed', 'tests/files/regions/hg38_filter_out.bed', 100, 100),
	('tests/files/regions/regionfile_correctionModel.bed', 'tests/files/regions/hg38_filter_out.bed', 50, 50),
	('tests/files/regions/regionfile_correctionModel.bed', 'tests/files/regions/hg38_filter_out.bed', 20, 20),
	('tests/files/regions/regionfile_correctionModel.bed', 'tests/files/regions/hg38_filter_out.bed', 10, 10),
	('tests/files/regions/regionfile_correctionModel.bed', 'tests/files/regions/hg38_filter_out.bed', 1, 1),
	('tests/files/regions/small_regions.bed', 'tests/files/regions/hg38_filter_out.bed', 100, 73), # There are only 73 regions
	('tests/files/regions/small_regions.bed', 'tests/files/regions/hg38_filter_out.bed', 50, 50),
	('tests/files/regions/small_regions.bed', 'tests/files/regions/hg38_filter_out.bed', 20, 20),
	('tests/files/regions/small_regions.bed', 'tests/files/regions/hg38_filter_out.bed', 10, 10),
	('tests/files/regions/small_regions.bed', 'tests/files/regions/hg38_filter_out.bed', 1, 1),
])
def testDivideWork(regionFile, blacklistFile, processCount, jobGroupCount):
	regionSet = ChromoRegionSet.loadBed(regionFile)
	regionSet.mergeRegions()

	blacklistRegionSet = ChromoRegionSet.loadBed(blacklistFile) if blacklistFile else None
	if blacklistRegionSet is not None:
		blacklistRegionSet.mergeRegions()
		regions = regionSet - blacklistRegionSet
	else:
		regions = regionSet
	binnedRegions = divideGenome(regions)

	jobGroups = divideWork(binnedRegions, regions.cumulativeRegionSize, processCount)
	jobGroups = divideWorkByChrom(jobGroups)

	assert len(jobGroups) <= processCount
	assert len(jobGroups) == jobGroupCount
