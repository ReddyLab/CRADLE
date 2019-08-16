import numpy as np
import statsmodels.formula.api as smf
import statsmodels.sandbox.stats.multicomp
import tempfile
import os
import scipy.stats
import pyBigWig

from CRADLE.CallPeak import vari

import warnings

cpdef getVariance(region):
	warnings.filterwarnings('ignore', r'All-NaN slice encountered')
	warnings.filterwarnings('ignore', r'Mean of empty slice')
	warnings.filterwarnings('ignore', r'Degrees of freedom <= 0 for slice')

	regionChromo = region[0]
	regionStart = int(region[1])
	regionEnd = int(region[2])

	numBin = int((regionEnd - regionStart) / vari.BINSIZE1)
	if(numBin == 0):
		numBin = 1

	totalRC = []
	for rep in range(vari.CTRLBW_NUM):
		bw = pyBigWig.open(vari.CTRLBW_NAMES[rep])
		temp = np.array(bw.stats(regionChromo, regionStart, regionEnd,  type="mean", nBins=numBin))
		temp[np.where(temp==None)] = np.nan
		totalRC.append(temp.tolist())
		bw.close()
	
	for rep in range(vari.EXPBW_NUM):
		bw = pyBigWig.open(vari.EXPBW_NAMES[rep])
		temp = np.array(bw.stats(regionChromo, regionStart, regionEnd,  type="mean", nBins=numBin))
		temp[np.where(temp==None)] = np.nan
		totalRC.append(temp.tolist())		
		bw.close()

	var = np.nanvar(np.array(totalRC), axis=0)
	var = np.array(var)
	idx = np.where(np.isnan(var)==True)
	var = np.delete(var, idx)

	if(len(var) > 0):
		return var.tolist()
	else:
		return None


cpdef getRegionCutoff(region):
	warnings.filterwarnings('ignore', r'All-NaN slice encountered')
	warnings.filterwarnings('ignore', r'Mean of empty slice')

	regionChromo = region[0]
	regionStart = int(region[1])
	regionEnd = int(region[2])
	numBin = int((regionEnd - regionStart) / vari.BINSIZE1)
	if(numBin == 0):
		numBin = 1

	sampleRC = []
	for rep in range(vari.CTRLBW_NUM):
		bw = pyBigWig.open(vari.CTRLBW_NAMES[rep])
		temp = np.array(bw.stats(regionChromo, regionStart, regionEnd,  type="mean", nBins=numBin))
		temp[np.where(temp==None)] = np.nan
		sampleRC.append(temp.tolist())
		bw.close()

	ctrlMean = np.nanmean(np.array(sampleRC), axis=0)

	sampleRC = []
	for rep in range(vari.EXPBW_NUM):
		bw = pyBigWig.open(vari.EXPBW_NAMES[rep])
		temp = np.array(bw.stats(regionChromo, regionStart, regionEnd,  type="mean", nBins=numBin))
		temp[np.where(temp==None)] = np.nan
		sampleRC.append(temp.tolist())
		bw.close()

	expMean = np.nanmean(np.array(sampleRC), axis=0)
	del sampleRC

	diff = np.array(np.absolute(expMean - ctrlMean))
	idx = np.where(np.isnan(diff)==True)
	diff = np.delete(diff, idx)

	if(len(diff) > 0):
		return diff.tolist()
	else:
		return None



cpdef defineRegion(region):
	warnings.filterwarnings('ignore', r'All-NaN slice encountered')
	warnings.filterwarnings('ignore', r'Mean of empty slice')

	analysis_chromo = region[0]
	analysis_start = int(region[1])
	analysis_end = int(region[2])

	#### Number of bin
	binNum = int( (analysis_end - analysis_start) / vari.BINSIZE1 )
	region_start = analysis_start
	if(binNum == 0):
		region_end = analysis_end
		lastBin_exist = 0
		binNum = 1
	else:
		if((analysis_start + binNum * vari.BINSIZE1) < analysis_end):
			lastBin_exist = 1   # exist! 
			region_end = region_start + binNum * vari.BINSIZE1
			lastBin_start = region_end
			lastBin_end = analysis_end
		else:
			lastBin_exist = 0
			region_end = analysis_end

	#### ctrlMean
	sampleRC1 = []
	for rep in range(vari.CTRLBW_NUM):
		bw = pyBigWig.open(vari.CTRLBW_NAMES[rep])
		temp = np.array(bw.stats(analysis_chromo, region_start, region_end,  type="mean", nBins=binNum))
		temp[np.where(temp==None)] = np.nan
		temp = temp.tolist()

		if(lastBin_exist == 1):
			last_value = bw.stats(analysis_chromo, lastBin_start, lastBin_end, type="mean", nBins=1)[0]
			if(last_value == None):
				last_value = np.nan
			temp.extend([last_value])

		sampleRC1.append(temp)
		bw.close()
	
	ctrlMean = np.nanmean( np.array(sampleRC1), axis=0)
	del sampleRC1

	#### expMean
	sampleRC2 = []
	for rep in range(vari.EXPBW_NUM):
		bw = pyBigWig.open(vari.EXPBW_NAMES[rep])
		temp = np.array(bw.stats(analysis_chromo, region_start, region_end,  type="mean", nBins=binNum))
		temp[np.where(temp==None)] = np.nan
		temp = temp.tolist()

		if(lastBin_exist == 1):
			last_value = bw.stats(analysis_chromo, lastBin_start, lastBin_end, type="mean", nBins=1)[0]
			if(last_value == None):
				last_value = np.nan
			temp.extend([last_value])
		
		sampleRC2.append(temp)
		bw.close()

	expMean = np.nanmean( np.array(sampleRC2), axis=0)
	del sampleRC2

	#### diff
	diff = np.array(expMean - ctrlMean, dtype=np.float64)
	cdef double [:] diff_view = diff
	del diff

	cdef int idx = 0
	pastGroupType = -2
	numRegion = 0
	definedRegion = []
	
	if(lastBin_exist == 1):
		binNum = binNum + 1
	
	while( idx < binNum ):
		if(np.isnan(diff_view[idx]) == True):
			if(pastGroupType == -2):
				idx = idx + 1
				continue
			else:
				definedRegion.append(regionVector)
				numRegion = numRegion + 1
				idx = idx + 1
				pastGroupType = -2
				continue

		if(np.absolute(diff_view[idx]) > vari.REGION_CUTOFF):
			if(diff_view[idx] > 0):
				currGroupType = 1   # enriched
			else:
				currGroupType = -1   # repressed
		else:
			currGroupType = 0

		if(pastGroupType == -2): # the first region
			regionVector = [analysis_chromo, (analysis_start+vari.BINSIZE1*idx), (analysis_start+vari.BINSIZE1*idx+vari.BINSIZE1), currGroupType]
			if(idx == (binNum-1)):
				regionVector[2] = analysis_end
				definedRegion.append(regionVector)
				numRegion = numRegion + 1
				break
			else:
				pastGroupType = currGroupType
				idx = idx + 1
				continue

		if(currGroupType != pastGroupType):
			## End a previous region
			regionVector[2] = analysis_start + vari.BINSIZE1 * idx + vari.BINSIZE1 - vari.BINSIZE1
			definedRegion.append(regionVector)
			numRegion = numRegion + 1
			
			## Start a new reigon
			regionVector = [analysis_chromo, (analysis_start+vari.BINSIZE1*idx), (analysis_start+vari.BINSIZE1*idx+vari.BINSIZE1), currGroupType]
		else:
			regionVector[2] = analysis_start + vari.BINSIZE1*idx + vari.BINSIZE1

		if(idx == (binNum-1)):
			regionVector[2] = analysis_end
			definedRegion.append(regionVector)
			numRegion = numRegion + 1
			break
	
		pastGroupType = currGroupType
		idx = idx + 1


	### variance check
	CTRLBW = [0] * vari.CTRLBW_NUM
	EXPBW = [0] * vari.EXPBW_NUM

	for rep in range(vari.CTRLBW_NUM): 
		CTRLBW[rep] = pyBigWig.open(vari.CTRLBW_NAMES[rep])
	for rep in range(vari.EXPBW_NUM):
		EXPBW[rep] = pyBigWig.open(vari.EXPBW_NAMES[rep])

	if(len(definedRegion) == 0):
		return None


	deleteIdx = []
	for regionIdx in range(len(definedRegion)):
		chromo = definedRegion[regionIdx][0]
		start = int(definedRegion[regionIdx][1])
		end = int(definedRegion[regionIdx][2])

		rc = []
		for rep in range(vari.CTRLBW_NUM):
			rc_temp = np.nanmean(CTRLBW[rep].values(chromo, start, end))
			rc.extend([rc_temp])

		for rep in range(vari.EXPBW_NUM):
			rc_temp = np.nanmean(EXPBW[rep].values(chromo, start, end))
			rc.extend([rc_temp])

		region_var = np.nanvar(rc)
		
		if(np.isnan(region_var) == True):
			deleteIdx.extend([regionIdx])
			continue
		
		thetaIdx = np.max(np.where(vari.FILTER_CUTOFFS < region_var))
		theta = vari.FILTER_CUTOFFS_THETAS[thetaIdx]			

		definedRegion[regionIdx].extend([ theta ])
	
	definedRegion = np.delete(np.array(definedRegion), deleteIdx, axis=0)
	definedRegion = definedRegion.tolist()


	if(len(definedRegion) == 0):
		return None

	subfile = tempfile.NamedTemporaryFile(mode="w+t", dir=vari.OUTPUT_DIR, delete=False)
	for line in definedRegion:
		subfile.write('\t'.join([str(x) for x in line]) + "\n")
	subfile.close()

	for rep in range(vari.CTRLBW_NUM):
		CTRLBW[rep].close()
	for rep in range(vari.EXPBW_NUM):
		EXPBW[rep].close()

	return subfile.name




cpdef doWindowApproach(arg):
	warnings.simplefilter("ignore", category=RuntimeWarning)

	input_filename = arg
	input_stream = open(input_filename)
	input_file = input_stream.readlines()

	subfile = tempfile.NamedTemporaryFile(mode="w+t", dir=vari.OUTPUT_DIR, delete=False)	
	simes_p = []
	writtenRegionNum = 0

	for regionNum in range(len(input_file)): 
		temp = input_file[regionNum].split()
		regionChromo = temp[0]
		regionStart = int(temp[1])
		regionEnd = int(temp[2])
		regionType = int(temp[3])
		regionTheta = int(temp[4])
	
		totalRC = []
		for rep in range(vari.CTRLBW_NUM):
			bw = pyBigWig.open(vari.CTRLBW_NAMES[rep])
			temp = bw.values(regionChromo, regionStart, regionEnd)
			totalRC.append(temp)
			bw.close()
		
		for rep in range(vari.EXPBW_NUM):
			bw = pyBigWig.open(vari.EXPBW_NAMES[rep])
			temp = bw.values(regionChromo, regionStart, regionEnd)
			totalRC.append(temp)
			bw.close()

		totalRC = np.array(totalRC)
		binStart_idx = 0
		analysisEnd_idx = regionEnd - regionStart

		windowPvalue = []
		windowEnrich = []

		while( (binStart_idx + vari.BINSIZE2) <= analysisEnd_idx ):
			if( (binStart_idx + vari.SHIFTSIZE2 + vari.BINSIZE2) > analysisEnd_idx):
				binEnd_idx = analysisEnd_idx
			else:
				binEnd_idx = binStart_idx + vari.BINSIZE2

			rc = np.nanmean(totalRC[:,binStart_idx:binEnd_idx], axis=1)

			if(len(np.where(np.isnan(rc) == True)[0]) > 0):
				windowPvalue.extend([np.nan])
				windowEnrich.extend([np.nan])
				if( (binEnd_idx == analysisEnd_idx) and (len(windowPvalue) != 0) ):
					#### calculate a Simes' p value for the reigon
					windowPvalue = np.array(windowPvalue)
					windowPvalue_woNan = windowPvalue[np.isnan(windowPvalue)==False]
					if(len(windowPvalue_woNan) == 0):
						break
					rankPvalue = scipy.stats.rankdata(windowPvalue_woNan)
					numWindow = len(windowPvalue_woNan)
					p_merged = np.min((windowPvalue_woNan * numWindow) / rankPvalue)
					simes_p.extend([p_merged])

					regionInfo = [regionChromo, regionStart, regionEnd, regionTheta, p_merged]		
					subfile.write('\t'.join([str(x) for x in regionInfo]) + "\t")
					subfile.write(','.join([str(x) for x in windowPvalue]) + "\t")
					subfile.write(','.join([str(x) for x in windowEnrich]) + "\n")

				binStart_idx = binStart_idx + vari.SHIFTSIZE2
				continue

			rc = rc.tolist()
			if(rc == vari.ALL_ZERO):
				windowPvalue.extend([np.nan])
				windowEnrich.extend([np.nan])

				if( (binEnd_idx == analysisEnd_idx) and (len(windowPvalue) != 0) ):
					#### calculate a Simes' p value for the reigon
					windowPvalue = np.array(windowPvalue)
					windowPvalue_woNan = windowPvalue[np.isnan(windowPvalue)==False]
					if(len(windowPvalue_woNan) == 0):
						break
					rankPvalue = scipy.stats.rankdata(windowPvalue_woNan)
					numWindow = len(windowPvalue_woNan)
					p_merged = np.min((windowPvalue_woNan * numWindow) / rankPvalue)
					simes_p.extend([p_merged])

					regionInfo = [regionChromo, regionStart, regionEnd, regionTheta, p_merged]
					subfile.write('\t'.join([str(x) for x in regionInfo]) + "\t")
					subfile.write(','.join([str(x) for x in windowPvalue]) + "\t")
					subfile.write(','.join([str(x) for x in windowEnrich]) + "\n")
					writtenRegionNum = writtenRegionNum + 1

				binStart_idx = binStart_idx + vari.SHIFTSIZE2
				continue

			windowInfo = doStatTesting(rc)
			pvalue = float(windowInfo[1])
			enrich = int(windowInfo[0])
			windowPvalue.extend([pvalue])
			windowEnrich.extend([enrich])

			if( (binEnd_idx == analysisEnd_idx) and (len(windowPvalue) != 0) ):
				#### calculate a Simes' p value for the reigon
				windowPvalue = np.array(windowPvalue)
				windowPvalue_woNan = windowPvalue[np.isnan(windowPvalue)==False]
				if(len(windowPvalue_woNan) == 0):
					break
				rankPvalue = scipy.stats.rankdata(windowPvalue_woNan)
				numWindow = len(windowPvalue_woNan)
				p_merged = np.min((windowPvalue_woNan * numWindow) / rankPvalue)
				simes_p.extend([p_merged])

				regionInfo = [regionChromo, regionStart, regionEnd, regionTheta, p_merged]
				subfile.write('\t'.join([str(x) for x in regionInfo]) + "\t")
				subfile.write(','.join([str(x) for x in windowPvalue]) + "\t")
				subfile.write(','.join([str(x) for x in windowEnrich]) + "\n")
				writtenRegionNum = writtenRegionNum + 1

			binStart_idx = binStart_idx + vari.SHIFTSIZE2

	subfile.close()

	os.remove(input_filename)

	if(writtenRegionNum == 0):
		os.remove(subfile.name)
		return None
	else:
		return subfile.name


cpdef doStatTesting(rc):
	ctrl_rc = []
	exp_rc = []

	for rep in range(vari.CTRLBW_NUM):
		rc[rep] = float(rc[rep])
		ctrl_rc.extend([rc[rep]])

	for rep in range(vari.EXPBW_NUM):
		rc[rep+vari.CTRLBW_NUM] = float(rc[rep+vari.CTRLBW_NUM])
		exp_rc.extend([rc[rep+vari.CTRLBW_NUM] ])
	
	ctrlVar = np.nanvar(ctrl_rc)
	expVar = np.nanvar(exp_rc)


	if( (ctrlVar == 0) and (expVar == 0)):
		statistics = float(np.nanmean(exp_rc) - np.nanmean(ctrl_rc))

		pvalue = scipy.stats.norm.cdf(statistics, loc=0, scale=vari.NULL_STD)

		if(pvalue > 0.5):
			pvalue = (1 - pvalue) * 2
			enrich = 1
		else:
			pvalue = pvalue * 2
			enrich = -1

	elif( (ctrlVar == 0) and (expVar != 0) ):
		loc = np.nanmean(ctrl_rc)
		tResult = scipy.stats.ttest_1samp(exp_rc, popmean=loc)

		if(tResult.statistic > 0):
			enrich = 1
		else:
			enrich = -1

		pvalue = tResult.pvalue

	elif( (ctrlVar != 0) and (expVar == 0) ):
		loc = np.nanmean(exp_rc)
		tResult = scipy.stats.ttest_1samp(ctrl_rc, popmean=loc)

		if(tResult.statistic > 0):
			enrich = -1
		else:
			enrich = 1
		pvalue = tResult.pvalue

	else:
		welchResult = scipy.stats.ttest_ind(ctrl_rc, exp_rc, equal_var=False)

		if(welchResult.statistic > 0):
			enrich = -1
		else:
			enrich = 1

		pvalue = welchResult.pvalue


	windowInfo = [enrich, pvalue]	

	return windowInfo	



cpdef doFDRprocedure(args):
	input_filename = args[0]
	selectRegionIdx = args[1]

	input_stream = open(input_filename)
	input_file = input_stream.readlines()

	subfile = tempfile.NamedTemporaryFile(mode="w+t", dir=vari.OUTPUT_DIR, delete=False)
	
	### open bw files to store diff value
	ctrlBW = [0] * vari.CTRLBW_NUM
	expBW = [0] * vari.EXPBW_NUM

	for i in range(vari.CTRLBW_NUM):
		ctrlBW[i] = pyBigWig.open(vari.CTRLBW_NAMES[i])
	for i in range(vari.EXPBW_NUM):
		expBW[i] = pyBigWig.open(vari.EXPBW_NAMES[i])

	for regionIdx in selectRegionIdx:
		regionInfo = input_file[regionIdx].split()
		regionChromo = regionInfo[0]
		regionStart = int(regionInfo[1])		
		regionEnd = int(regionInfo[2])
		windowPvalue = list(map(float, regionInfo[5].split(",")))
		windowEnrich = list(map(float, regionInfo[6].split(",")))
		windowNum = len(windowPvalue)

		windowPvalue = np.array(windowPvalue)
		PVALUE_region_bh = np.array([0.0] * len(windowPvalue))
		nanIdx = np.where(np.isnan(windowPvalue)==True)
		PVALUE_region_bh[nanIdx] = np.nan

		nonNanIdx = np.where(np.isnan(windowPvalue)==False)
		windowPvalue_temp = windowPvalue[nonNanIdx]

		PVALUE_region_bh_temp = statsmodels.sandbox.stats.multicomp.multipletests(windowPvalue_temp, alpha=vari.ADJ_FDR, method='fdr_bh')[0]
		for nonIdx in range(len(nonNanIdx[0])):
			PVALUE_region_bh[nonNanIdx[0][nonIdx]] = PVALUE_region_bh_temp[nonIdx]

		PVALUE_region_bh = np.array(PVALUE_region_bh)
		selectWindowIdx = np.where(PVALUE_region_bh== True)[0]

		if(len(selectWindowIdx) == 0):
			continue

		### merge if the windows are overlapping and 'enrich' are the same
		idx = selectWindowIdx[0]
		pastStart = regionStart + idx * vari.SHIFTSIZE2
		pastEnd = pastStart + vari.BINSIZE2
		pastPvalue = windowPvalue[idx]
		pastEnrich = windowEnrich[idx]
		pastPvalueSets = [pastPvalue]

		selectWindowVector = [regionChromo, pastStart, pastEnd, pastEnrich]
	
		lastIdx = selectWindowIdx[len(selectWindowIdx)-1]

		if(lastIdx == selectWindowIdx[0]):
			selectWindowVector.extend([ pastPvalue ])

			if(lastIdx == (windowNum-1)):
				pastEnd = regionEnd
				selectWindowVector[2] = pastEnd
			
			ctrlRC = []
			for rep in range(vari.CTRLBW_NUM):
				ctrlRC.append(ctrlBW[rep].values(selectWindowVector[0], selectWindowVector[1], selectWindowVector[2]))
			ctrlRC = np.array(ctrlRC)
			ctrlRC_posMean = np.mean(ctrlRC, axis=0)

			expRC = []
			for rep in range(vari.EXPBW_NUM):
				expRC.append(expBW[rep].values(selectWindowVector[0], selectWindowVector[1], selectWindowVector[2]))
			expRC = np.array(expRC)
			expRC_posMean = np.mean(expRC, axis=0)

			diff_pos = expRC_posMean - ctrlRC_posMean
			diff_pos = np.array(diff_pos)
			diff_pos_nanNum = len(np.where(np.isnan(diff_pos)==True)[0])
			if(diff_pos_nanNum == len(diff_pos)):
				continue

			subPeakStarts, subPeakEnds, subPeakDiffs = truncateNan(selectWindowVector[1], selectWindowVector[2], diff_pos)

			for subPeakNum in range(len(subPeakStarts)):
				temp = list(selectWindowVector)
				temp[1] = subPeakStarts[subPeakNum]
				temp[2] = subPeakEnds[subPeakNum]
				temp[3] = int(temp[3])
				temp.extend([ subPeakDiffs[subPeakNum]  ])

				testResult = testSubPeak(temp, selectWindowVector[3])
				if(testResult == True):
					subfile.write('\t'.join([str(x) for x in temp]) + "\n")

			continue


		for idx in selectWindowIdx[1:]:
			currStart = regionStart + idx * vari.SHIFTSIZE2	
			currEnd = currStart + vari.BINSIZE2
			currPvalue = windowPvalue[idx]
			currEnrich = windowEnrich[idx]

			if((currStart >= pastStart) and (currStart <= pastEnd) and (pastEnrich == currEnrich)):
				selectWindowVector[2] = currEnd
				pastPvalueSets.extend([currPvalue])
			else:
				### End a previous region
				selectWindowVector[2] = pastEnd
				selectWindowVector.extend([ np.min(pastPvalueSets) ])

				ctrlRC = []
				for rep in range(vari.CTRLBW_NUM):
					ctrlRC.append(ctrlBW[rep].values(selectWindowVector[0], selectWindowVector[1], selectWindowVector[2]))
				ctrlRC = np.array(ctrlRC)
				ctrlRC_posMean = np.mean(ctrlRC, axis=0)

				expRC = []
				for rep in range(vari.EXPBW_NUM):
					expRC.append(expBW[rep].values(selectWindowVector[0], selectWindowVector[1], selectWindowVector[2]))
				expRC = np.array(expRC)
				expRC_posMean = np.mean(expRC, axis=0)

				diff_pos = expRC_posMean - ctrlRC_posMean
				diff_pos = np.array(diff_pos)
				diff_pos_nanNum = len(np.where(np.isnan(diff_pos)==True)[0])
				if(diff_pos_nanNum == len(diff_pos)):				
					## stark a new region
					pastPvalueSets = [currPvalue]
					selectWindowVector = [regionChromo, currStart, currEnd, currEnrich]

					pastStart = currStart
					pastEnd = currEnd				
					pastEnrich = currEnrich
					continue
	
				subPeakStarts, subPeakEnds, subPeakDiffs = truncateNan(selectWindowVector[1], selectWindowVector[2], diff_pos)
		
				for subPeakNum in range(len(subPeakStarts)):
					temp = list(selectWindowVector)					
					temp[1] = subPeakStarts[subPeakNum]
					temp[2] = subPeakEnds[subPeakNum]
					temp[3] = int(temp[3])
					temp.extend([ subPeakDiffs[subPeakNum]  ])

					testResult = testSubPeak(temp, selectWindowVector[3])
					if(testResult == True):
						subfile.write('\t'.join([str(x) for x in temp]) + "\n")

				### Start a new region
				pastPvalueSets = [currPvalue]
				selectWindowVector = [regionChromo, currStart, currEnd, currEnrich]

			if(idx == lastIdx):
				if(lastIdx == (windowNum-1)):
					pastEnd = regionEnd								
					selectWindowVector[2] = pastEnd

				selectWindowVector.extend([ np.min(pastPvalueSets) ])

				ctrlRC = []
				for rep in range(vari.CTRLBW_NUM):
					ctrlRC.append(ctrlBW[rep].values(selectWindowVector[0], selectWindowVector[1], selectWindowVector[2]))
				ctrlRC = np.array(ctrlRC)
				ctrlRC_posMean = np.mean(ctrlRC, axis=0)
	
				expRC = []
				for rep in range(vari.EXPBW_NUM):
					expRC.append(expBW[rep].values(selectWindowVector[0], selectWindowVector[1], selectWindowVector[2]))
				expRC = np.array(expRC)
				expRC_posMean = np.mean(expRC, axis=0)

				diff_pos = expRC_posMean - ctrlRC_posMean
				diff_pos = np.array(diff_pos)
				diff_pos_nanNum = len(np.where(np.isnan(diff_pos)==True)[0])
				if(diff_pos_nanNum == len(diff_pos)):	
					break
			
				subPeakStarts, subPeakEnds, subPeakDiffs = truncateNan(selectWindowVector[1], selectWindowVector[2], diff_pos)

				for subPeakNum in range(len(subPeakStarts)):
					temp = list(selectWindowVector)
					temp[1] = subPeakStarts[subPeakNum]
					temp[2] = subPeakEnds[subPeakNum]
					temp[3] = int(temp[3])
					temp.extend([ subPeakDiffs[subPeakNum]  ])

					testResult = testSubPeak(temp, selectWindowVector[3])
					if(testResult == True):
						subfile.write('\t'.join([str(x) for x in temp]) + "\n")

				break

			pastStart = currStart
			pastEnd = currEnd
			pastEnrich = currEnrich

	subfile.close()

	for rep in range(vari.CTRLBW_NUM):
		ctrlBW[rep].close()
	for rep in range(vari.EXPBW_NUM):
		expBW[rep].close()
	
	os.remove(input_filename)

	return subfile.name

cpdef testSubPeak(subpeak, binEnrichType):
	diff = int(subpeak[5])
	
	if(diff == 0):
		return False
	if( (binEnrichType == 1) and (diff < 0)):
		return False
	if( (binEnrichType == -1) and (diff > 0)):
		return False

	return True


cpdef truncateNan(peakStart, peakEnd, diff_pos):	
	
	idx = np.where(np.isnan(diff_pos)==False)[0]
	if(len(idx) == len(diff_pos)):
		peakDiff = int(np.round(np.mean(diff_pos)))
		return [peakStart], [peakEnd], [peakDiff]
	else:
		filteredIdx = []
		nanIdx = np.where(np.isnan(diff_pos)==True)[0]

		prevPosIdx = nanIdx[0]
		nanPosStartIdx = prevPosIdx
		i = 1
		while(i < len(nanIdx)):
			currPosIdx = nanIdx[i]

			if(currPosIdx == (prevPosIdx+1)):

				if(i == (len(nanIdx)-1)):
					strechLen = currPosIdx - nanPosStartIdx
					if(strechLen >= 20):   ## Save it to the filter out list
						filteredIdx.extend(list(range(nanPosStartIdx, (currPosIdx+1))))
						break
					if(nanPosStartIdx==0):  ## if np.nan stretch  exists in the beginning of the peak region
						filteredIdx.extend(list(range(nanPosStartIdx, (currPosIdx+1))))
						break
					if(currPosIdx == (len(diff_pos)-1)):   ### if np.nan strech exists in the end of the peak region
						filteredIdx.extend(list(range(nanPosStartIdx, (currPosIdx+1))))
						break
	
				prevPosIdx = currPosIdx
				i = i + 1	
			else:
				#### End a subfiltered
				strechLen = prevPosIdx - nanPosStartIdx
				if(strechLen >= 20):   ## Save it to the filter out list
					filteredIdx.extend(list(range(nanPosStartIdx, (prevPosIdx+1))))
				if(nanPosStartIdx==0):
					filteredIdx.extend(list(range(nanPosStartIdx, (prevPosIdx+1))))
		
				if( (i == (len(nanIdx)-1)) and (currPosIdx == (len(diff_pos)-1))):
					filteredIdx.extend([currPosIdx])
					break
		
				prevPosIdx = currPosIdx
				nanPosStartIdx = currPosIdx
				i = i + 1

		##### Get subpeak regions
		if(len(filteredIdx) > 0):
			totalPosIdx = np.array(list(range(len(diff_pos))))
			totalPosIdx = np.delete(totalPosIdx, filteredIdx)

			subPeakStarts = []
			subPeakEnds = []
			subPeakDiffs = []

			prevPosIdx = totalPosIdx[0]
			subPeakStartIdx = prevPosIdx
			i = 1
			while(i < len(totalPosIdx)):
				currPosIdx = totalPosIdx[i]

				if(i == (len(totalPosIdx)-1)):
					subPeakEndIdx = currPosIdx + 1
					subPeakDiff = int(np.round(np.nanmean(diff_pos[subPeakStartIdx:subPeakEndIdx])))

					subPeakStarts.extend([ subPeakStartIdx + peakStart ])
					subPeakEnds.extend([ subPeakEndIdx + peakStart ])
					subPeakDiffs.extend([ subPeakDiff  ])
					break

				if(currPosIdx == (prevPosIdx+1)):
					prevPosIdx = currPosIdx
					i = i + 1
				else:
					#### End a region
					subPeakEndIdx = prevPosIdx + 1
					subPeakDiff = int(np.round(np.nanmean(diff_pos[subPeakStartIdx:subPeakEndIdx])))
					subPeakStarts.extend([ subPeakStartIdx + peakStart ])
					subPeakEnds.extend([ subPeakEndIdx + peakStart ])
					subPeakDiffs.extend([ subPeakDiff  ])

					### Start a new region
					subPeakStartIdx = currPosIdx
					prevPosIdx = currPosIdx
					i = i + 1
			
			return subPeakStarts, subPeakEnds, subPeakDiffs

		else:		
			peakDiff = int(np.round(np.nanmean(diff_pos)))
			return [peakStart], [peakEnd], [peakDiff]



cpdef selectTheta(metaDataName):
	input_filename = metaDataName
	input_stream = open(input_filename)
	input_file = input_stream.readlines()

	totalRegionNum_array = []
	selectRegionNum_array = []

	for thetaIdx in range(len(vari.FILTER_CUTOFFS_THETAS)):
		theta = int(vari.FILTER_CUTOFFS_THETAS[thetaIdx])	

		PVALUE_simes = []

		for subFileIdx in range(len(input_file)):
			subfile_name = input_file[subFileIdx].split()[0]
			subfile_stream = open(subfile_name)
			subfile_file = subfile_stream.readlines()

			for regionIdx in range(len(subfile_file)):
				line = subfile_file[regionIdx].split()
				regionTheta = int(line[3]) 
				regionPvalue = float(line[4])

				if(np.isnan(regionPvalue) == True):
					continue

				if(regionTheta >= theta):
					PVALUE_simes.extend([ regionPvalue ])

		totalRegionNum =  len(PVALUE_simes)
		PVALUE_group_bh = statsmodels.sandbox.stats.multicomp.multipletests(PVALUE_simes, alpha=vari.FDR, method='fdr_bh')
		selectRegionNum = len(np.where(PVALUE_group_bh[0] == True)[0])		

		totalRegionNum_array.extend([totalRegionNum])
		selectRegionNum_array.extend([selectRegionNum])

	selectRegionNum_array = np.array(selectRegionNum_array)
	maxNum = np.max(selectRegionNum_array)
	idx = np.where(selectRegionNum_array == maxNum)
	idx = idx[0][0]

	return [vari.FILTER_CUTOFFS_THETAS[idx], selectRegionNum_array[idx], totalRegionNum_array[idx]]




